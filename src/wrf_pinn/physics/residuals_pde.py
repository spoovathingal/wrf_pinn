"""Dry Cartesian PDE residuals without moisture or SGS transport.

State order: u, v, w, theta, p_prime, rho_d.
Residuals: mass, three momentum equations, potential temperature.

Physical fields are recovered from normalization before evaluation.
Dry EOS pressure and p_prime diagnose hydrostatic base density:
    rho_d_h = -d(p_EOS - p_prime)/dz / g
    rho_d_prime = rho_d - rho_d_h

Both autograd and forward-Taylor implementations evaluate the same
conservative equations. Coriolis and external forcing are excluded.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig
from wrf_pinn.physics.taylor import Jet, taylor_forward, where
from wrf_pinn.models.conditional import ConditionalModelConfig

TensorDict = dict[str, torch.Tensor]


def _gradient(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Return gradient of a scalar field with respect to all coordinates."""

    if field.ndim != 2 or field.shape[1] != 1:
        raise ValueError("field must have shape (n_points, 1).")

    gradient = torch.autograd.grad(
        field,
        coordinates,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]

    if gradient is None:
        gradient = torch.zeros_like(coordinates)

    return gradient


def _physical_gradient(
    field: torch.Tensor,
    coordinates: torch.Tensor,
    scaling: ResidualScalingConfig,
) -> torch.Tensor:
    """Return physical-coordinate gradient of a scalar field.

    Autograd differentiates with respect to the model coordinates supplied to
    the network. Those coordinates may be normalized. If

    ``x_physical = offset + scale * x_normalized``,

    then

    ``d(field) / d(x_physical) = d(field) / d(x_normalized) / scale``.
    """

    gradient = _gradient(field, coordinates)
    coordinate_scales = _coordinate_scale_tensor(
        scaling.coordinate_scales(), gradient.device, gradient.dtype)
    return gradient / coordinate_scales


def _product_gradient(*factors: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Gradient of a product of fields from each factor's (field, gradient) pair.

    ``grad(f_1 ... f_n) = sum_i (prod_{j != i} f_j) grad(f_i)``. Fields are (n, 1),
    gradients (n, n_coord), in the same (physical) coordinates."""
    fields = [f for f, _ in factors]
    total = None
    for i, (_, grad) in enumerate(factors):
        others = None
        for j, f in enumerate(fields):
            if j != i:
                others = f if others is None else others * f
        term = grad if others is None else others * grad
        total = term if total is None else total + term
    return total


@lru_cache(maxsize=32)
def _coordinate_scale_tensor(scales: tuple[float, ...], device: torch.device,
                             dtype: torch.dtype) -> torch.Tensor:
    """(1, n_coord) coordinate scales on ``device``, built once per (scales, device,
    dtype). Building it per call copied it host-to-device each time, and every such
    copy made the CPU wait for the GPU. Callers must not modify it in place."""
    return torch.tensor(scales, dtype=dtype, device=device).reshape(1, -1)



def _split_state(
    state: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, ...]:
    """Return u, v, w, theta, p_prime, and density columns from the model state output."""

    if state.ndim != 2 or state.shape[1] != physics.state_dim:
        raise ValueError(f"state must have shape (n_points, {physics.state_dim}).")

    return tuple( state[:, i:i + 1] for i in range(physics.state_dim) )


def _to_physical_state(
    u, v, w, theta, p_prime, rho_d,
    scaling: ResidualScalingConfig,
):
    """Convert the six normalized fields to physical units."""
    return (
        scaling.u.offset + scaling.u.scale * u,
        scaling.v.offset + scaling.v.scale * v,
        scaling.w.offset + scaling.w.scale * w,
        scaling.theta.offset + scaling.theta.scale * theta,
        scaling.p_prime.offset + scaling.p_prime.scale * p_prime,
        scaling.rho_d.offset + scaling.rho_d.scale * rho_d,
    )

# not used for now until k_m training turned back on
def _to_physical_eddy_viscosity(
    k_m_unconstrained: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    """Map the unconstrained network output to bounded physical K_m. 0< K_m <100"""

    k_m_min = physics.constants.eddy_viscosity_min
    k_m_max = physics.constants.eddy_viscosity_max

    return k_m_min + (k_m_max - k_m_min) * torch.sigmoid(
        k_m_unconstrained
    )

def _validate_physics_config(physics: PhysicsConfig) -> None:
    expected_variables = (
        "u", "v", "w", "theta", "p_prime", "rho_d",
    )
    expected_residuals = (
        "mass", "x_momentum", "y_momentum", "z_momentum", "potential_temperature",
    )

    if physics.coordinate_system != "local_cartesian":
        raise ValueError("Only local_cartesian coordinates are supported.")

    if physics.active_variables != expected_variables:
        raise ValueError(f"Expected active variables {expected_variables}.")

    if physics.residuals != expected_residuals:
        raise ValueError(f"Expected residuals {expected_residuals}.")

    if not physics.forcing_is_zero:
        raise ValueError("This system requires zero external forcing.")

    for name in (
        "include_gravity", "include_pressure_gradient", "include_temperature",
    ):
        if not getattr(physics, name):
            raise ValueError(f"{name} must be enabled.")

    for name in ("include_moisture", "include_turbulence",
        "include_microphysics", "include_coriolis"):
        if getattr(physics, name):
            raise ValueError(f"{name} must be disabled.")

def _dry_pressure(theta, rho_d, physics: PhysicsConfig):
    """Dry EOS pressure for either Tensor fields or first-order Jets."""
    c = physics.constants
    return (
        c.dry_air_gas_constant * rho_d * theta / c.reference_pressure**c.kappa
    ).pow(c.gamma)

def _brunt_vaisala_squared(theta: torch.Tensor, theta_z: torch.Tensor, physics: PhysicsConfig) -> torch.Tensor:
    """Return N^2 from local dry potential temperature. used for l_sgs"""

    theta_floor = torch.finfo(theta.dtype).tiny
    theta_safe = theta.clamp_min(theta_floor)

    return (physics.constants.gravity * theta_z / theta_safe)

def _sgs_delta(like, filter_width):
    """Validate and return the required local SGS filter width."""
    if filter_width is None:
        raise ValueError("A local filter_width from grid_geometry is required.")

    is_jet = isinstance(like, Jet)
    if isinstance(filter_width, Jet) != is_jet:
        raise TypeError(
            "Local filter_width must be a Jet for Taylor or a Tensor for autograd."
        )

    width = filter_width.v if is_jet else filter_width
    values = like.v if is_jet else like

    if not isinstance(width, torch.Tensor):
        raise TypeError("Local filter_width must contain a Tensor.")
    if width.shape != values.shape:
        raise ValueError("Local filter_width must match the scalar field shape.")
    if width.dtype != values.dtype or width.device != values.device:
        raise ValueError("Local filter_width must match field dtype and device.")
    if is_jet and filter_width.d.shape != like.d.shape:
        raise ValueError("Local filter_width must match the field's Jet derivative shape.")

    return filter_width

def _sgs_mixing_length(n_squared: torch.Tensor, e_sgs: torch.Tensor, physics: PhysicsConfig,
                        *, filter_width):
    """Return the SGS mixing length, preserving local-width derivatives."""

    delta = _sgs_delta(e_sgs, filter_width)
    minimum_length = physics.minimum_mixing_length

    numerical_floor = torch.finfo(e_sgs.dtype).tiny
    sqrt_e_sgs = (e_sgs.clamp_min(0.0) + numerical_floor).sqrt()

    # torch.where evaluates both branches, so N^2 must be clamped before
    # taking its square root even where the unstable branch is selected.
    stable_n = n_squared.clamp_min(numerical_floor).sqrt()
    stable_candidate = (physics.stable_mixing_length_coefficient * sqrt_e_sgs / stable_n)

    # = max(min(candidate, delta), minimum_length), since minimum_length <= delta
    stable_length = where(
        stable_candidate <= delta, stable_candidate, delta,
    ).clamp_min(minimum_length)

    return where(n_squared > 0.0, stable_length, delta)

def _diagnostic_sgs_coefficients(
    theta: torch.Tensor,
    theta_z: torch.Tensor,
    e_sgs: torch.Tensor,
    physics: PhysicsConfig,
    *,
    filter_width,
):
    """Diagnose N^2, mixing length, K_m, and scalar diffusivity."""
    delta = _sgs_delta(e_sgs, filter_width)
    n_squared = _brunt_vaisala_squared(theta, theta_z, physics)
    mixing_length = _sgs_mixing_length(n_squared, e_sgs, physics, filter_width=filter_width)

    numerical_floor = torch.finfo(e_sgs.dtype).tiny
    sqrt_e_sgs = (e_sgs.clamp_min(0.0) + numerical_floor).sqrt()

    k_m = physics.sgs_mixing_coefficient * mixing_length * sqrt_e_sgs
    k_scalar = k_m * (1.0 + 2.0 * mixing_length / delta)

    return (n_squared, mixing_length, k_m, k_scalar)

def cartesian_zero_forcing_residuals(
    coordinates: torch.Tensor,
    state: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
    *,
    filter_width: torch.Tensor | None = None,
) -> TensorDict:
    """Five dry, SGS-free residuals; filter_width is unused."""
    _validate_physics_config(physics)

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_points, 4).")

    if not coordinates.requires_grad:
        raise ValueError("coordinates must have requires_grad=True.")

    u, v, w, theta, p_prime, rho_d = _to_physical_state(
        *_split_state(state, physics), scaling=scaling
    )

    fields = (u, v, w, theta, p_prime, rho_d)
    grads = tuple(_physical_gradient(f, coordinates, scaling) for f in fields)

    gu, gv, gw, gtheta, gp, grho = grads

    rho = (rho_d, grho)
    velocities = ((u, gu), (v, gv), (w, gw))

    # Preserve the existing hydrostatic base-density diagnosis.
    pressure = _dry_pressure(theta, rho_d, physics)
    grad_base_pressure = _physical_gradient(
        pressure - p_prime, coordinates, scaling
    )
    g = physics.constants.gravity
    rho_d_h = -grad_base_pressure[:, 2:3] / g
    rho_d_prime = rho_d - rho_d_h

    def transport(field):
        # d(rho*field)/dt + sum_j d(rho*u_j*field)/dx_j.
        result = _product_gradient(rho, field)[:, 3:4]
        for j, velocity in enumerate(velocities):
            result = result + _product_gradient(
                rho, velocity, field
            )[:, j:j + 1]
        return result

    mass = grho[:, 3:4]
    for j, velocity in enumerate(velocities):
        mass = mass + _product_gradient(
            rho, velocity
        )[:, j:j + 1]

    return {
        "mass": mass,
        "x_momentum": transport((u, gu)) + gp[:, 0:1],
        "y_momentum": transport((v, gv)) + gp[:, 1:2],
        "z_momentum": (
            transport((w, gw)) + gp[:, 2:3] + g * rho_d_prime
        ),
        "potential_temperature": transport((theta, gtheta)),
    }

def cartesian_zero_forcing_residuals_taylor(
    net: torch.nn.Sequential,
    coordinates: torch.Tensor,
    z_rows: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
    sizes: "torch.Tensor | int | None" = None,
    out: "Jet | None" = None,
    *,
    output_config: ConditionalModelConfig,
    filter_width: "Jet | None" = None,
) -> TensorDict:
    """Same five dry residuals using first derivatives; filter_width is unused."""
    _validate_physics_config(physics)

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_points, 4).")

    if out is None:
        out = taylor_forward(
            net,
            coordinates,
            z_rows,
            scaling.coordinate_scales(),
            second=False,
            sizes=sizes,
            output_config=output_config,
        )

    if out.v.shape[1] != physics.state_dim:
        raise ValueError(f"Expected {physics.state_dim} state columns.")

    def physical(name):
        j = out.column(physics.variable_index(name))
        s = getattr(scaling, name)
        return Jet(
            s.offset + s.scale * j.v,
            s.scale * j.d,
        )

    u, v, w, theta, p_prime, rho_d = (
        physical(name) for name in physics.active_variables
    )

    pressure = _dry_pressure(theta, rho_d, physics)
    g = physics.constants.gravity
    rho_d_h = -(pressure - p_prime).d[2] / g
    rho_d_prime = rho_d.v - rho_d_h

    velocities = (u, v, w)

    def transport(field):
        result = (rho_d * field).d[3]
        for j, velocity in enumerate(velocities):
            result = result + (rho_d * velocity * field).d[j]
        return result

    mass = rho_d.d[3]
    for j, velocity in enumerate(velocities):
        mass = mass + (rho_d * velocity).d[j]

    return {
        "mass": mass,
        "x_momentum": transport(u) + p_prime.d[0],
        "y_momentum": transport(v) + p_prime.d[1],
        "z_momentum": (
            transport(w) + p_prime.d[2] + g * rho_d_prime
        ),
        "potential_temperature": transport(theta),
    }