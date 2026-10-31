"""Residual equations for the simplified Cartesian WRF PINN model.

This module implements the reduced moist system with diagnostic SGS closure described in
``wrf_pinn.config.physics``:

- local Cartesian coordinates
- zero external forcing & Coriolis
- predicted dry density and hydrostatic density inferred from d(p_EOS - p_prime)/dz
- q_l and f_cond = 0

The residuals are evaluated at continuous PINN collocation points. Inputs are
expected in coordinate order (x, y, z, t), and state outputs are expected in
physics-variable order (u, v, w, theta, p_prime, q_v, e_sgs, rho_d). Coordinates and state outputs may be
normalized; residual scaling maps them back to physical units before the PDE
terms are assembled.
Eight state variables are predicted, but only six PDE residuals are returned.
SGS TKE enters stresses and diffusivities without a separate transport residual.
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
    """Return u, v, w, theta, p_prime, q_v, e_sgs, and density columns from the model state output."""

    if state.ndim != 2 or state.shape[1] != physics.state_dim:
        raise ValueError(f"state must have shape (n_points, {physics.state_dim}).")

    return tuple( state[:, i:i + 1] for i in range(physics.state_dim) )


def _to_physical_state(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    theta: torch.Tensor,
    p_prime: torch.Tensor,
    q_v: torch.Tensor,
    e_sgs: torch.Tensor,
    rho_d: torch.Tensor,
    scaling: ResidualScalingConfig,
) -> tuple[torch.Tensor, ...]:
    """Map normalized state variables to physical state variables."""
    return (
        scaling.u.offset + scaling.u.scale * u,
        scaling.v.offset + scaling.v.scale * v,
        scaling.w.offset + scaling.w.scale * w,
        scaling.theta.offset + scaling.theta.scale * theta,
        scaling.p_prime.offset + scaling.p_prime.scale * p_prime,
        scaling.q_v.offset + scaling.q_v.scale * q_v,
        scaling.e_sgs.offset + scaling.e_sgs.scale * e_sgs,
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
    """Ensure this implementation is used only for the supported reduced system."""

    if physics.coordinate_system != "local_cartesian":
        raise ValueError("Only local_cartesian coordinates are supported.")

    if physics.active_variables != ("u", "v", "w", "theta", "p_prime", "q_v", "e_sgs", "rho_d"):
        raise ValueError("Residuals currently require active variables u, v, w, theta, p_prime, q_v, e_sgs, rho_d")

    expected_residuals = ("mass", "x_momentum", "y_momentum", "z_momentum", "potential_temperature", "water_vapor")
    if physics.residuals != expected_residuals:
        raise ValueError(f"Residuals require equations {expected_residuals}.")

    if not physics.forcing_is_zero:
        raise ValueError("Residuals currently assume zero forcing.")

#These variables validate that PhysicsConfig describes the same equations that this code actually calculates.
#  They do not calculate any physics; they are safety checks executed before residual assembly.
    required_terms: Mapping[str, bool] = {
        "include_gravity": physics.include_gravity,
        "include_pressure_gradient": physics.include_pressure_gradient,
        "include_temperature": physics.include_temperature,
        "include_moisture": physics.include_moisture,
        "include_turbulence": physics.include_turbulence,
    } # all must be true
    disabled_terms = [name for name, enabled in required_terms.items() if not enabled]
    if disabled_terms:
        raise ValueError(f"Required physics terms are disabled: {disabled_terms}.")

    unsupported_terms: Mapping[str, bool] = {
        "include_coriolis": physics.include_coriolis,
        "include_microphysics": physics.include_microphysics,
    } # even if turned on, not supported.
    enabled_terms = [name for name, enabled in unsupported_terms.items() if enabled]
    if enabled_terms:
        raise ValueError(f"Unsupported physics terms are enabled: enabled terms: {enabled_terms}.")

def _diagnose_moist_state(theta: torch.Tensor, q_v: torch.Tensor, rho_d: torch.Tensor,
    physics: PhysicsConfig) -> tuple[torch.Tensor | Jet, ...]:
    """Calculate thermodynamics from predicted dry density.

    Inputs are in physical units, except q_v retains the dataset's
    mixing-ratio units until converted locally to kg/kg.

    ``q_v`` is the water-vapor mixing ratio in the data's units; it is taken
    to kg/kg with ``physics.q_v_to_kg_per_kg``.
    """

    rd = physics.constants.dry_air_gas_constant
    rv = physics.constants.water_vapor_gas_constant
    p0 = physics.constants.reference_pressure
    kappa = physics.constants.kappa
    gamma = physics.constants.gamma

    q_v_kg = q_v * physics.q_v_to_kg_per_kg
    theta_m = theta * (1.0 + (rv / rd) * q_v_kg)

    # Total pressure and moist potential temperature.
    pressure = (rd * rho_d * theta_m / p0**kappa).pow(gamma)
    temperature = theta * (pressure / p0).pow(kappa)

    # The supplied FE dataset has q_l = 0.
    rho_m = rho_d * (1.0 + q_v_kg)
    density_ratio = rho_d / rho_m

    return pressure, temperature, rho_m, density_ratio

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
    filter_width: torch.Tensor,
) -> TensorDict:
    """Compute six moist Cartesian residuals with TKE-dependent SGS closure.

    Parameters
    ----------
    coordinates:
        Collocation coordinates with shape ``(n_points, 4)`` and column order
        ``(x, y, z, t)``. The tensor must have ``requires_grad=True``.
    state:
        Model outputs with shape ``(n_points, 8)`` and column order
        ``(u, v, w, theta, p', q_v, e_sgs, rho_d)``.
    physics:
        Physics configuration. Only the default reduced configuration is
        supported by this implementation.
    scaling:
        Affine maps from normalized coordinates/state variables to physical
        coordinates/state variables. Identity scaling preserves the old
        behavior.

    Returns
    -------
    dict[str, torch.Tensor]
        Residual tensors keyed by ``mass``, ``x_momentum``, ``y_momentum``, ``z_momentum``,
        ''potential_temperature'', and "water vapor" Each tensor has shape ``(n_points, 1)``.
    """

    _validate_physics_config(physics)

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_points, 4).")

    if not coordinates.requires_grad:
        raise ValueError("coordinates must have requires_grad=True.")

    u, v, w, theta, p_prime, q_v, e_sgs, rho_d = _to_physical_state(
        *_split_state(state, physics), scaling=scaling,)

    pressure, _temperature, rho_m, density_ratio = _diagnose_moist_state(
        theta, q_v, rho_d, physics)

    # Infer base-state density from the physical vertical pressure gradient.
    p_d_h = pressure - p_prime
    grad_p_d_h = _physical_gradient(p_d_h, coordinates, scaling)
    rho_d_h = -grad_p_d_h[:, 2:3] / physics.constants.gravity
    rho_m_prime = rho_m - rho_d_h

    grad_u = _physical_gradient(u, coordinates, scaling)
    grad_v = _physical_gradient(v, coordinates, scaling)
    grad_w = _physical_gradient(w, coordinates, scaling)
    grad_theta = _physical_gradient(theta, coordinates, scaling)
    grad_q_v = _physical_gradient(q_v, coordinates, scaling)
    grad_p_prime = _physical_gradient(p_prime, coordinates, scaling)

    u_x, u_y, u_z, u_t = grad_u.split(1, dim=1)
    v_x, v_y, v_z, v_t = grad_v.split(1, dim=1)
    w_x, w_y, w_z, w_t = grad_w.split(1, dim=1)

    theta_x, theta_y, theta_z, _theta_t = grad_theta.split(1, dim=1)
    q_v_x, q_v_y, q_v_z, _q_v_t = grad_q_v.split(1, dim=1)
    p_prime_x, p_prime_y, p_prime_z, _p_prime_t = grad_p_prime.split(1, dim=1)

    # Diagnose the SGS coefficients from local theta_z and e_sgs.
    _, _, k_m, k_scalar = _diagnostic_sgs_coefficients(theta,
        theta_z, e_sgs, physics, filter_width=filter_width)

    # The current first-pass model uses the same SGS diffusivity for
    # potential temperature and water vapor.
    k_theta = k_scalar
    k_q_v = k_scalar

    # Dry-air mass and momentum storage/transport products. Only rho_d needs its own
    # backward pass; the product gradients follow from the product rule on the
    # gradients already computed (autograd would apply the same rule, at the cost
    # of one full backward pass through the network per product).
    grad_rho_d = _physical_gradient(rho_d, coordinates, scaling)
    rho = (rho_d, grad_rho_d)
    gu, gv, gw = (u, grad_u), (v, grad_v), (w, grad_w)
    grad_rho_d_u = _product_gradient(rho, gu)
    grad_rho_d_v = _product_gradient(rho, gv)
    grad_rho_d_w = _product_gradient(rho, gw)

    _, _, _, rho_d_t = grad_rho_d.split(1, dim=1)
    rho_d_u_x, _, _, rho_d_u_t = grad_rho_d_u.split(1, dim=1)
    _, rho_d_v_y, _, rho_d_v_t = grad_rho_d_v.split(1, dim=1)
    _, _, rho_d_w_z, rho_d_w_t = grad_rho_d_w.split(1, dim=1)

    rho_d_uu_x = _product_gradient(rho, gu, gu)[:, 0:1]
    rho_d_vv_y = _product_gradient(rho, gv, gv)[:, 1:2]
    rho_d_ww_z = _product_gradient(rho, gw, gw)[:, 2:3]

    grad_rho_d_uv = _product_gradient(rho, gu, gv)
    grad_rho_d_uw = _product_gradient(rho, gu, gw)
    grad_rho_d_vw = _product_gradient(rho, gv, gw)

    rho_d_uv_x = grad_rho_d_uv[:, 0:1]
    rho_d_uv_y = grad_rho_d_uv[:, 1:2]
    rho_d_uw_x = grad_rho_d_uw[:, 0:1]
    rho_d_uw_z = grad_rho_d_uw[:, 2:3]
    rho_d_vw_y = grad_rho_d_vw[:, 1:2]
    rho_d_vw_z = grad_rho_d_vw[:, 2:3]

    # Conservative potential-temperature products (product rule, as above).
    gth, gqv = (theta, grad_theta), (q_v, grad_q_v)
    rho_d_theta_t = _product_gradient(rho, gth)[:, 3:4]
    rho_d_u_theta_x = _product_gradient(rho, gu, gth)[:, 0:1]
    rho_d_v_theta_y = _product_gradient(rho, gv, gth)[:, 1:2]
    rho_d_w_theta_z = _product_gradient(rho, gw, gth)[:, 2:3]

    # Conservative water-vapor products.
    rho_d_q_v_t = _product_gradient(rho, gqv)[:, 3:4]
    rho_d_u_q_v_x = _product_gradient(rho, gu, gqv)[:, 0:1]
    rho_d_v_q_v_y = _product_gradient(rho, gv, gqv)[:, 1:2]
    rho_d_w_q_v_z = _product_gradient(rho, gw, gqv)[:, 2:3]

    divergence = u_x + v_y + w_z
    # Physical SGS momentum stresses:
    # tau_ij = -rho_d K_m S_ij + (2/3) rho_d e_sgs delta_ij.
    tau_xx = -rho_d * k_m * (2.0 * u_x - (2.0 / 3.0) * divergence) + (2.0 / 3.0) * rho_d * e_sgs
    tau_yy = -rho_d * k_m * (2.0 * v_y - (2.0 / 3.0) * divergence) + (2.0 / 3.0) * rho_d * e_sgs
    tau_zz = -rho_d * k_m * (2.0 * w_z - (2.0 / 3.0) * divergence) + (2.0 / 3.0) * rho_d * e_sgs
    tau_xy = -rho_d * k_m * (u_y + v_x)
    tau_xz = -rho_d * k_m * (u_z + w_x)
    tau_yz = -rho_d * k_m * (v_z + w_y)


    grad_tau_xx = _physical_gradient(tau_xx, coordinates, scaling)
    grad_tau_yy = _physical_gradient(tau_yy, coordinates, scaling)
    grad_tau_zz = _physical_gradient(tau_zz, coordinates, scaling)
    grad_tau_xy = _physical_gradient(tau_xy, coordinates, scaling)
    grad_tau_xz = _physical_gradient(tau_xz, coordinates, scaling)
    grad_tau_yz = _physical_gradient(tau_yz, coordinates, scaling)

    tau_xx_x = grad_tau_xx[:, 0:1]
    tau_yy_y = grad_tau_yy[:, 1:2]
    tau_zz_z = grad_tau_zz[:, 2:3]
    tau_xy_x = grad_tau_xy[:, 0:1]
    tau_xy_y = grad_tau_xy[:, 1:2]
    tau_xz_x = grad_tau_xz[:, 0:1]
    tau_xz_z = grad_tau_xz[:, 2:3]
    tau_yz_y = grad_tau_yz[:, 1:2]
    tau_yz_z = grad_tau_yz[:, 2:3]

    # SGS potential-temperature and water-vapor fluxes.
    tau_theta_x = -rho_d * k_theta * theta_x
    tau_theta_y = -rho_d * k_theta * theta_y
    tau_theta_z = -rho_d * k_theta * theta_z

    grad_theta_x = _physical_gradient(tau_theta_x, coordinates, scaling)
    grad_theta_y = _physical_gradient(tau_theta_y, coordinates, scaling)
    grad_theta_z = _physical_gradient(tau_theta_z, coordinates, scaling)

    tau_theta_x_x = grad_theta_x[:, 0:1]
    tau_theta_y_y = grad_theta_y[:, 1:2]
    tau_theta_z_z = grad_theta_z[:, 2:3]

    tau_q_v_x = -rho_d * k_q_v * q_v_x
    tau_q_v_y = -rho_d * k_q_v * q_v_y
    tau_q_v_z = -rho_d * k_q_v * q_v_z

    grad_tau_q_v_x = _physical_gradient(tau_q_v_x, coordinates, scaling)
    grad_tau_q_v_y = _physical_gradient(tau_q_v_y, coordinates, scaling)
    grad_tau_q_v_z = _physical_gradient(tau_q_v_z, coordinates, scaling)

    tau_q_v_x_x = grad_tau_q_v_x[:, 0:1]
    tau_q_v_y_y = grad_tau_q_v_y[:, 1:2]
    tau_q_v_z_z = grad_tau_q_v_z[:, 2:3]

    # Loss
    mass = rho_d_t + rho_d_u_x + rho_d_v_y + rho_d_w_z

    x_momentum = (
        rho_d_u_t + rho_d_uu_x + rho_d_uv_y + rho_d_uw_z  
        + density_ratio*p_prime_x + tau_xx_x + tau_xy_y + tau_xz_z
    )

    y_momentum = (
        rho_d_v_t + rho_d_uv_x + rho_d_vv_y + rho_d_vw_z
        + density_ratio*p_prime_y + tau_xy_x + tau_yy_y + tau_yz_z
    )

    z_momentum = (
        rho_d_w_t + rho_d_uw_x + rho_d_vw_y + rho_d_ww_z
        + density_ratio*(p_prime_z + physics.constants.gravity * rho_m_prime) 
        + tau_xz_x + tau_yz_y + tau_zz_z
    )

    potential_temperature = (
        rho_d_theta_t + rho_d_u_theta_x + rho_d_v_theta_y + rho_d_w_theta_z
        + tau_theta_x_x + tau_theta_y_y + tau_theta_z_z
    )

    water_vapor = (
        rho_d_q_v_t + rho_d_u_q_v_x + rho_d_v_q_v_y + rho_d_w_q_v_z
        + tau_q_v_x_x + tau_q_v_y_y + tau_q_v_z_z
    )

    return {
        "mass": mass,
        "x_momentum": x_momentum,
        "y_momentum": y_momentum,
        "z_momentum": z_momentum,
        "potential_temperature": potential_temperature,
        "water_vapor": water_vapor,
    }


def cartesian_zero_forcing_residuals_taylor(
    net: torch.nn.Sequential,
    coordinates: torch.Tensor,
    z_rows: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
    sizes: "torch.Tensor | int | None" = None,
    *,
    output_config: ConditionalModelConfig,
    filter_width: Jet,
) -> TensorDict:
    """The residuals of ``cartesian_zero_forcing_residuals``, without nested autograd.

    Same equations, term for term. Instead of evaluating the network and then taking
    one ``autograd.grad`` per field (19 backward passes, each differentiated again by
    training), the network's coordinate derivatives come from one forward Taylor
    pass (``taylor_forward``) and every pointwise quantity is a ``Jet`` carrying its
    gradient, so products, the equation of state and the SGS stresses differentiate
    themselves. ``net`` is the Linear/Tanh MLP fed ``[coordinates | z_rows]``
    (``ConditionalModel.net``); ``coordinates`` need not require grad. With
    ``sizes``, ``z_rows`` holds one latent row per case (``ConditionalModel.forward``).
    """

    _validate_physics_config(physics)

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_points, 4).")

    out = taylor_forward(net, coordinates, z_rows, scaling.coordinate_scales(), sizes=sizes,
                            output_config=output_config)
    if out.v.shape[1] != physics.state_dim:
        raise ValueError(
            f"Expected {physics.state_dim} state columns; got {out.v.shape[1]}.")

    def physical(name: str) -> Jet:
        j = out.column(physics.variable_index(name))
        s = getattr(scaling, name)
        return Jet(s.offset + s.scale * j.v, s.scale * j.d, s.scale * j.h)

    u, v, w, theta, p_prime, q_v, e_sgs, rho_d = (
        physical(name) for name in physics.active_variables)

    pressure, _temperature, rho_m, density_ratio = _diagnose_moist_state(
        theta, q_v, rho_d, physics)

    p_d_h = pressure - p_prime
    rho_d_h = -p_d_h.d[2] / physics.constants.gravity

    # Only the buoyancy value is needed, not its coordinate derivatives.
    rho_m_prime = rho_m.v - rho_d_h

    p_prime_x, p_prime_y, p_prime_z, _p_prime_t = p_prime.d

    # Diagnose the SGS coefficients from local theta_z and e_sgs (jets in x, y, z,
    # so their gradients carry theta's second derivatives).
    _n_squared, _mixing_length, k_m, k_scalar = _diagnostic_sgs_coefficients(
        theta, theta.deriv(2), e_sgs, physics, filter_width=filter_width)
    k_theta = k_scalar
    k_q_v = k_scalar

    # Dry-air mass and momentum storage/transport products.
    rho_d_u, rho_d_v, rho_d_w = rho_d * u, rho_d * v, rho_d * w
    rho_d_t = rho_d.d[3]
    rho_d_u_x, rho_d_u_t = rho_d_u.d[0], rho_d_u.d[3]
    rho_d_v_y, rho_d_v_t = rho_d_v.d[1], rho_d_v.d[3]
    rho_d_w_z, rho_d_w_t = rho_d_w.d[2], rho_d_w.d[3]

    rho_d_uu_x = (rho_d_u * u).d[0]
    rho_d_vv_y = (rho_d_v * v).d[1]
    rho_d_ww_z = (rho_d_w * w).d[2]
    rho_d_uv, rho_d_uw, rho_d_vw = rho_d_u * v, rho_d_u * w, rho_d_v * w
    rho_d_uv_x, rho_d_uv_y = rho_d_uv.d[0], rho_d_uv.d[1]
    rho_d_uw_x, rho_d_uw_z = rho_d_uw.d[0], rho_d_uw.d[2]
    rho_d_vw_y, rho_d_vw_z = rho_d_vw.d[1], rho_d_vw.d[2]

    # Conservative potential-temperature and water-vapor products.
    rho_d_theta_t = (rho_d * theta).d[3]
    rho_d_u_theta_x = (rho_d_u * theta).d[0]
    rho_d_v_theta_y = (rho_d_v * theta).d[1]
    rho_d_w_theta_z = (rho_d_w * theta).d[2]
    rho_d_q_v_t = (rho_d * q_v).d[3]
    rho_d_u_q_v_x = (rho_d_u * q_v).d[0]
    rho_d_v_q_v_y = (rho_d_v * q_v).d[1]
    rho_d_w_q_v_z = (rho_d_w * q_v).d[2]

    # Physical SGS momentum stresses, as jets of the velocity derivatives:
    # tau_ij = -rho_d K_m S_ij + (2/3) rho_d e_sgs delta_ij.
    u_x, u_y, u_z = u.deriv(0), u.deriv(1), u.deriv(2)
    v_x, v_y, v_z = v.deriv(0), v.deriv(1), v.deriv(2)
    w_x, w_y, w_z = w.deriv(0), w.deriv(1), w.deriv(2)
    divergence = u_x + v_y + w_z
    tau_xx = -rho_d * k_m * (2.0 * u_x - (2.0 / 3.0) * divergence) + (2.0 / 3.0) * rho_d * e_sgs
    tau_yy = -rho_d * k_m * (2.0 * v_y - (2.0 / 3.0) * divergence) + (2.0 / 3.0) * rho_d * e_sgs
    tau_zz = -rho_d * k_m * (2.0 * w_z - (2.0 / 3.0) * divergence) + (2.0 / 3.0) * rho_d * e_sgs
    tau_xy = -rho_d * k_m * (u_y + v_x)
    tau_xz = -rho_d * k_m * (u_z + w_x)
    tau_yz = -rho_d * k_m * (v_z + w_y)

    tau_xx_x, tau_yy_y, tau_zz_z = tau_xx.d[0], tau_yy.d[1], tau_zz.d[2]
    tau_xy_x, tau_xy_y = tau_xy.d[0], tau_xy.d[1]
    tau_xz_x, tau_xz_z = tau_xz.d[0], tau_xz.d[2]
    tau_yz_y, tau_yz_z = tau_yz.d[1], tau_yz.d[2]

    # SGS potential-temperature and water-vapor flux divergences.
    tau_theta_x_x = (-rho_d * k_theta * theta.deriv(0)).d[0]
    tau_theta_y_y = (-rho_d * k_theta * theta.deriv(1)).d[1]
    tau_theta_z_z = (-rho_d * k_theta * theta.deriv(2)).d[2]
    tau_q_v_x_x = (-rho_d * k_q_v * q_v.deriv(0)).d[0]
    tau_q_v_y_y = (-rho_d * k_q_v * q_v.deriv(1)).d[1]
    tau_q_v_z_z = (-rho_d * k_q_v * q_v.deriv(2)).d[2]

    ratio = density_ratio.v
    mass = rho_d_t + rho_d_u_x + rho_d_v_y + rho_d_w_z

    x_momentum = (
        rho_d_u_t + rho_d_uu_x + rho_d_uv_y + rho_d_uw_z
        + ratio * p_prime_x + tau_xx_x + tau_xy_y + tau_xz_z
    )

    y_momentum = (
        rho_d_v_t + rho_d_uv_x + rho_d_vv_y + rho_d_vw_z
        + ratio * p_prime_y + tau_xy_x + tau_yy_y + tau_yz_z
    )

    z_momentum = (
        rho_d_w_t + rho_d_uw_x + rho_d_vw_y + rho_d_ww_z
        + ratio * (p_prime_z + physics.constants.gravity * rho_m_prime)
        + tau_xz_x + tau_yz_y + tau_zz_z
    )

    potential_temperature = (
        rho_d_theta_t + rho_d_u_theta_x + rho_d_v_theta_y + rho_d_w_theta_z
        + tau_theta_x_x + tau_theta_y_y + tau_theta_z_z
    )

    water_vapor = (
        rho_d_q_v_t + rho_d_u_q_v_x + rho_d_v_q_v_y + rho_d_w_q_v_z
        + tau_q_v_x_x + tau_q_v_y_y + tau_q_v_z_z
    )

    return {
        "mass": mass,
        "x_momentum": x_momentum,
        "y_momentum": y_momentum,
        "z_momentum": z_momentum,
        "potential_temperature": potential_temperature,
        "water_vapor": water_vapor,
    }
