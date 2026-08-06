"""Residual equations for the simplified Cartesian WRF PINN model.

This module implements the dry neutral boundary layer system described in
``wrf_pinn.config.physics``:

- local Cartesian coordinates
- zero external forcing
- no moisture equations
- active state variables are u, v, w, theta, p_prime

The residuals are evaluated at continuous PINN collocation points. Inputs are
expected in coordinate order (x, y, z, t), and state outputs are expected in
physics-variable order (u, v, w, theta, p_prime). Coordinates and state outputs may be
normalized; residual scaling maps them back to physical units before the PDE
terms are assembled.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig


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
    coordinate_scales = torch.tensor(
        scaling.coordinate_scales(),
        dtype=gradient.dtype,
        device=gradient.device,
    ).reshape(1, -1)
    return gradient / coordinate_scales

def _hydrostatic_reference_state(
    z: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hydrostatic pressure and density at physical height z (m)."""

    gravity = physics.constants.gravity
    cp = physics.constants.dry_air_specific_heat_cp
    rd = physics.constants.dry_air_gas_constant
    p0 = physics.constants.reference_pressure

    theta_h = torch.where(z <= 500.0, torch.full_like(z, 300.0),
        torch.where(z <= 650.0, 300.0 + 0.08 * (z - 500.0),
            312.0 + 0.003 * (z - 650.0)))

    # \int^z_0 1/300 dz = z/300
    integral_lower = z / 300.0
    # \int^500_0 1/300 dz + \int^z_500 1/(300+0.08(z-500)) dz 
    integral_middle = (500.0 / 300.0 + 
                       1 / 0.08 * torch.log((300.0 + 0.08 * (z - 500.0)) / 300.0))
    # \int^500_0 1/300 dz + \int^650_500 1/(300+0.08(z-500)) dz + \int^z_650 1/(312+0.003(z-650)) dz
    integral_upper = (500.0 / 300.0 + 
                      1 / 0.08 * torch.log(torch.tensor(312.0 / 300.0, 
                        dtype=z.dtype, device=z.device)) # compatible and no CPU-GPU mismatch
        + 1 / 0.003 * torch.log((312.0 + 0.003 * (z - 650.0)) / 312.0))

    integral = torch.where(z <= 500.0, integral_lower,
        torch.where(z <= 650.0, integral_middle, integral_upper))
    #\Pi_H(0) = 1
    pi_h = 1.0 - gravity * integral / cp

    p_h = p0 * pi_h.pow(cp / rd)
    t_h = theta_h * (p_h / p0).pow(rd / cp)
    rho_h = p_h / (rd * t_h)

    return p_h, rho_h

def _split_state(
    state: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return u, v, w, theta, and p_prime columns from the model state output."""

    if state.ndim != 2:
        raise ValueError("state must have shape (n_points, n_variables).")

    if state.shape[1] < physics.state_dim:
        msg = (
            "state has too few columns for the configured active variables; "
            f"expected at least {physics.state_dim}, got {state.shape[1]}."
        )
        raise ValueError(msg)

    u = state[:, physics.variable_index("u") : physics.variable_index("u") + 1]
    v = state[:, physics.variable_index("v") : physics.variable_index("v") + 1]
    w = state[:, physics.variable_index("w") : physics.variable_index("w") + 1]
    theta = state[:, physics.variable_index("theta") : physics.variable_index("theta") + 1]
    p_prime = state[:, physics.variable_index("p_prime") : physics.variable_index("p_prime") + 1]
    return u, v, w, theta, p_prime


def _to_physical_state(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    theta: torch.Tensor,
    p_prime: torch.Tensor,
    scaling: ResidualScalingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map normalized state variables to physical state variables."""

    u_physical = scaling.u.offset + scaling.u.scale * u
    v_physical = scaling.v.offset + scaling.v.scale * v
    w_physical = scaling.w.offset + scaling.w.scale * w
    theta_physical = scaling.theta.offset + scaling.theta.scale * theta
    p_prime_physical = scaling.p_prime.offset + scaling.p_prime.scale * p_prime
    return u_physical, v_physical, w_physical, theta_physical, p_prime_physical


def _validate_physics_config(physics: PhysicsConfig) -> None:
    """Ensure this implementation is used only for the supported reduced system."""

    if physics.coordinate_system != "local_cartesian":
        raise ValueError("Only local_cartesian coordinates are supported.")

    if physics.active_variables != ("u", "v", "w", "theta", "p_prime"):
        raise ValueError("Residuals currently require active variables u, v, w, theta, p_prime")

    expected_residuals = ("mass", "x_momentum", "y_momentum", "z_momentum", "potential_temperature")
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
        "include_turbulence": physics.include_turbulence,
    } # all must be true
    disabled_terms = [name for name, enabled in required_terms.items() if not enabled]
    if disabled_terms:
        raise ValueError(f"Required physics terms are disabled: {disabled_terms}.")

    unsupported_terms: Mapping[str, bool] = {
        "include_coriolis": physics.include_coriolis,
        "include_moisture": physics.include_moisture,
        "include_microphysics": physics.include_microphysics,
    } # even if turned on, not supported.
    enabled_terms = [name for name, enabled in unsupported_terms.items() if enabled]
    if enabled_terms:
        raise ValueError(f"Unsupported physics terms are enabled: enabled terms: {enabled_terms}.")

def cartesian_zero_forcing_residuals(
    coordinates: torch.Tensor,
    state: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
) -> TensorDict:
    """Compute reduced Cartesian continuity and momentum residuals.

    Parameters
    ----------
    coordinates:
        Collocation coordinates with shape ``(n_points, 4)`` and column order
        ``(x, y, z, t)``. The tensor must have ``requires_grad=True``.
    state:
        Model outputs with shape ``(n_points, 5)`` and column order
        ``(u, v, w, theta, p')``.
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
        Residual tensors keyed by ``mass``, ``x_momentum``, ``y_momentum``,
        ``z_momentum``, and ''potential_temperature''. Each tensor has shape ``(n_points, 1)``.
    """

    _validate_physics_config(physics)

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_points, 4).")

    if not coordinates.requires_grad:
        raise ValueError("coordinates must have requires_grad=True.")

    u_normalized, v_normalized, w_normalized, theta_normalized, p_prime_normalized = _split_state(
        state, physics)

    u, v, w, theta, p_prime = _to_physical_state(
        u_normalized, v_normalized, w_normalized,
        theta_normalized, p_prime_normalized, scaling)

    z = scaling.z.offset + scaling.z.scale * coordinates[:, 2:3]
    p_h, rho_h = _hydrostatic_reference_state(z, physics)

    p = p_h + p_prime
    temperature = theta * (p / physics.constants.reference_pressure).pow(
        physics.constants.kappa)
    rho = p / (physics.constants.dry_air_gas_constant * temperature)
    rho_prime = rho - rho_h

    grad_u = _physical_gradient(u, coordinates, scaling)
    grad_v = _physical_gradient(v, coordinates, scaling)
    grad_w = _physical_gradient(w, coordinates, scaling)
    grad_rho = _physical_gradient(rho, coordinates, scaling)
    grad_rho_u = _physical_gradient(rho * u, coordinates, scaling)
    grad_rho_v = _physical_gradient(rho * v, coordinates, scaling)
    grad_rho_w = _physical_gradient(rho * w, coordinates, scaling)

    u_x, u_y, u_z, u_t = grad_u.split(1, dim=1)
    v_x, v_y, v_z, v_t = grad_v.split(1, dim=1)
    w_x, w_y, w_z, w_t = grad_w.split(1, dim=1)
    _, _, _, rho_t = grad_rho.split(1, dim=1)
    rho_u_x, _, _, rho_u_t = grad_rho_u.split(1, dim=1)
    _, rho_v_y, _, rho_v_t = grad_rho_v.split(1, dim=1)
    _, _, rho_w_z, rho_w_t = grad_rho_w.split(1, dim=1)

    rho_uu_x = _physical_gradient(rho * u * u, coordinates, scaling)[:, 0:1]
    grad_rho_uv = _physical_gradient(rho * u * v, coordinates, scaling)
    grad_rho_uw = _physical_gradient(rho * u * w, coordinates, scaling)
    rho_vv_y = _physical_gradient(rho * v * v, coordinates, scaling)[:, 1:2]
    grad_rho_vw = _physical_gradient(rho * v * w, coordinates, scaling)
    rho_ww_z = _physical_gradient(rho * w * w, coordinates, scaling)[:, 2:3]

    rho_uv_x = grad_rho_uv[:, 0:1]
    rho_uv_y = grad_rho_uv[:, 1:2]
    rho_uw_x = grad_rho_uw[:, 0:1]
    rho_uw_z = grad_rho_uw[:, 2:3]
    rho_vw_y = grad_rho_vw[:, 1:2]
    rho_vw_z = grad_rho_vw[:, 2:3]

    grad_p_prime = _physical_gradient(p_prime, coordinates, scaling)
    p_prime_x, p_prime_y, p_prime_z, _ = grad_p_prime.split(1, dim=1)

    rho_theta_t = _physical_gradient(rho * theta, coordinates, scaling)[:, 3:4]
    rho_u_theta_x = _physical_gradient(rho * u * theta, coordinates, scaling)[:, 0:1]
    rho_v_theta_y = _physical_gradient(rho * v * theta, coordinates, scaling)[:, 1:2]
    rho_w_theta_z = _physical_gradient(rho * w * theta, coordinates, scaling)[:, 2:3]

    divergence = u_x + v_y + w_z

    k_m = physics.constants.eddy_viscosity

    tau_xx = rho * k_m * (2.0 * u_x - (2.0 / 3.0) * divergence)
    tau_yy = rho * k_m * (2.0 * v_y - (2.0 / 3.0) * divergence)
    tau_zz = rho * k_m * (2.0 * w_z - (2.0 / 3.0) * divergence)
    tau_xy = rho * k_m * (u_y + v_x)
    tau_xz = rho * k_m * (u_z + w_x)
    tau_yz = rho * k_m * (v_z + w_y)

    tau_xx_x = _physical_gradient(tau_xx, coordinates, scaling)[:, 0:1]
    tau_yy_y = _physical_gradient(tau_yy, coordinates, scaling)[:, 1:2]
    tau_zz_z = _physical_gradient(tau_zz, coordinates, scaling)[:, 2:3]

    grad_tau_xy = _physical_gradient(tau_xy, coordinates, scaling)
    grad_tau_xz = _physical_gradient(tau_xz, coordinates, scaling)
    grad_tau_yz = _physical_gradient(tau_yz, coordinates, scaling)

    tau_xy_x = grad_tau_xy[:, 0:1]
    tau_xy_y = grad_tau_xy[:, 1:2]
    tau_xz_x = grad_tau_xz[:, 0:1]
    tau_xz_z = grad_tau_xz[:, 2:3]
    tau_yz_y = grad_tau_yz[:, 1:2]
    tau_yz_z = grad_tau_yz[:, 2:3]

    mass = rho_t + rho_u_x + rho_v_y + rho_w_z

    x_momentum = (
        rho_u_t + rho_uu_x + rho_uv_y + rho_uw_z + p_prime_x
        - tau_xx_x - tau_xy_y - tau_xz_z
    )

    y_momentum = (
        rho_v_t + rho_uv_x + rho_vv_y + rho_vw_z + p_prime_y
        - tau_xy_x - tau_yy_y - tau_yz_z
    )

    z_momentum = (
        rho_w_t + rho_uw_x + rho_vw_y + rho_ww_z + p_prime_z
        + physics.constants.gravity * rho_prime
        - tau_xz_x - tau_yz_y - tau_zz_z
    )

    potential_temperature = (
        rho_theta_t + rho_u_theta_x + rho_v_theta_y + rho_w_theta_z
    )

    return {
        "mass": mass,
        "x_momentum": x_momentum,
        "y_momentum": y_momentum,
        "z_momentum": z_momentum,
        "potential_temperature": potential_temperature,
    }