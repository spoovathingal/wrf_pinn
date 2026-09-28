"""Residual equations for the simplified Cartesian WRF PINN model.

This module implements the dry neutral boundary layer system described in
``wrf_pinn.config.physics``:

- local Cartesian coordinates
- zero external forcing & Coriolis
- fixed hydrostatic reference potential temperature = 300 K
- q_l and f_cond = 0

The residuals are evaluated at continuous PINN collocation points. Inputs are
expected in coordinate order (x, y, z, t), and state outputs are expected in
physics-variable order (u, v, w, theta, p_prime, q_v, e_sgs). Coordinates and state outputs may be
normalized; residual scaling maps them back to physical units before the PDE
terms are assembled.
"""

from __future__ import annotations
import math
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
    theta_h = physics.hydrostatic_reference_potential_temperature
    # theta_h = torch.where(z <= 500.0, torch.full_like(z, 300.0),
    #     torch.where(z <= 650.0, 300.0 + 0.08 * (z - 500.0),
    #         312.0 + 0.003 * (z - 650.0)))

    # # \int^z_0 1/300 dz = z/300
    # integral_lower = z / 300.0
    # # \int^500_0 1/300 dz + \int^z_500 1/(300+0.08(z-500)) dz 
    # integral_middle = (500.0 / 300.0 + 
    #                    1 / 0.08 * torch.log((300.0 + 0.08 * (z - 500.0)) / 300.0))
    # # \int^500_0 1/300 dz + \int^650_500 1/(300+0.08(z-500)) dz + \int^z_650 1/(312+0.003(z-650)) dz
    # integral_upper = (500.0 / 300.0 + 
    #                   1 / 0.08 * torch.log(torch.tensor(312.0 / 300.0, 
    #                     dtype=z.dtype, device=z.device)) # compatible and no CPU-GPU mismatch
    #     + 1 / 0.003 * torch.log((312.0 + 0.003 * (z - 650.0)) / 312.0))

    # integral = torch.where(z <= 500.0, integral_lower,
    #     torch.where(z <= 650.0, integral_middle, integral_upper))
    #\Pi_H(0) = 1
    # pi_h = 1.0 - gravity * integral / cp

    # For the current flat ocean crop:
    # z_reference = 0 m, p_reference = p0, theta_d,h = 300 K.
    exner_h = 1.0 - gravity * z / (cp * theta_h)
    p_d_h = p0 * exner_h.pow(cp / rd)
    t_d_h = theta_h * exner_h
    rho_d_h = p_d_h / (rd * t_d_h)

    return p_d_h, rho_d_h

def _split_state(
    state: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return u, v, w, theta, p_prime, q_v, and e_sgs columns from the model state output."""

    if state.ndim != 2:
        raise ValueError("state must have shape (n_points, n_variables).")

    if state.shape[1] < physics.state_dim:
        raise ValueError(
            "state has too few columns for the configured active variables; "
            f"expected at least {physics.state_dim}, got {state.shape[1]}."
        )

    u = state[:, physics.variable_index("u") : physics.variable_index("u") + 1]
    v = state[:, physics.variable_index("v") : physics.variable_index("v") + 1]
    w = state[:, physics.variable_index("w") : physics.variable_index("w") + 1]
    theta = state[:, physics.variable_index("theta") : physics.variable_index("theta") + 1]
    p_prime = state[:, physics.variable_index("p_prime") : physics.variable_index("p_prime") + 1]
    q_v = state[:, physics.variable_index("q_v") : physics.variable_index("q_v") + 1]
    e_sgs = state[:, physics.variable_index("e_sgs") : physics.variable_index("e_sgs") + 1]
    return u, v, w, theta, p_prime, q_v, e_sgs


def _to_physical_state(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    theta: torch.Tensor,
    p_prime: torch.Tensor,
    q_v: torch.Tensor,
    e_sgs: torch.Tensor,
    scaling: ResidualScalingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map normalized state variables to physical state variables."""

    u_physical = scaling.u.offset + scaling.u.scale * u
    v_physical = scaling.v.offset + scaling.v.scale * v
    w_physical = scaling.w.offset + scaling.w.scale * w
    theta_physical = scaling.theta.offset + scaling.theta.scale * theta
    p_prime_physical = scaling.p_prime.offset + scaling.p_prime.scale * p_prime
    q_v_physical = scaling.q_v.offset + scaling.q_v.scale * q_v
    e_sgs_physical = scaling.e_sgs.offset + scaling.e_sgs.scale * e_sgs
    return u_physical, v_physical, w_physical, theta_physical, p_prime_physical, q_v_physical, e_sgs_physical
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

    if physics.active_variables != ("u", "v", "w", "theta", "p_prime", "q_v", "e_sgs"):
        raise ValueError("Residuals currently require active variables u, v, w, theta, p_prime, q_v, e_sgs")

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

def _diagnose_moist_state(theta: torch.Tensor, p_prime: torch.Tensor, q_v: torch.Tensor,
    z: torch.Tensor, physics: PhysicsConfig) -> tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnose pressure, temperature, and moist-density quantities.

    ``q_v`` must be the water-vapor mixing ratio in kg/kg relative to
    dry-air mass.
    """

    rd = physics.constants.dry_air_gas_constant
    rv = physics.constants.water_vapor_gas_constant
    p0 = physics.constants.reference_pressure
    kappa = physics.constants.kappa
    gamma = physics.constants.gamma

    p_d_h, rho_d_h = _hydrostatic_reference_state(z, physics)

    # Total pressure and moist potential temperature.
    pressure = p_d_h + p_prime
    theta_m = theta * (1.0 + (rv / rd) * q_v)

    # Dry-air density obtained from the moist equation of state.
    rho_d = (  p0**kappa * pressure.pow(1.0 / gamma) / (rd * theta_m)  )

    temperature = theta * (pressure / p0).pow(kappa)

    # q_l = 0 for the current first-pass system.
    rho_m = rho_d * (1.0 + q_v)
    rho_m_prime = rho_m - rho_d_h

    # Multiplies the pressure-gradient and buoyancy bracket.
    density_ratio = rho_d / rho_m

    return (pressure, temperature, rho_d, rho_m, rho_m_prime, density_ratio)

def _brunt_vaisala_squared(theta: torch.Tensor, theta_z: torch.Tensor, physics: PhysicsConfig) -> torch.Tensor:
    """Return N^2 from local dry potential temperature. used for l_sgs"""

    theta_floor = torch.finfo(theta.dtype).tiny
    theta_safe = theta.clamp_min(theta_floor)

    return (physics.constants.gravity * theta_z / theta_safe)

def _sgs_mixing_length(n_squared: torch.Tensor, e_sgs: torch.Tensor, physics: PhysicsConfig) -> torch.Tensor:
    """Return the SGS mixing length for stable and unstable flow."""

    delta = e_sgs.new_tensor(physics.uniform_filter_width)
    minimum_length = e_sgs.new_tensor(physics.minimum_mixing_length)

    numerical_floor = torch.finfo(e_sgs.dtype).tiny
    sqrt_e_sgs = torch.sqrt(e_sgs.clamp_min(0.0) + numerical_floor)

    # torch.where evaluates both branches, so N^2 must be clamped before
    # taking its square root even where the unstable branch is selected.
    stable_n = torch.sqrt(n_squared.clamp_min(numerical_floor))

    stable_candidate = (physics.stable_mixing_length_coefficient * sqrt_e_sgs / stable_n)

    stable_length = torch.maximum(torch.minimum(stable_candidate, delta), minimum_length)

    return torch.where(n_squared > 0.0, stable_length, delta)

def _diagnostic_sgs_coefficients(
    theta: torch.Tensor,
    theta_z: torch.Tensor,
    e_sgs: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnose N^2, mixing length, K_m, and scalar diffusivity."""
    n_squared = _brunt_vaisala_squared(theta, theta_z, physics)
    mixing_length = _sgs_mixing_length(n_squared, e_sgs, physics)

    numerical_floor = torch.finfo(e_sgs.dtype).tiny
    sqrt_e_sgs = torch.sqrt(e_sgs.clamp_min(0.0) + numerical_floor)

    k_m = (physics.sgs_mixing_coefficient * mixing_length * sqrt_e_sgs)

    delta = e_sgs.new_tensor(physics.uniform_filter_width)

    k_scalar = k_m * (1.0 + 2.0 * mixing_length / delta)

    return (n_squared, mixing_length, k_m, k_scalar)

def cartesian_zero_forcing_residuals(
    coordinates: torch.Tensor,
    state: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
) -> TensorDict:
    """Compute reduced moist Cartesian continuity and momentum residuals.

    Parameters
    ----------
    coordinates:
        Collocation coordinates with shape ``(n_points, 4)`` and column order
        ``(x, y, z, t)``. The tensor must have ``requires_grad=True``.
    state:
        Model outputs with shape ``(n_points, 7)`` and column order
        ``(u, v, w, theta, p', q_v, e_sgs)``.
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
        ``z_momentum``, ''potential_temperature'', and "water vapor" Each tensor has shape ``(n_points, 1)``.
    """

    _validate_physics_config(physics)

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_points, 4).")

    if not coordinates.requires_grad:
        raise ValueError("coordinates must have requires_grad=True.")

    (u_normalized, v_normalized, w_normalized, theta_normalized, p_prime_normalized,
      q_v_normalized, e_sgs_normalized) = _split_state(state, physics)

    u, v, w, theta, p_prime, q_v, e_sgs = _to_physical_state(
        u_normalized, v_normalized, w_normalized, theta_normalized, 
        p_prime_normalized, q_v_normalized, e_sgs_normalized, scaling)
    #k_m = _to_physical_eddy_viscosity(k_m_unconstrained, physics)

    z = scaling.z.offset + scaling.z.scale * coordinates[:, 2:3]
    (_pressure, _temperature, rho_d, _rho_m, rho_m_prime, density_ratio,
     ) = _diagnose_moist_state(theta, p_prime, q_v, z, physics)

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
    _n_squared, _mixing_length, k_m, k_scalar = _diagnostic_sgs_coefficients(theta,
        theta_z, e_sgs, physics)

    # The current first-pass model uses the same SGS diffusivity for
    # potential temperature and water vapor.
    k_theta = k_scalar
    k_q_v = k_scalar

    # Dry-air mass and momentum storage/transport products.
    grad_rho_d = _physical_gradient(rho_d, coordinates, scaling)
    grad_rho_d_u = _physical_gradient(rho_d * u, coordinates, scaling)
    grad_rho_d_v = _physical_gradient(rho_d * v, coordinates, scaling)
    grad_rho_d_w = _physical_gradient(rho_d * w, coordinates, scaling)

    _, _, _, rho_d_t = grad_rho_d.split(1, dim=1)
    rho_d_u_x, _, _, rho_d_u_t = grad_rho_d_u.split(1, dim=1)
    _, rho_d_v_y, _, rho_d_v_t = grad_rho_d_v.split(1, dim=1)
    _, _, rho_d_w_z, rho_d_w_t = grad_rho_d_w.split(1, dim=1)

    rho_d_uu_x = _physical_gradient(rho_d * u * u, coordinates, scaling)[:, 0:1]
    rho_d_vv_y = _physical_gradient(rho_d * v * v, coordinates, scaling)[:, 1:2]    
    rho_d_ww_z = _physical_gradient(rho_d * w * w, coordinates, scaling)[:, 2:3]

    grad_rho_d_uv = _physical_gradient(rho_d * u * v, coordinates, scaling)
    grad_rho_d_uw = _physical_gradient(rho_d * u * w, coordinates, scaling)
    grad_rho_d_vw = _physical_gradient(rho_d * v * w, coordinates, scaling)

    rho_d_uv_x = grad_rho_d_uv[:, 0:1]
    rho_d_uv_y = grad_rho_d_uv[:, 1:2]
    rho_d_uw_x = grad_rho_d_uw[:, 0:1]
    rho_d_uw_z = grad_rho_d_uw[:, 2:3]
    rho_d_vw_y = grad_rho_d_vw[:, 1:2]
    rho_d_vw_z = grad_rho_d_vw[:, 2:3]

    # Conservative potential-temperature products.
    rho_d_theta_t = _physical_gradient(rho_d * theta, coordinates, scaling)[:, 3:4]
    rho_d_u_theta_x = _physical_gradient(rho_d * u * theta, coordinates, scaling)[:, 0:1]
    rho_d_v_theta_y = _physical_gradient(rho_d * v * theta, coordinates, scaling)[:, 1:2]
    rho_d_w_theta_z = _physical_gradient(rho_d * w * theta, coordinates, scaling)[:, 2:3]

    # Conservative water-vapor products.
    rho_d_q_v_t = _physical_gradient(rho_d * q_v, coordinates, scaling)[:, 3:4]
    rho_d_u_q_v_x = _physical_gradient(rho_d * u * q_v, coordinates, scaling)[:, 0:1]
    rho_d_v_q_v_y = _physical_gradient(rho_d * v * q_v, coordinates, scaling)[:, 1:2]
    rho_d_w_q_v_z = _physical_gradient(rho_d * w * q_v, coordinates, scaling)[:, 2:3]

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
        "water_vapor": water_vapor
    }