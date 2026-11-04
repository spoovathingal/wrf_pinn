"""Boundary-condition residuals for WRF PINN training.

This module contains residual definitions for physical boundary constraints.
It does not read boundary geometry files, sample points, or assemble losses.
Those responsibilities belong to the data/sampling and training layers.
"""

from __future__ import annotations

import torch
import math

from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.physics.residuals_pde import _diagnostic_sgs_coefficients, _physical_gradient
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig


def _validate_state(state: torch.Tensor) -> None:
    """Validate model-output tensor shape for boundary residuals."""

    if state.ndim != 2:
        raise ValueError(
            "state must be a 2D tensor with shape "
            "(num_wall_points, num_state_variables)."
        )


def _validate_indices(
    indices: tuple[int, ...],
    num_state_variables: int,
    *,
    name: str,
) -> None:
    """Validate state-column indices."""

    for index in indices:
        if index < 0 or index >= num_state_variables:
            raise ValueError(
                f"{name} index is outside the state tensor columns; "
                f"got index {index} for {num_state_variables} columns."
            )


def _physical_velocity_components(
    state: torch.Tensor,
    velocity_indices: tuple[int, int, int],
    scaling: ResidualScalingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physical u, v, and w from normalized model state."""

    if len(velocity_indices) != 3:
        raise ValueError(
            "velocity_indices must contain exactly three indices for u, v, w; "
            f"got {velocity_indices}."
        )

    _validate_indices(
        velocity_indices,
        state.shape[1],
        name="velocity",
    )

    u_index, v_index, w_index = velocity_indices
    u = scaling.u.offset + scaling.u.scale * state[:, u_index]
    v = scaling.v.offset + scaling.v.scale * state[:, v_index]
    w = scaling.w.offset + scaling.w.scale * state[:, w_index]
    return u, v, w


def no_slip_wall_residuals(
    state: torch.Tensor,
    velocity_indices: tuple[int, int, int] = (0, 1, 2),
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
) -> dict[str, torch.Tensor]:
    """Return no-slip wall residuals from model predictions at wall points.

    The no-slip condition requires the velocity components to vanish at wall
    coordinates:

    ``u = 0, v = 0, w = 0``.

    The residuals are computed in physical velocity units. If the model outputs
    are normalized, ``scaling`` maps them back to physical values before the
    zero-velocity condition is evaluated.

    Parameters
    ----------
    state:
        Model output evaluated at wall coordinates. Expected shape is
        ``(num_wall_points, num_state_variables)``.

    velocity_indices:
        Column indices for ``u``, ``v``, and ``w`` in the model output.
    scaling:
        Affine maps from normalized model outputs to physical velocity values.
        Identity scaling preserves the old behavior.

    Returns
    -------
    dict[str, torch.Tensor]
        Residual tensors that should be driven to zero at the wall.
    """

    _validate_state(state)
    u, v, w = _physical_velocity_components(state, velocity_indices, scaling)

    return {
        "no_slip_u": u/scaling.u.scale,
        "no_slip_v": v/scaling.v.scale,
        "no_slip_w": w/scaling.w.scale,
    }

def _physical_density_from_state(state: torch.Tensor,
                                 physics: PhysicsConfig, scaling: ResidualScalingConfig) -> torch.Tensor:
    """Calculate physical dry-air density from model prediction."""
    if state.shape[1] != physics.state_dim:
        raise ValueError(
            f"Expected {physics.state_dim} state columns, including rho_d; "
            f"got {state.shape[1]}."
        )

    rho_d_index = physics.variable_index("rho_d")
    rho_d_normalized = state[:, rho_d_index:rho_d_index + 1]

    return scaling.rho_d.offset + scaling.rho_d.scale * rho_d_normalized

def _validate_no_penetration_z_wall_inputs(coordinates: torch.Tensor, state: torch.Tensor,
                                           reference_coordinates: torch.Tensor, reference_state: torch.Tensor, 
                                           physics: PhysicsConfig, scaling: ResidualScalingConfig,
                                           surface_stress_scale: float,
                                           *,
                                           check_values: bool = True,
                                           terrain_height: torch.Tensor,
                                           terrain_slopes: torch.Tensor,
                                           ) -> "torch.Tensor | None":
    """Validate surface inputs and return the bottom-point mask.
    With check_values=False, return None: the caller guarantees that every
    row lies on the supplied terrain surface, Z = h(x, y).
    """
    _validate_state(state)
    _validate_state(reference_state)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (num_wall_points, 4).")

    if reference_coordinates.shape != coordinates.shape:
        raise ValueError("reference_coordinates must have the same shape as coordinates.")

    if state.shape[0] != coordinates.shape[0]:
        raise ValueError("state and coordinates must have the same number of rows.")

    if reference_state.shape[0] != reference_coordinates.shape[0]:
        raise ValueError("reference_state and reference_coordinates must have the same number of rows.")

    if state.shape[1] != physics.state_dim:
        raise ValueError(
            f"state must contain exactly {physics.state_dim} columns, including rho_d.")

    if reference_state.shape[1] != physics.state_dim:
        raise ValueError(
            f"reference_state must contain exactly {physics.state_dim} columns, including rho_d.")

    if not math.isfinite(surface_stress_scale) or surface_stress_scale <= 0.0:
        raise ValueError("surface_stress_scale must be finite and positive.")

    for name, value, width in (
        ("terrain_height", terrain_height, 1), ("terrain_slopes", terrain_slopes, 2),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor.")
        if value.shape != (coordinates.shape[0], width):
            raise ValueError(f"{name} must have shape (num_wall_points, {width}).")
        if check_values and not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")

    if not check_values:
        return None

    z_physical = scaling.z.offset + scaling.z.scale * coordinates[:, 2:3]
    h = terrain_height.to(coordinates)

    # Allow float32 roundoff when reconstructing physical height.
    tolerance = max(1.0e-5,
        4.0 * torch.finfo(coordinates.dtype).eps * max(abs(scaling.z.offset), abs(scaling.z.scale), 1.0))
    bottom_mask = torch.isclose(z_physical - h, torch.zeros_like(z_physical),
                                rtol=0.0, atol=tolerance).squeeze(dim=1)
    if not torch.any(bottom_mask):
        raise ValueError("No points on the supplied terrain surface were supplied.")
    
    return bottom_mask

def _surface_layer_transfer_coefficients(
    theta, z1, z0m, fric_vel, ht_flux, physics, *, check_values=True,
):
    """2020 Dyer coefficients with |zeta| <= 2.5.

    All inputs are physical (N, 1) tensors. Thermal roughness is
    assumed equal to momentum roughness, as in the 2020 formula.
    """
    kappa = physics.constants.von_karman_constant
    gravity = physics.constants.gravity

    # Neutral conditions, including u_*=0 with zero heat flux.
    ustar = torch.where(ht_flux == 0.0, torch.ones_like(fric_vel), fric_vel)

    # zeta = (z1 + z0m)/L.
    zeta = (-kappa * gravity * (z1 + z0m) * ht_flux / (ustar.pow(3) * theta))
    zeta = zeta.clamp(min=-2.5, max=2.5)

    # Keep the inactive unstable branch real at stable points.
    unstable_zeta = torch.minimum(zeta, torch.zeros_like(zeta))
    xi = (1.0 - 16.0 * unstable_zeta).pow(0.25)

    psi_m_unstable = (2.0 * torch.log((1.0 + xi) / 2.0) + torch.log((1.0 + xi.square()) / 2.0)
        - 2.0 * torch.atan(xi) + math.pi / 2.0
    )
    psi_h_unstable = (
        2.0 * torch.log((1.0 + xi.square()) / 2.0)
    )

    psi_m = torch.where(zeta < 0.0, psi_m_unstable, -5.0 * zeta)
    psi_h = torch.where(zeta < 0.0, psi_h_unstable, -5.0 * zeta)

    log_z = torch.log((z1 + z0m) / z0m)
    Dm = log_z - psi_m
    Dh = log_z - psi_h

    if check_values:
        if not bool((torch.isfinite(Dm) & torch.isfinite(Dh) & (Dm > 0.0) & (Dh > 0.0)).all()):
            raise ValueError("Surface transfer denominators must be finite and positive.")

    cd = kappa**2 / Dm.square()
    ch = kappa**2 / (Dm * Dh)
    return cd, ch

def no_penetration_z_wall_residuals(
    coordinates: torch.Tensor, state: torch.Tensor,
    reference_coordinates: torch.Tensor, reference_state: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
    surface_stress_scale: float = 3.8273,
    *, fric_vel: torch.Tensor, ht_flux: torch.Tensor, surface_fe_state: torch.Tensor,
    z0m: torch.Tensor, surface_heat_flux_scale: float = 1.0,
    state_jacobian: "torch.Tensor | None" = None,
    check_values: bool = True,
    terrain_height: torch.Tensor, terrain_slopes: torch.Tensor,
    filter_width: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return terrain no-penetration, surface stress, and heat-flux residuals.

    coordinates and state describe points on the physical terrain surface.
    reference_coordinates and reference_state describe paired points at
    the first off-wall reference height.

    fric_vel [m/s] and ht_flux [K m/s] are physical (N, 1) tensors.
    terrain_height is physical elevation [m]; terrain_slopes contains
    physical (dh/dx, dh/dy).

    terrain_height and terrain_slopes are required, including for flat
    terrain, where the corresponding elevations/slopes can be zero.

    filter_width is a required physical (N, 1) tensor [m]. It uses the
    first native cell's vertical spacing, compressed by terrain and
    extended down to the surface. Supply the values from grid_geometry.
    A Tensor is used for both the autograd and Taylor boundary paths.

    surface_fe_state contains physical FE values at native z-index 0,
    with shape (N, 4) and columns (rho_d, u, v, theta). Each row must
    correspond to the same location and time as reference_state.
    z0m contains physical momentum roughness lengths, shape (N, 1).

    state_jacobian, when supplied, contains derivatives of normalized
    state with respect to physical coordinates, with shape
    (n_coord, N, n_state). Otherwise, derivatives use autograd.

    check_values=False skips value checks and surface-point masking.
    The caller then guarantees valid inputs and that every row lies
    on the supplied terrain surface.

    Heat-flux supervision compares expanded surface-layer fluxes at
    matched first-cell points. Effective skin potential temperature is
    inferred from FE observations only and shared by both calculations.
    It is not a network output or a separately loaded temperature field.
    """
    bottom_mask = _validate_no_penetration_z_wall_inputs(
        coordinates=coordinates, state=state,
        reference_coordinates=reference_coordinates, reference_state=reference_state,
        physics=physics, scaling=scaling, surface_stress_scale=surface_stress_scale,
        check_values=check_values, terrain_height=terrain_height, terrain_slopes=terrain_slopes,
    )

    for name, field in (("fric_vel", fric_vel), ("ht_flux", ht_flux), ("z0m", z0m)):
        if field.shape != (state.shape[0], 1):
            raise ValueError(f"{name} must have shape (num_wall_points, 1).")
        if check_values and not torch.isfinite(field).all():
            raise ValueError(f"{name} must contain only finite values.")
    if check_values and torch.any(fric_vel < 0.0):
        raise ValueError("fric_vel must be nonnegative.")
    if not math.isfinite(surface_heat_flux_scale) or surface_heat_flux_scale <= 0.0:
        raise ValueError("surface_heat_flux_scale must be finite and positive.")

    if surface_fe_state.shape != (state.shape[0], 4):
        raise ValueError("surface_fe_state must have shape (N, 4): rho_d, u, v, theta.")

    if check_values:
        if not torch.isfinite(surface_fe_state).all():
            raise ValueError("surface_fe_state must contain finite values.")
        if torch.any(surface_fe_state[:, [0, 3]] <= 0.0):
            raise ValueError("FE density and potential temperature must be positive.")
        if torch.any(z0m <= 0.0):
            raise ValueError("z0m must be positive.")
        if torch.any((fric_vel == 0.0) & (ht_flux != 0.0)):
            raise ValueError("Nonzero heat flux requires positive friction velocity.")

    # These are prescribed observations, not trainable model outputs.
    fric_vel = fric_vel.to(state).detach()
    ht_flux = ht_flux.to(state).detach()
    z0m = z0m.to(state).detach()
    surface_fe_state = surface_fe_state.to(state).detach()

    u_index = physics.variable_index("u")
    v_index = physics.variable_index("v")
    w_index = physics.variable_index("w")
    theta_index = physics.variable_index("theta")
    e_sgs_index = physics.variable_index("e_sgs")

    # Physical wall velocities.
    u_wall = (scaling.u.offset + scaling.u.scale * state[:, u_index : u_index + 1])
    v_wall = (scaling.v.offset + scaling.v.scale * state[:, v_index : v_index + 1])
    w_wall = (scaling.w.offset + scaling.w.scale * state[:, w_index : w_index + 1])
    theta_wall = (scaling.theta.offset + scaling.theta.scale * state[:, theta_index : theta_index + 1])
    e_sgs_wall = (scaling.e_sgs.offset + scaling.e_sgs.scale * state[:, e_sgs_index : e_sgs_index + 1])

    # Physical first-reference-level velocities.
    u_1 = scaling.u.offset + scaling.u.scale * reference_state[:, u_index : u_index + 1]
    v_1 = scaling.v.offset + scaling.v.scale * reference_state[:, v_index : v_index + 1]

    theta_1 = (scaling.theta.offset + scaling.theta.scale * reference_state[:, theta_index:theta_index + 1])

    z1 = (scaling.z.offset + scaling.z.scale * reference_coordinates[:, 2:3]
        - terrain_height.to(reference_coordinates)).to(state)

    if check_values and (torch.any(z1 <= 0.0) or torch.any(theta_1 <= 0.0)):
        raise ValueError("Reference height above ground and predicted theta must be positive.")

    rho_wall = _physical_density_from_state(state, physics, scaling)
    rho_1 = _physical_density_from_state(reference_state, physics, scaling)

    if state_jacobian is not None:
        grad = lambda name, i: getattr(scaling, name).scale * state_jacobian[:, :, i].T
        grad_u, grad_v = grad("u", u_index), grad("v", v_index)
        grad_w, grad_theta = grad("w", w_index), grad("theta", theta_index)
    else:
        grad_u = _physical_gradient(u_wall, coordinates, scaling)
        grad_v = _physical_gradient(v_wall, coordinates, scaling)
        grad_w = _physical_gradient(w_wall, coordinates, scaling)
        grad_theta = _physical_gradient(theta_wall, coordinates, scaling)

    u_z = grad_u[:, 2:3]
    v_z = grad_v[:, 2:3]
    w_x = grad_w[:, 0:1]
    w_y = grad_w[:, 1:2]
    theta_z = grad_theta[:, 2:3]
    # Diagnose wall K_m using the same closure as the interior PDE.
    _, _, k_m_wall, k_theta_wall = _diagnostic_sgs_coefficients(
        theta=theta_wall, theta_z=theta_z, e_sgs=e_sgs_wall, physics=physics, filter_width=filter_width)
    # Ordinary SGS stress evaluated on the fluid side of the wall.
    tau_xz_sgs = -rho_wall * k_m_wall * (u_z + w_x)
    tau_yz_sgs = -rho_wall * k_m_wall * (v_z + w_y)

    # FE observations and model predictions at the same first cell.
    rho_fe, u_fe, v_fe, theta_fe = surface_fe_state.split(1, dim=1)

    cd_model, ch_model = _surface_layer_transfer_coefficients(
        theta_1, z1, z0m, fric_vel, ht_flux, physics, check_values=check_values)

    # Prescribed skin temperature comes exclusively from FE inputs.
    # No model prediction or trainable gradient enters this calculation.
    with torch.no_grad():
        cd_fe, ch_fe = _surface_layer_transfer_coefficients(
            theta_fe, z1, z0m, fric_vel, ht_flux, physics, check_values=check_values)

        U_fe = torch.sqrt(u_fe.square() + v_fe.square())

        if check_values and not bool((torch.isfinite(U_fe) & (U_fe > 0.0)).all()):
            raise ValueError("Inferring skin theta requires positive finite FE wind speed.")

        # htFlux_FE = ch_FE * U_FE * (theta_skin_FE - theta_FE).
        theta_skin_fe = theta_fe + ht_flux / (ch_fe * U_fe)

        if check_values and not bool((torch.isfinite(theta_skin_fe) & (theta_skin_fe > 0.0)).all()):
            raise ValueError("Inferred skin theta must be finite and positive.")

    # Smooth the model wind norm to keep calm-wind gradients finite.
    U_model = torch.sqrt(u_1.square() + v_1.square() + 1.0e-12)

    # Expanded surface-layer momentum fluxes.
    tau_xz_sl_model = -rho_1 * cd_model * u_1 * U_model
    tau_yz_sl_model = -rho_1 * cd_model * v_1 * U_model

    tau_xz_sl_fe = -rho_fe * cd_fe * u_fe * U_fe
    tau_yz_sl_fe = -rho_fe * cd_fe * v_fe * U_fe

    # Active: model SL stresses versus FE SL stresses.
    r_xz = (tau_xz_sl_model - tau_xz_sl_fe) / surface_stress_scale
    r_yz = (tau_yz_sl_model - tau_yz_sl_fe) / surface_stress_scale

    # Alternative: model SL stresses versus ordinary SGS stresses.
    # Disable the active assignments above before enabling these.
    # r_xz = (tau_xz_sl_model - tau_xz_sgs) / surface_stress_scale
    # r_yz = (tau_yz_sl_model - tau_yz_sgs) / surface_stress_scale

    # Density-weighted potential-temperature flux; no cp factor.
    q_sgs = -rho_wall * k_theta_wall * theta_z

    # Both SL expressions use the SAME prescribed FE-derived skin theta.
    q_sl_model = (rho_1 * ch_model * U_model * (theta_skin_fe - theta_1))
    q_sl_fe = (rho_fe * ch_fe * U_fe * (theta_skin_fe - theta_fe))

    # Active: model SL heat flux versus FE SL heat flux.
    r_heat = (q_sl_model - q_sl_fe) / surface_heat_flux_scale

    # Alternative: ordinary SGS heat flux versus expanded model SL flux.
    # Disable the active assignment above before enabling this.
    # r_heat = (q_sgs - q_sl_model) / surface_heat_flux_scale

    slopes = terrain_slopes.to(state)
    hx = slopes[:, 0:1]
    hy = slopes[:, 1:2]
    normal_velocity = (w_wall - u_wall * hx - v_wall * hy) / torch.sqrt(1.0 + hx.square() + hy.square())
    bottom = (lambda r: r) if bottom_mask is None else (lambda r: r[bottom_mask])
    return {
        "no_penetration_w": normal_velocity / scaling.w.scale, # z = 0 and z = z_max
        "surface_stress_xz": bottom(r_xz), # z = 0
        "surface_stress_yz": bottom(r_yz), # z = 0
        "surface_heat_flux": bottom(r_heat),
    }
