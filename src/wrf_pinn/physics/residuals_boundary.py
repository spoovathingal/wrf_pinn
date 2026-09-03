"""Boundary-condition residuals for WRF PINN training.

This module contains residual definitions for physical boundary constraints.
It does not read boundary geometry files, sample points, or assemble losses.
Those responsibilities belong to the data/sampling and training layers.
"""

from __future__ import annotations

import torch
import math

from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.physics.residuals_pde import _hydrostatic_reference_state, _physical_gradient, _to_physical_eddy_viscosity
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

def _physical_density_from_state(coordinates: torch.Tensor, state: torch.Tensor,
                                 physics: PhysicsConfig, scaling: ResidualScalingConfig) -> torch.Tensor:
    """Calculate physical dry-air density from theta and p-prime."""

    theta_index = physics.variable_index("theta")
    p_prime_index = physics.variable_index("p_prime")

    theta_normalized = state[:, theta_index : theta_index + 1]
    p_prime_normalized = state[:, p_prime_index : p_prime_index + 1]

    theta = (scaling.theta.offset + scaling.theta.scale * theta_normalized)
    p_prime = (scaling.p_prime.offset + scaling.p_prime.scale * p_prime_normalized)

    z = (scaling.z.offset + scaling.z.scale * coordinates[:, 2:3])

    p_h, _ = _hydrostatic_reference_state(z, physics)
    p = p_h + p_prime

    temperature = theta * (p / physics.constants.reference_pressure).pow(physics.constants.kappa)

    return p / (physics.constants.dry_air_gas_constant * temperature)

def _validate_no_penetration_z_wall_inputs(coordinates: torch.Tensor, state: torch.Tensor,
                                           reference_coordinates: torch.Tensor, reference_state: torch.Tensor, 
                                           physics: PhysicsConfig, scaling: ResidualScalingConfig,
                                           surface_stress_scale: float) -> torch.Tensor:
    """Validate surface-boundary inputs and return the bottom-point mask."""
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

    if state.shape[1] < physics.state_dim:
        raise ValueError("state has too few columns for the configured physics variables.")

    if reference_state.shape[1] < physics.state_dim:
        raise ValueError("reference_state has too few columns for the configured physics variables.")

    if not math.isfinite(surface_stress_scale) or surface_stress_scale <= 0.0:
        raise ValueError("surface_stress_scale must be finite and positive.")

    z_physical = (scaling.z.offset + scaling.z.scale * coordinates[:, 2:3])

    bottom_mask = torch.isclose(z_physical, torch.zeros_like(z_physical),
                                rtol=0.0, atol=1.0e-5).squeeze(dim=1)
    if not torch.any(bottom_mask):
        raise ValueError("No z=0 bottom-boundary points were supplied.")
    
    return bottom_mask

def no_penetration_z_wall_residuals(
    coordinates: torch.Tensor,
    state: torch.Tensor,
    reference_coordinates: torch.Tensor,
    reference_state: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
    surface_stress_scale: float = 3.8273,
) -> dict[str, torch.Tensor]:
    """Return no-penetration and neutral surface stress residuals.

    ``coordinates`` and ``state`` describe the physical wall at z=0.
    ``reference_coordinates`` and ``reference_state`` describe paired points
    at the first off-wall reference height z=z1.
    """
    bottom_mask = _validate_no_penetration_z_wall_inputs(
        coordinates=coordinates,
        state=state,
        reference_coordinates=reference_coordinates,
        reference_state=reference_state,
        physics=physics,
        scaling=scaling,
        surface_stress_scale=surface_stress_scale,
    )

    u_index = physics.variable_index("u")
    v_index = physics.variable_index("v")
    w_index = physics.variable_index("w")
    k_m_index = physics.variable_index("k_m")

    # Physical wall velocities.
    u_wall = (scaling.u.offset + scaling.u.scale * state[:, u_index : u_index + 1])
    v_wall = (scaling.v.offset + scaling.v.scale * state[:, v_index : v_index + 1])
    w_wall = (scaling.w.offset + scaling.w.scale * state[:, w_index : w_index + 1])

    # Physical first-reference-level velocities.
    u_1 = scaling.u.offset + scaling.u.scale * reference_state[:, u_index : u_index + 1]
    v_1 = scaling.v.offset + scaling.v.scale * reference_state[:, v_index : v_index + 1]

    rho_wall = _physical_density_from_state(coordinates, state, physics, scaling)
    rho_1 = _physical_density_from_state(reference_coordinates, reference_state, physics, scaling)

    k_m_wall = _to_physical_eddy_viscosity(state[:, k_m_index : k_m_index + 1], physics)

    grad_u = _physical_gradient(u_wall, coordinates, scaling)
    grad_v = _physical_gradient(v_wall, coordinates, scaling)
    grad_w = _physical_gradient(w_wall, coordinates, scaling)

    u_z = grad_u[:, 2:3]
    v_z = grad_v[:, 2:3]
    w_x = grad_w[:, 0:1]
    w_y = grad_w[:, 1:2]

    von_karman = physics.constants.von_karman_constant
    z_0 = physics.constants.surface_roughness_length
    z_1 = physics.constants.surface_reference_height
    drag_coefficient = (von_karman**2 / math.log((z_1 + z_0) / z_0) ** 2)
    horizontal_speed_1 = torch.sqrt(u_1.square() + v_1.square() + 1.0e-12)

    # Ordinary SGS stress evaluated on the fluid side of the wall.
    tau_xz_wall = rho_wall * k_m_wall * (u_z + w_x)
    tau_yz_wall = rho_wall * k_m_wall * (v_z + w_y)

    tau_xz_surface = -drag_coefficient * rho_1 * u_1 * horizontal_speed_1
    tau_yz_surface = -drag_coefficient * rho_1 * v_1 * horizontal_speed_1

    return {
        "no_penetration_w": w_wall / scaling.w.scale, # z = 0 and z = z_max
        "surface_stress_xz": ((tau_xz_wall + tau_xz_surface) / surface_stress_scale)[bottom_mask], # z = 0
        "surface_stress_yz": ((tau_yz_wall + tau_yz_surface) / surface_stress_scale)[bottom_mask], # z = 0
    }
