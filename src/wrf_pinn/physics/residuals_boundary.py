"""Boundary-condition residuals for WRF PINN training.

This module contains residual definitions for physical boundary constraints.
It does not read boundary geometry files, sample points, or assemble losses.
Those responsibilities belong to the data/sampling and training layers.
"""

from __future__ import annotations

import torch
import math

from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
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

def no_penetration_z_wall_residuals(
    coordinates: torch.Tensor,
    state: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
    *,
    terrain_height: torch.Tensor,
    terrain_slopes: torch.Tensor,
    no_penetration_scale: float | None = None,
    check_values: bool = True,
    raw_residuals: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Terrain-normal velocity only, evaluated at Z = h(x,y).

    terrain_height: (N,1), metres.
    terrain_slopes: (N,2), physical dh/dx and dh/dy.

    Returned loss residual is dimensionless.
    Logged raw residual is in m/s.

    check_values=False assumes all rows are finite terrain-surface points.
    """
    _validate_state(state)
    n = state.shape[0]

    if state.shape[1] != physics.state_dim:
        raise ValueError(f"Expected {physics.state_dim} state columns.")

    if coordinates.shape != (n, 4):
        raise ValueError("coordinates must have shape (N,4).")

    if terrain_height.shape != (n, 1):
        raise ValueError("terrain_height must have shape (N,1).")

    if terrain_slopes.shape != (n, 2):
        raise ValueError("terrain_slopes must have shape (N,2).")

    normal_scale = (
        scaling.w.scale
        if no_penetration_scale is None
        else no_penetration_scale
    )
    if not math.isfinite(normal_scale) or normal_scale <= 0.0:
        raise ValueError("no_penetration_scale must be finite and positive.")

    if check_values:
        for name, value in (
            ("coordinates", coordinates),
            ("state", state),
            ("terrain_height", terrain_height),
            ("terrain_slopes", terrain_slopes),
        ):
            if not torch.isfinite(value).all():
                raise ValueError( f"{name} must contain only finite values." )

        z = scaling.z.offset + scaling.z.scale * coordinates[:, 2:3]

        # Allow float32 roundoff in physical-height reconstruction.
        tolerance = max(
            1.0e-5,
            4.0 * torch.finfo(coordinates.dtype).eps
            * max(
                abs(scaling.z.offset), abs(scaling.z.scale), 1.0,
            ),
        )

        if not torch.allclose(
            z, terrain_height.to(z), rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                "All wall points must lie on the terrain surface Z=h."
            )

    indices = tuple(
        physics.variable_index(name) for name in ("u", "v", "w")
    )
    u, v, w = _physical_velocity_components(state, indices, scaling)

    slopes = terrain_slopes.to(state)
    hx, hy = slopes[:, 0], slopes[:, 1]

    normal_velocity = (
        (w - u * hx - v * hy) / torch.sqrt(1.0 + hx.square() + hy.square())
    )[:, None]

    if raw_residuals is not None:
        raw_residuals["no_penetration_w"] = normal_velocity.detach()

    return {"no_penetration_w": normal_velocity / normal_scale}