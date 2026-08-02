"""Boundary-condition residuals for WRF PINN training.

This module contains residual definitions for physical boundary constraints.
It does not read boundary geometry files, sample points, or assemble losses.
Those responsibilities belong to the data/sampling and training layers.
"""

from __future__ import annotations

import torch

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
        "no_slip_u": u,
        "no_slip_v": v,
        "no_slip_w": w,
    }


def no_penetration_z_wall_residuals(
    state: torch.Tensor,
    w_index: int = 2,
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING,
) -> dict[str, torch.Tensor]:
    """Return no-penetration residuals for a flat wall at ``z = 0``.

    For this special test case, the wall normal is assumed to be aligned with
    the vertical direction. The no-penetration condition is therefore:

    ``u dot n = w = 0``.

    The residual is computed in physical velocity units. If the model outputs
    are normalized, ``scaling`` maps ``w`` back to its physical value before the
    zero-penetration condition is evaluated.
    """

    _validate_state(state)
    num_state_variables = state.shape[1]
    _validate_indices((w_index,), num_state_variables, name="w")

    w = scaling.w.offset + scaling.w.scale * state[:, w_index]

    return {
        "no_penetration_w": w,
    }
