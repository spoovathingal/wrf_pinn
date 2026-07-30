"""Boundary-condition residuals for WRF PINN training.

This module contains residual definitions for physical boundary constraints.
It does not read boundary geometry files, sample points, or assemble losses.
Those responsibilities belong to the data/sampling and training layers.
"""

from __future__ import annotations

import torch


def no_slip_wall_residuals(
    state: torch.Tensor,
    velocity_indices: tuple[int, int, int] = (0, 1, 2),
) -> dict[str, torch.Tensor]:
    """Return no-slip wall residuals from model predictions at wall points.

    The no-slip condition requires the velocity components to vanish at wall
    coordinates:

    ``u = 0, v = 0, w = 0``.

    Since the target value is zero, each residual is simply the corresponding
    predicted velocity component.

    Parameters
    ----------
    state:
        Model output evaluated at wall coordinates. Expected shape is
        ``(num_wall_points, num_state_variables)``.

    velocity_indices:
        Column indices for ``u``, ``v``, and ``w`` in the model output.

    Returns
    -------
    dict[str, torch.Tensor]
        Residual tensors that should be driven to zero at the wall.
    """

    if state.ndim != 2:
        raise ValueError(
            "state must be a 2D tensor with shape "
            "(num_wall_points, num_state_variables)."
        )

    if len(velocity_indices) != 3:
        raise ValueError(
            "velocity_indices must contain exactly three indices for u, v, w; "
            f"got {velocity_indices}."
        )

    u_index, v_index, w_index = velocity_indices
    num_state_variables = state.shape[1]
    for index in velocity_indices:
        if index < 0 or index >= num_state_variables:
            raise ValueError(
                "velocity index is outside the state tensor columns; "
                f"got index {index} for {num_state_variables} columns."
            )

    return {
        "no_slip_u": state[:, u_index],
        "no_slip_v": state[:, v_index],
        "no_slip_w": state[:, w_index],
    }
