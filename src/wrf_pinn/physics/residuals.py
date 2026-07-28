"""Residual equations for the simplified Cartesian WRF PINN model.

This module implements the reduced system described in
``wrf_pinn.config.physics``:

- local Cartesian coordinates
- zero forcing
- no temperature or moisture equations
- active state variables are u, v, w, and rho

The residuals are evaluated at continuous PINN collocation points. Inputs are
expected in coordinate order (x, y, z, t), and state outputs are expected in
physics-variable order (u, v, w, rho).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig


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
    )[0]

    if gradient is None:
        raise RuntimeError("Could not compute gradient for residual evaluation.")

    return gradient


def _split_state(
    state: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return u, v, w, and rho columns from the model state output."""

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
    rho = state[
        :,
        physics.variable_index("rho") : physics.variable_index("rho") + 1,
    ]
    return u, v, w, rho


def _validate_physics_config(physics: PhysicsConfig) -> None:
    """Ensure this implementation is used only for the supported reduced system."""

    if physics.coordinate_system != "local_cartesian":
        raise ValueError("Only local_cartesian coordinates are supported.")

    if physics.active_variables != ("u", "v", "w", "rho"):
        raise ValueError("Residuals currently require active variables (u, v, w, rho).")

    if not physics.forcing_is_zero:
        raise ValueError("Residuals currently assume zero forcing.")

    unsupported_terms: Mapping[str, bool] = {
        "include_coriolis": physics.include_coriolis,
        "include_gravity": physics.include_gravity,
        "include_pressure_gradient": physics.include_pressure_gradient,
        "include_temperature": physics.include_temperature,
        "include_moisture": physics.include_moisture,
        "include_turbulence": physics.include_turbulence,
        "include_microphysics": physics.include_microphysics,
    }
    enabled_terms = [name for name, enabled in unsupported_terms.items() if enabled]
    if enabled_terms:
        raise ValueError(
            "Residuals currently require disabled physics terms; "
            f"enabled terms: {enabled_terms}."
        )


def cartesian_zero_forcing_residuals(
    coordinates: torch.Tensor,
    state: torch.Tensor,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
) -> TensorDict:
    """Compute reduced Cartesian continuity and momentum residuals.

    Parameters
    ----------
    coordinates:
        Collocation coordinates with shape ``(n_points, 4)`` and column order
        ``(x, y, z, t)``. The tensor must have ``requires_grad=True``.
    state:
        Model outputs with shape ``(n_points, 4)`` and column order
        ``(u, v, w, rho)``.
    physics:
        Physics configuration. Only the default reduced configuration is
        supported by this implementation.

    Returns
    -------
    dict[str, torch.Tensor]
        Residual tensors keyed by ``mass``, ``x_momentum``, ``y_momentum``,
        and ``z_momentum``. Each tensor has shape ``(n_points, 1)``.
    """

    _validate_physics_config(physics)

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_points, 4).")

    if not coordinates.requires_grad:
        raise ValueError("coordinates must have requires_grad=True.")

    u, v, w, rho = _split_state(state, physics)

    grad_u = _gradient(u, coordinates)
    grad_v = _gradient(v, coordinates)
    grad_w = _gradient(w, coordinates)
    grad_rho = _gradient(rho, coordinates)
    grad_rho_u = _gradient(rho * u, coordinates)
    grad_rho_v = _gradient(rho * v, coordinates)
    grad_rho_w = _gradient(rho * w, coordinates)

    u_x, u_y, u_z, u_t = grad_u.split(1, dim=1)
    v_x, v_y, v_z, v_t = grad_v.split(1, dim=1)
    w_x, w_y, w_z, w_t = grad_w.split(1, dim=1)
    _, _, _, rho_t = grad_rho.split(1, dim=1)
    rho_u_x, _, _, _ = grad_rho_u.split(1, dim=1)
    _, rho_v_y, _, _ = grad_rho_v.split(1, dim=1)
    _, _, rho_w_z, _ = grad_rho_w.split(1, dim=1)

    advect_u = u * u_x + v * u_y + w * u_z
    advect_v = u * v_x + v * v_y + w * v_z
    advect_w = u * w_x + v * w_y + w * w_z

    return {
        "mass": rho_t + rho_u_x + rho_v_y + rho_w_z,
        "x_momentum": u_t + advect_u,
        "y_momentum": v_t + advect_v,
        "z_momentum": w_t + advect_w,
    }
