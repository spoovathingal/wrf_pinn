"""Loss assembly for WRF PINN training.

This module combines already-computed errors into a weighted training loss.
Residual evaluation, data interpolation, and condition evaluation live outside
this file.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch

from wrf_pinn.config.conditions import DEFAULT_CONDITIONS, ConditionSpec
from wrf_pinn.config.conditions import ConditionsConfig


TensorMap = dict[str, torch.Tensor]

PDE_RESIDUAL_NAMES: tuple[str, ...] = ("mass", "x_momentum", "y_momentum", "z_momentum",
                                       "potential_temperature")

@dataclass(frozen=True)
class PDEResidualScales:
    """Fixed characteristic scales used to nondimensionalize PDE residuals."""

    mass: float = 1.0
    x_momentum: float = 1.0
    y_momentum: float = 1.0
    z_momentum: float = 1.0
    potential_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in PDE_RESIDUAL_NAMES:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"PDE residual scale {name} must be finite and positive; "
                    f"got {value}."
                )

    def scale_for(self, name: str) -> float:
        """Return the characteristic scale for one PDE residual."""

        if name not in PDE_RESIDUAL_NAMES:
            raise ValueError(f"Unknown PDE residual name: {name}.")

        return float(getattr(self, name))

DEFAULT_PDE_RESIDUAL_SCALES = PDEResidualScales()

@dataclass(frozen=True)
class LossBreakdown:
    """Weighted total loss and individual unweighted loss terms."""

    total: torch.Tensor
    terms: TensorMap
    weighted_terms: TensorMap
    pde_raw_mse: TensorMap
    pde_scaled_mse: TensorMap


def _zero_like_available(*groups: TensorMap | None) -> torch.Tensor:
    """Return a scalar zero tensor matching the first available input tensor."""

    for group in groups:
        if group is None:
            continue

        for value in group.values():
            return value.new_zeros(())

    return torch.tensor(0.0)


def _reduce_tensor(value: torch.Tensor, spec: ConditionSpec) -> torch.Tensor:
    """Reduce one tensor according to a condition spec."""

    squared = value.square()
    if spec.reduction == "mean":
        return squared.mean()
    if spec.reduction == "sum":
        return squared.sum()

    raise ValueError(f"Unsupported reduction: {spec.reduction}.")


def _group_loss(
    errors: TensorMap | None,
    spec: ConditionSpec,
    zero: torch.Tensor,
) -> torch.Tensor:
    """Compute one condition loss from a dictionary of error tensors."""

    if not spec.active:
        return zero

    if errors is None or not errors:
        raise ValueError(f"{spec.name} condition is active but no errors were given.")

    reduced = [_reduce_tensor(error, spec) for error in errors.values()]
    return torch.stack(reduced).mean()


def assemble_pinn_loss(
    *,
    pde_residuals: TensorMap | None = None,
    boundary_residuals: TensorMap | None = None,
    sensor_data_errors: TensorMap | None = None,
    flow_field_data_errors: TensorMap | None = None,
    conditions: ConditionsConfig = DEFAULT_CONDITIONS,
    pde_residual_scales: PDEResidualScales = DEFAULT_PDE_RESIDUAL_SCALES,
) -> LossBreakdown:
    """Assemble the weighted PINN training loss.

    Parameters
    ----------
    pde_residuals:
        Residual tensors such as ``mass``, ``x_momentum``, ``y_momentum``, and
        ``z_momentum``.
    boundary_residuals:
        Boundary-condition residual tensors such as no-slip wall velocity
        residuals.
    sensor_data_errors:
        Prediction-minus-measurement tensors for sparse sensor data.
    flow_field_data_errors:
        Prediction-minus-measurement tensors for dense flowfield data.
    conditions:
        Condition activation, weighting, and reduction configuration.

    Returns
    -------
    LossBreakdown
        Total weighted loss plus unweighted and weighted component terms.
    """
    scaled_pde_residuals: TensorMap | None = None
    pde_raw_mse: TensorMap = {}
    pde_scaled_mse: TensorMap = {}

    if pde_residuals is not None:
        scaled_pde_residuals = {}

        for name, residual in pde_residuals.items():
            scale = pde_residual_scales.scale_for(name)
            scaled_residual = residual / scale

            scaled_pde_residuals[name] = scaled_residual
            pde_raw_mse[name] = residual.square().mean()
            pde_scaled_mse[name] = scaled_residual.square().mean()

    zero = _zero_like_available(
        scaled_pde_residuals,
        boundary_residuals,
        sensor_data_errors,
        flow_field_data_errors,
    )

    terms: TensorMap = {
        "pde": _group_loss(scaled_pde_residuals, conditions.pde, zero),
        "boundary": _group_loss(boundary_residuals, conditions.boundary, zero),
        "sensor_data": _group_loss(sensor_data_errors, conditions.sensor_data, zero),
        "flow_field_data": _group_loss(
            flow_field_data_errors,
            conditions.flow_field_data,
            zero,
        ),
    }
    weighted_terms: TensorMap = {
        name: terms[name] * condition.weight
        for name, condition in (
            ("pde", conditions.pde),
            ("boundary", conditions.boundary),
            ("sensor_data", conditions.sensor_data),
            ("flow_field_data", conditions.flow_field_data),
        )
    }
    total = torch.stack(tuple(weighted_terms.values())).sum()

    return LossBreakdown(
        total=total,
        terms=terms,
        weighted_terms=weighted_terms,
        pde_raw_mse=pde_raw_mse,
        pde_scaled_mse=pde_scaled_mse,
    )
