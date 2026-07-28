"""Loss assembly for WRF PINN training.

This module combines already-computed errors into a weighted training loss.
Residual evaluation, data interpolation, and condition evaluation live outside
this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wrf_pinn.config.conditions import DEFAULT_CONDITIONS, ConditionSpec
from wrf_pinn.config.conditions import ConditionsConfig


TensorMap = dict[str, torch.Tensor]


@dataclass(frozen=True)
class LossBreakdown:
    """Weighted total loss and individual unweighted loss terms."""

    total: torch.Tensor
    terms: TensorMap
    weighted_terms: TensorMap


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
    data_errors: TensorMap | None = None,
    initial_errors: TensorMap | None = None,
    boundary_errors: TensorMap | None = None,
    conditions: ConditionsConfig = DEFAULT_CONDITIONS,
) -> LossBreakdown:
    """Assemble the weighted PINN training loss.

    Parameters
    ----------
    pde_residuals:
        Residual tensors such as ``mass``, ``x_momentum``, ``y_momentum``, and
        ``z_momentum``.
    data_errors:
        Prediction-minus-measurement tensors for WRF, HRRR, or observation data.
    initial_errors:
        Prediction-minus-target tensors at the initial time boundary.
    boundary_errors:
        Prediction-minus-target tensors on active spatial or temporal boundaries.
    conditions:
        Condition activation, weighting, and reduction configuration.

    Returns
    -------
    LossBreakdown
        Total weighted loss plus unweighted and weighted component terms.
    """

    zero = _zero_like_available(
        pde_residuals,
        data_errors,
        initial_errors,
        boundary_errors,
    )

    terms: TensorMap = {
        "pde": _group_loss(pde_residuals, conditions.pde, zero),
        "initial": _group_loss(initial_errors, conditions.initial, zero),
        "boundary": _group_loss(boundary_errors, conditions.boundary, zero),
        "data": _group_loss(data_errors, conditions.data, zero),
        "regularization": zero,
    }
    weighted_terms: TensorMap = {
        name: terms[name] * condition.weight
        for name, condition in (
            ("pde", conditions.pde),
            ("initial", conditions.initial),
            ("boundary", conditions.boundary),
            ("data", conditions.data),
            ("regularization", conditions.regularization),
        )
    }
    total = torch.stack(tuple(weighted_terms.values())).sum()

    return LossBreakdown(
        total=total,
        terms=terms,
        weighted_terms=weighted_terms,
    )
