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
    boundary_residuals: TensorMap | None = None,
    inlet_errors: TensorMap | None = None,
    simulation_errors: TensorMap | None = None,
    sensor_errors: TensorMap | None = None,
    conditions: ConditionsConfig = DEFAULT_CONDITIONS,
) -> LossBreakdown:
    """Assemble the weighted PINN training loss.

    ``inlet_errors`` / ``simulation_errors`` / ``sensor_errors`` are the
    prediction-minus-target tensors for each data source (per the case's source
    tag). ``pde_residuals`` and ``boundary_residuals`` are physics/BC terms. Each
    term is weighted by its condition.
    """

    zero = _zero_like_available(
        pde_residuals, boundary_residuals,
        inlet_errors, simulation_errors, sensor_errors,
    )

    by_name = (
        ("pde", pde_residuals, conditions.pde),
        ("boundary", boundary_residuals, conditions.boundary),
        ("inlet", inlet_errors, conditions.inlet),
        ("simulation", simulation_errors, conditions.simulation),
        ("sensor", sensor_errors, conditions.sensor),
    )

    terms: TensorMap = {name: _group_loss(errors, spec, zero) for name, errors, spec in by_name}
    weighted_terms: TensorMap = {name: terms[name] * spec.weight for name, _, spec in by_name}
    total = torch.stack(tuple(weighted_terms.values())).sum()

    return LossBreakdown(total=total, terms=terms, weighted_terms=weighted_terms)
