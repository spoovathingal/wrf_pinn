"""Training-condition configuration for WRF PINN experiments.

This module describes which constraints are active during training and how they
are weighted. The current default recipe uses dense flow-field data and PDE
residuals only. It does not compute losses; loss assembly belongs in the
training package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConditionName = Literal[
    "pde",
    "initial",
    "boundary",
    "data",
    "regularization",
]
ReductionName = Literal["mean", "sum"]


@dataclass(frozen=True)
class ConditionSpec:
    """Configuration for one training condition."""

    name: ConditionName
    active: bool = True
    weight: float = 1.0
    reduction: ReductionName = "mean"

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"{self.name} weight must be nonnegative.")

        if not self.active and self.weight != 0.0:
            raise ValueError(f"{self.name} must have weight 0.0 when inactive.")


@dataclass(frozen=True)
class ConditionsConfig:
    """Top-level configuration for active PINN training conditions."""

    pde: ConditionSpec = ConditionSpec("pde", active=True, weight=1.0)
    initial: ConditionSpec = ConditionSpec("initial", active=False, weight=0.0)
    boundary: ConditionSpec = ConditionSpec("boundary", active=False, weight=0.0)
    data: ConditionSpec = ConditionSpec("data", active=True, weight=1.0)
    regularization: ConditionSpec = ConditionSpec(
        "regularization",
        active=False,
        weight=0.0,
    )

    @property
    def active(self) -> tuple[ConditionSpec, ...]:
        """Return active condition specs in canonical training order."""

        return tuple(condition for condition in self.as_tuple() if condition.active)

    def as_tuple(self) -> tuple[ConditionSpec, ...]:
        """Return all condition specs in canonical training order."""

        return (
            self.pde,
            self.initial,
            self.boundary,
            self.data,
            self.regularization,
        )

    def weights(self) -> dict[ConditionName, float]:
        """Return condition weights keyed by condition name."""

        return {condition.name: condition.weight for condition in self.as_tuple()}


DEFAULT_CONDITIONS = ConditionsConfig()
