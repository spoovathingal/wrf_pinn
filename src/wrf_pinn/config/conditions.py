"""Global loss-mode configuration for WRF PINN experiments.

This module controls which major training modes are active and how strongly
each mode contributes to the total loss. It does not read data, evaluate
residuals, or assemble losses; those responsibilities live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConditionName = Literal[
    "pde",
    "boundary",
    "sensor_data",
    "flow_field_data",
]
ReductionName = Literal["mean", "sum"]


@dataclass(frozen=True)
class ConditionSpec:
    """Activation, weight, and reduction for one loss mode."""

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
    """Top-level configuration for active global loss modes."""

    pde: ConditionSpec = ConditionSpec("pde", active=True, weight=1.0)
    boundary: ConditionSpec = ConditionSpec("boundary", active=False, weight=0.0)
    sensor_data: ConditionSpec = ConditionSpec("sensor_data", active=False, weight=0.0)
    flow_field_data: ConditionSpec = ConditionSpec(
        "flow_field_data",
        active=True,
        weight=1.0,
    )

    @property
    def active(self) -> tuple[ConditionSpec, ...]:
        """Return active condition specs in canonical training order."""

        return tuple(condition for condition in self.as_tuple() if condition.active)

    def as_tuple(self) -> tuple[ConditionSpec, ...]:
        """Return all condition specs in canonical training order."""

        return (
            self.pde,
            self.boundary,
            self.sensor_data,
            self.flow_field_data,
        )

    def weights(self) -> dict[ConditionName, float]:
        """Return condition weights keyed by condition name."""

        return {condition.name: condition.weight for condition in self.as_tuple()}


DEFAULT_CONDITIONS = ConditionsConfig()
