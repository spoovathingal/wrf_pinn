"""Global loss-mode configuration for WRF PINN experiments.

Controls which loss modes are active and how strongly each contributes. The two
data modes (``simulation``, ``sensor``) mirror the pre-processor source tags, so
a case's rows are weighted per source. It does not read data, evaluate residuals,
or assemble losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConditionName = Literal["pde", "boundary", "simulation", "sensor"]
ReductionName = Literal["mean", "sum"]

#: Data condition names, in source-tag order (map to source codes 0, 1).
DATA_CONDITIONS: tuple[ConditionName, ...] = ("simulation", "sensor")


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
    simulation: ConditionSpec = ConditionSpec("simulation", active=True, weight=1.0)
    sensor: ConditionSpec = ConditionSpec("sensor", active=True, weight=1.0)

    @property
    def active(self) -> tuple[ConditionSpec, ...]:
        """Return active condition specs in canonical training order."""

        return tuple(condition for condition in self.as_tuple() if condition.active)

    def as_tuple(self) -> tuple[ConditionSpec, ...]:
        """Return all condition specs in canonical training order."""

        return (self.pde, self.boundary, self.simulation, self.sensor)

    def weights(self) -> dict[ConditionName, float]:
        """Return condition weights keyed by condition name."""

        return {condition.name: condition.weight for condition in self.as_tuple()}


DEFAULT_CONDITIONS = ConditionsConfig()
