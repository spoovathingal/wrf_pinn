"""Sampling configuration for WRF PINN training.

This module controls sample counts and sampling methods only. It does not
decide whether a loss term is active and it does not set lambda weights. Those
global objective controls live in ``config.conditions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CollocationSamplingMethod = Literal["random_uniform", "latin_hypercube", "grid"]
DatasetSamplingMethod = Literal["random", "sequential", "all"]


def _validate_optional_n_points(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive when provided; got {value}.")


@dataclass(frozen=True)
class CollocationSamplingConfig:
    """Sampling settings for PDE residual collocation points."""

    n_points: int = 10000
    method: CollocationSamplingMethod = "latin_hypercube"

    def __post_init__(self) -> None:
        if self.n_points < 1:
            raise ValueError(f"collocation n_points must be positive; got {self.n_points}.")


@dataclass(frozen=True)
class BoundarySamplingConfig:
    """Sampling settings for wall boundary coordinate rows."""

    n_points: int | None = None
    method: DatasetSamplingMethod = "all"

    def __post_init__(self) -> None:
        _validate_optional_n_points("boundary n_points", self.n_points)


@dataclass(frozen=True)
class SensorSamplingConfig:
    """Sampling settings for normalized sensor-data rows."""

    n_points: int | None = None
    method: DatasetSamplingMethod = "all"

    def __post_init__(self) -> None:
        _validate_optional_n_points("sensor n_points", self.n_points)


@dataclass(frozen=True)
class FlowFieldSamplingConfig:
    """Sampling settings for normalized dense flowfield rows."""

    n_points: int | None = None
    method: DatasetSamplingMethod = "all"

    def __post_init__(self) -> None:
        _validate_optional_n_points("flowfield n_points", self.n_points)


@dataclass(frozen=True)
class ValidationSamplingConfig:
    """Sampling settings for validation rows or points."""

    n_points: int | None = None
    method: DatasetSamplingMethod = "all"

    def __post_init__(self) -> None:
        _validate_optional_n_points("validation n_points", self.n_points)


@dataclass(frozen=True)
class SamplingConfig:
    """Top-level sampling configuration for one training run."""

    collocation: CollocationSamplingConfig = CollocationSamplingConfig()
    boundary: BoundarySamplingConfig = BoundarySamplingConfig()
    sensor_data: SensorSamplingConfig = SensorSamplingConfig()
    flow_field_data: FlowFieldSamplingConfig = FlowFieldSamplingConfig()
    validation: ValidationSamplingConfig = ValidationSamplingConfig()
    seed: int = 42

    @property
    def requested_finite_points(self) -> int:
        """Return finite sample counts explicitly requested by this config.

        Dataset configs with ``n_points=None`` mean "use all available rows" and
        are therefore not included in this count.
        """

        total = self.collocation.n_points
        for dataset_sampling in (
            self.boundary,
            self.sensor_data,
            self.flow_field_data,
            self.validation,
        ):
            if dataset_sampling.n_points is not None:
                total += dataset_sampling.n_points
        return total


DEFAULT_SAMPLING = SamplingConfig()
