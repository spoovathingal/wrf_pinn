"""Sampling configuration for WRF PINN training points.

This module describes how many points should be sampled from different parts of
the domain. It does not generate points; samplers belong in the data or training
packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wrf_pinn.config.boundaries import BOUNDARY_NAMES, BoundaryName


SamplingMethod = Literal["random_uniform", "latin_hypercube", "grid"]


@dataclass(frozen=True)
class CollocationSamplingConfig:
    """Interior points where PDE residuals are evaluated."""

    n_points: int = 10000
    method: SamplingMethod = "latin_hypercube"
    include_boundaries: bool = False

    def __post_init__(self) -> None:
        if self.n_points < 1:
            raise ValueError(f"n_points must be positive; got {self.n_points}.")


@dataclass(frozen=True)
class BoundarySamplingConfig:
    """Points sampled on named boundary faces."""

    n_points_per_boundary: int = 1000
    active_boundaries: tuple[BoundaryName, ...] = ("initial",)
    method: SamplingMethod = "random_uniform"

    def __post_init__(self) -> None:
        if self.n_points_per_boundary < 1:
            msg = (
                "n_points_per_boundary must be positive; "
                f"got {self.n_points_per_boundary}."
            )
            raise ValueError(msg)

        unknown = set(self.active_boundaries) - set(BOUNDARY_NAMES)
        if unknown:
            raise ValueError(f"Unknown active boundaries: {sorted(unknown)}.")


@dataclass(frozen=True)
class DataSamplingConfig:
    """Points sampled from discrete WRF, HRRR, or synthetic datasets."""

    n_points: int = 5000
    source_names: tuple[str, ...] = ("wrf", "hrrr")
    method: SamplingMethod = "random_uniform"
    require_all_sources: bool = False

    def __post_init__(self) -> None:
        if self.n_points < 1:
            raise ValueError(f"n_points must be positive; got {self.n_points}.")

        if not self.source_names:
            raise ValueError("At least one data source name is required.")


@dataclass(frozen=True)
class ValidationSamplingConfig:
    """Held-out points used to check model behavior during development."""

    n_points: int = 2000
    method: SamplingMethod = "random_uniform"

    def __post_init__(self) -> None:
        if self.n_points < 1:
            raise ValueError(f"n_points must be positive; got {self.n_points}.")


@dataclass(frozen=True)
class SamplingConfig:
    """Top-level sampling configuration for one experiment."""

    collocation: CollocationSamplingConfig = CollocationSamplingConfig()
    boundary: BoundarySamplingConfig = BoundarySamplingConfig()
    data: DataSamplingConfig = DataSamplingConfig()
    validation: ValidationSamplingConfig = ValidationSamplingConfig()
    seed: int = 42

    @property
    def total_requested_points(self) -> int:
        """Return the total nominal number of requested sample points."""

        return (
            self.collocation.n_points
            + self.boundary.n_points_per_boundary
            * len(self.boundary.active_boundaries)
            + self.data.n_points
            + self.validation.n_points
        )


DEFAULT_SAMPLING = SamplingConfig()
