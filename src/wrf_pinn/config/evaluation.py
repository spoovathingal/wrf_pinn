"""Evaluation configuration for WRF PINN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wrf_pinn.config.physics import DEFAULT_PHYSICS


MetricName = Literal["rmse", "mae", "bias", "relative_l2"]
DiagnosticName = Literal[
    "pde_residual_norm",
    "mass_residual",
    "momentum_residual",
]
OutputFormat = Literal["csv", "json", "netcdf"]


@dataclass(frozen=True)
class SliceEvaluationConfig:
    """Horizontal or temporal slices to evaluate after training."""

    z_levels: tuple[float, ...] = ()
    times: tuple[float, ...] = ()


@dataclass(frozen=True)
class ProfileEvaluationConfig:
    """Vertical profiles to evaluate after training."""

    locations_xy: tuple[tuple[float, float], ...] = ()
    times: tuple[float, ...] = ()


@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level evaluation configuration."""

    fields: tuple[str, ...] = DEFAULT_PHYSICS.active_variables
    metrics: tuple[MetricName, ...] = ("rmse", "mae", "bias", "relative_l2")
    diagnostics: tuple[DiagnosticName, ...] = (
        "pde_residual_norm",
        "mass_residual",
        "momentum_residual",
    )
    data_sources: tuple[str, ...] = ("wrf", "hrrr")
    output_dir: str = "outputs/evaluation"
    output_formats: tuple[OutputFormat, ...] = ("csv", "json")
    slices: SliceEvaluationConfig = SliceEvaluationConfig()
    profiles: ProfileEvaluationConfig = ProfileEvaluationConfig()

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("At least one evaluation field is required.")

        if not self.metrics:
            raise ValueError("At least one evaluation metric is required.")

        if not self.output_dir:
            raise ValueError("output_dir cannot be empty.")

        if not self.output_formats:
            raise ValueError("At least one output format is required.")


DEFAULT_EVALUATION = EvaluationConfig()
