"""Data-role configuration for WRF PINN experiments.

Data is organized by how it enters training rather than by where it came from.
For example, HRRR may provide initial conditions, LES may provide dense flow
fields, and sensors may provide sparse measurements. Concrete readers live in
the ``wrf_pinn.data`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DataFormat = Literal["csv", "netcdf", "zarr", "grib", "numpy"]
InterpolationMethod = Literal["nearest", "linear"]


def _validate_path_when_active(role: str, path: str, active: bool) -> None:
    if active and not path:
        raise ValueError(f"Active {role} data requires a path.")


@dataclass(frozen=True)
class ColumnSpec:
    """Column names for coordinate and state-like tabular data."""

    coordinates: tuple[str, ...] = ("x", "y", "z", "t")
    values: tuple[str, ...] = ("u", "v", "w", "rho")

    def __post_init__(self) -> None:
        if not self.coordinates:
            raise ValueError("At least one coordinate column is required.")

        if not self.values:
            raise ValueError("At least one value column is required.")


@dataclass(frozen=True)
class InterpolationConfig:
    """Interpolation settings for mapping data to requested PINN points."""

    method: InterpolationMethod = "linear"
    extrapolate: bool = False
    fill_value: float | None = None


@dataclass(frozen=True)
class InitialConditionDataConfig:
    """Configuration for initial-condition data."""

    path: str = ""
    file_format: DataFormat = "csv"
    columns: ColumnSpec = ColumnSpec()
    interpolation: InterpolationConfig = InterpolationConfig()
    active: bool = False

    def __post_init__(self) -> None:
        _validate_path_when_active("initial condition", self.path, self.active)


@dataclass(frozen=True)
class BoundaryConditionDataConfig:
    """Configuration for boundary-condition data."""

    path: str = ""
    file_format: DataFormat = "csv"
    boundary_name: str = ""
    columns: ColumnSpec = ColumnSpec()
    interpolation: InterpolationConfig = InterpolationConfig()
    active: bool = False

    def __post_init__(self) -> None:
        _validate_path_when_active("boundary condition", self.path, self.active)
        if self.active and not self.boundary_name:
            raise ValueError("Active boundary-condition data requires boundary_name.")


@dataclass(frozen=True)
class FlowFieldDataConfig:
    """Configuration for dense 3D or 4D training flow-field data."""

    path: str = ""
    file_format: DataFormat = "csv"
    columns: ColumnSpec = ColumnSpec()
    interpolation: InterpolationConfig = InterpolationConfig()
    active: bool = False

    def __post_init__(self) -> None:
        _validate_path_when_active("flow-field", self.path, self.active)


@dataclass(frozen=True)
class SensorDataConfig:
    """Configuration for sparse sensor measurements."""

    path: str = ""
    file_format: DataFormat = "csv"
    columns: ColumnSpec = ColumnSpec(
        coordinates=("x", "y", "z", "t"),
        values=("u", "v", "w"),
    )
    interpolation: InterpolationConfig = InterpolationConfig(method="nearest")
    active: bool = False

    def __post_init__(self) -> None:
        _validate_path_when_active("sensor", self.path, self.active)


@dataclass(frozen=True)
class DataConfig:
    """Top-level data configuration grouped by training role."""

    initial_condition: InitialConditionDataConfig = InitialConditionDataConfig()
    boundary_conditions: tuple[BoundaryConditionDataConfig, ...] = ()
    flow_field: FlowFieldDataConfig = FlowFieldDataConfig()
    sensors: tuple[SensorDataConfig, ...] = ()

    @property
    def has_initial_condition(self) -> bool:
        """Return whether initial-condition data is active."""

        return self.initial_condition.active

    @property
    def active_boundary_conditions(self) -> tuple[BoundaryConditionDataConfig, ...]:
        """Return active boundary-condition datasets."""

        return tuple(boundary for boundary in self.boundary_conditions if boundary.active)

    @property
    def has_flow_field(self) -> bool:
        """Return whether dense flow-field data is active."""

        return self.flow_field.active

    @property
    def active_sensors(self) -> tuple[SensorDataConfig, ...]:
        """Return active sensor datasets."""

        return tuple(sensor for sensor in self.sensors if sensor.active)

DEFAULT_DATA = DataConfig()
