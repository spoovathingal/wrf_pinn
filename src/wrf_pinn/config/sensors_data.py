"""Sensor-data configuration for WRF PINN experiments.

Sensor preprocessing is assumed to happen outside this codebase. By the time
sensor data reaches this package, it should already be collated, normalized, and
scaled into one CSV with columns ``x,y,z,t,u,v,w``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SensorDataFormat = Literal["csv"]


@dataclass(frozen=True)
class SensorColumnConfig:
    """Column names for normalized sensor coordinates and velocity values."""

    coordinates: tuple[str, str, str, str] = ("x", "y", "z", "t")
    values: tuple[str, str, str] = ("u", "v", "w")

    def __post_init__(self) -> None:
        if len(self.coordinates) != 4:
            raise ValueError(
                "Sensor coordinate columns must contain exactly four columns "
                f"for x,y,z,t; got {self.coordinates}."
            )

        if len(self.values) != 3:
            raise ValueError(
                "Sensor value columns must contain exactly three columns "
                f"for u,v,w; got {self.values}."
            )


@dataclass(frozen=True)
class SensorDataConfig:
    """Configuration for one normalized sensor CSV dataset."""

    path: str = ""
    file_format: SensorDataFormat = "csv"
    columns: SensorColumnConfig = SensorColumnConfig()


@dataclass(frozen=True)
class SensorsConfig:
    """Collection of normalized sensor datasets."""

    datasets: tuple[SensorDataConfig, ...] = ()

    @property
    def has_datasets(self) -> bool:
        """Return whether any sensor datasets are configured."""

        return bool(self.datasets)


DEFAULT_SENSORS = SensorsConfig()
