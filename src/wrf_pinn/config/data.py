"""Data-source configuration for WRF PINN experiments.

This module describes which atmospheric datasets are available and how their
variables map onto the PINN state. It does not read files; concrete readers
belong in the ``wrf_pinn.data`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DataSourceType = Literal["wrf", "hrrr", "synthetic"]
InterpolationMethod = Literal["nearest", "linear"]
VerticalCoordinateType = Literal["cartesian_z", "wrf_eta", "pressure_level"]
HorizontalCoordinateType = Literal["cartesian_xy", "lat_lon", "projected"]


@dataclass(frozen=True)
class VariableMapping:
    """External data variable names mapped to the reduced PINN state."""

    u: str
    v: str
    w: str | None = None
    rho: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return mapping keyed by PINN state variable name."""

        return {
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "rho": self.rho,
        }


@dataclass(frozen=True)
class InterpolationConfig:
    """Interpolation choices for mapping gridded data to PINN points."""

    method: InterpolationMethod = "linear"
    extrapolate: bool = False
    fill_value: float | None = None


@dataclass(frozen=True)
class DataSourceConfig:
    """Configuration for one external or synthetic data source."""

    name: str
    source_type: DataSourceType
    path: str
    variables: VariableMapping
    time_variable: str
    horizontal_coordinates: HorizontalCoordinateType
    vertical_coordinate: VerticalCoordinateType
    x_variable: str | None = None
    y_variable: str | None = None
    z_variable: str | None = None
    latitude_variable: str | None = None
    longitude_variable: str | None = None
    projection: str | None = None
    interpolation: InterpolationConfig = InterpolationConfig()
    active: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Data source name cannot be empty.")

        if self.source_type != "synthetic" and not self.path:
            raise ValueError(f"{self.name} data source requires a path.")


@dataclass(frozen=True)
class DataConfig:
    """Collection of data sources used by an experiment."""

    sources: tuple[DataSourceConfig, ...] = ()

    @property
    def active_sources(self) -> tuple[DataSourceConfig, ...]:
        """Return enabled data sources."""

        return tuple(source for source in self.sources if source.active)

    def source_names(self) -> tuple[str, ...]:
        """Return configured data-source names."""

        return tuple(source.name for source in self.sources)

    def get_source(self, name: str) -> DataSourceConfig:
        """Return one data source by name."""

        for source in self.sources:
            if source.name == name:
                return source

        raise KeyError(f"Unknown data source: {name}.")


DEFAULT_WRF_SOURCE = DataSourceConfig(
    name="wrf",
    source_type="wrf",
    path="wrfout.nc",
    variables=VariableMapping(
        u="U",
        v="V",
        w="W",
        rho="RHO",
    ),
    time_variable="Time",
    horizontal_coordinates="cartesian_xy",
    vertical_coordinate="wrf_eta",
    x_variable="west_east",
    y_variable="south_north",
    z_variable="bottom_top",
    active=False,
)

DEFAULT_HRRR_SOURCE = DataSourceConfig(
    name="hrrr",
    source_type="hrrr",
    path="hrrr.grib2",
    variables=VariableMapping(
        u="UGRD",
        v="VGRD",
        w="VVEL",
        rho=None,
    ),
    time_variable="time",
    horizontal_coordinates="projected",
    vertical_coordinate="pressure_level",
    latitude_variable="latitude",
    longitude_variable="longitude",
    projection="lambert_conformal",
    active=False,
)

DEFAULT_DATA = DataConfig(
    sources=(
        DEFAULT_WRF_SOURCE,
        DEFAULT_HRRR_SOURCE,
    )
)
