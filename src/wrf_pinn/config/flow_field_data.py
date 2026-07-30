"""Flowfield-data configuration for WRF PINN experiments.

Flowfield data is assumed to be preprocessed outside this codebase. By the time
it reaches this package, it should already be collated, normalized, and scaled
into a CSV with columns ``x,y,z,t,u,v,w,rho`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FlowFieldDataFormat = Literal["csv"]


@dataclass(frozen=True)
class FlowFieldColumnConfig:
    """Column names for normalized dense flowfield coordinates and values."""

    coordinates: tuple[str, str, str, str] = ("x", "y", "z", "t")
    values: tuple[str, str, str, str] = ("u", "v", "w", "rho")

    def __post_init__(self) -> None:
        if len(self.coordinates) != 4:
            raise ValueError(
                "Flowfield coordinate columns must contain exactly four "
                f"columns for x,y,z,t; got {self.coordinates}."
            )

        if len(self.values) != 4:
            raise ValueError(
                "Flowfield value columns must contain exactly four columns "
                f"for u,v,w,rho; got {self.values}."
            )


@dataclass(frozen=True)
class FlowFieldDataConfig:
    """Configuration for one normalized dense flowfield CSV."""

    path: str = ""
    file_format: FlowFieldDataFormat = "csv"
    columns: FlowFieldColumnConfig = FlowFieldColumnConfig()


DEFAULT_FLOW_FIELD_DATA = FlowFieldDataConfig()
