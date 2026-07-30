"""Boundary-condition configuration for WRF PINN experiments.

At this stage, the only supported boundary constraint is a no-slip wall. Wall
boundary points are read from a CSV containing normalized space-time
coordinates, usually ``x,y,z,t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WallBoundaryPointFormat = Literal["csv"]


@dataclass(frozen=True)
class WallBoundaryPointConfig:
    """CSV source for no-slip wall space-time points."""

    path: str = ""
    file_format: WallBoundaryPointFormat = "csv"
    coordinate_columns: tuple[str, str, str, str] = ("x", "y", "z", "t")

    def __post_init__(self) -> None:
        if len(self.coordinate_columns) != 4:
            raise ValueError(
                "Wall boundary coordinate_columns must contain exactly four "
                f"columns; got {self.coordinate_columns}."
            )


@dataclass(frozen=True)
class NoSlipWallConfig:
    """Configuration for the no-slip wall residual definition."""

    points: WallBoundaryPointConfig = WallBoundaryPointConfig()
    velocity_components: tuple[str, ...] = ("u", "v", "w")

    def __post_init__(self) -> None:
        if not self.velocity_components:
            raise ValueError("No-slip wall requires at least one velocity component.")


@dataclass(frozen=True)
class BoundaryConfig:
    """Top-level boundary configuration."""

    no_slip_wall: NoSlipWallConfig = NoSlipWallConfig()


DEFAULT_BOUNDARIES = BoundaryConfig()
