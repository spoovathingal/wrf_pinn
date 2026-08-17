"""Boundary-condition configuration for WRF PINN experiments.

At this stage, the only supported boundary constraint is a no-slip wall. Wall
surface geometry is read from a CSV containing normalized spatial coordinates,
usually ``x,y,z``. Time-dependent boundary training points can be generated
later by combining this surface geometry with a training-time sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WallSurfaceFormat = Literal["csv"]

#: Which wall constraint the boundary residual enforces. ``no_slip`` drives all
#: velocity components to zero on the wall; ``no_penetration_z`` only constrains
#: the wall-normal (z) component.
WallCondition = Literal["no_slip", "no_penetration_z"]


@dataclass(frozen=True)
class WallSurfaceConfig:
    """CSV source for no-slip wall surface geometry."""

    path: str = ""
    file_format: WallSurfaceFormat = "csv"
    coordinate_columns: tuple[str, str, str] = ("x", "y", "z")

    def __post_init__(self) -> None:
        if len(self.coordinate_columns) != 3:
            raise ValueError(
                "Wall surface coordinate_columns must contain exactly three "
                f"columns; got {self.coordinate_columns}."
            )


@dataclass(frozen=True)
class NoSlipWallConfig:
    """Configuration for the wall residual definition.

    ``condition`` selects which wall constraint the residual enforces, so the
    boundary behavior is self-describing here rather than being toggled by a
    loose boolean at the training call site.
    """

    surface: WallSurfaceConfig = WallSurfaceConfig()
    velocity_components: tuple[str, ...] = ("u", "v", "w")
    condition: WallCondition = "no_slip"

    def __post_init__(self) -> None:
        if not self.velocity_components:
            raise ValueError("No-slip wall requires at least one velocity component.")


@dataclass(frozen=True)
class BoundaryConfig:
    """Top-level boundary configuration."""

    no_slip_wall: NoSlipWallConfig = NoSlipWallConfig()


DEFAULT_BOUNDARIES = BoundaryConfig()
