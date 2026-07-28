"""Cartesian WRF domain configuration for PINN experiments.

The residual code should treat space and time as continuous variables, but WRF
and HRRR data arrive on discrete grids. These small configuration objects keep
those ideas separate: physical extents define the continuous PINN domain, while
optional grid spacing and grid counts document the numerical/data grid used to
sample that domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AxisName = Literal["x", "y", "z", "t"]


@dataclass(frozen=True)
class CoordinateRange:
    """Closed interval for one coordinate direction."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.maximum <= self.minimum:
            msg = (
                "CoordinateRange requires maximum > minimum; "
                f"got minimum={self.minimum}, maximum={self.maximum}."
            )
            raise ValueError(msg)

    @property
    def length(self) -> float:
        """Return the coordinate interval length."""

        return self.maximum - self.minimum

    @property
    def center(self) -> float:
        """Return the midpoint of the coordinate interval."""

        return 0.5 * (self.minimum + self.maximum)

    def normalize(self, value: float) -> float:
        """Map a physical coordinate value to [-1, 1]."""

        return 2.0 * (value - self.minimum) / self.length - 1.0

    def denormalize(self, value: float) -> float:
        """Map a normalized coordinate value in [-1, 1] back to physical units."""

        return self.minimum + 0.5 * (value + 1.0) * self.length


@dataclass(frozen=True)
class GridSpec:
    """Discrete Cartesian grid metadata associated with the continuous domain."""

    nx: int
    ny: int
    nz: int
    nt: int | None = None

    def __post_init__(self) -> None:
        required_counts = {"nx": self.nx, "ny": self.ny, "nz": self.nz}
        for name, value in required_counts.items():
            if value < 2:
                raise ValueError(f"{name} must be at least 2; got {value}.")

        if self.nt is not None and self.nt < 2:
            raise ValueError(f"nt must be at least 2 when provided; got {self.nt}.")


@dataclass(frozen=True)
class CartesianWRFDomain:
    """Continuous Cartesian domain for WRF-style atmospheric PINNs.

    Coordinates use SI-oriented physical units by convention:
    x, y, z are in meters and t is in seconds. The vertical coordinate is
    geometric height here; WRF eta-coordinate transforms can be layered on later
    without changing the rest of the training configuration.
    """

    x: CoordinateRange
    y: CoordinateRange
    z: CoordinateRange
    t: CoordinateRange
    grid: GridSpec | None = None

    @property
    def coordinate_names(self) -> tuple[AxisName, ...]:
        """Return coordinate ordering expected by samplers and models."""

        return ("x", "y", "z", "t")

    @property
    def spatial_extent(self) -> tuple[float, float, float]:
        """Return physical lengths in x, y, and z."""

        return (self.x.length, self.y.length, self.z.length)

    @property
    def time_extent(self) -> float:
        """Return the physical length of the training time window."""

        return self.t.length

    @property
    def grid_spacing(self) -> tuple[float, float, float] | None:
        """Return dx, dy, dz if grid metadata is available."""

        if self.grid is None:
            return None

        dx = self.x.length / (self.grid.nx - 1)
        dy = self.y.length / (self.grid.ny - 1)
        dz = self.z.length / (self.grid.nz - 1)
        return (dx, dy, dz)

    @property
    def time_step(self) -> float | None:
        """Return dt if temporal grid metadata is available."""

        if self.grid is None or self.grid.nt is None:
            return None

        return self.t.length / (self.grid.nt - 1)

    def normalize_point(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
    ) -> tuple[float, float, float, float]:
        """Map a physical space-time point to normalized coordinates."""

        return (
            self.x.normalize(x),
            self.y.normalize(y),
            self.z.normalize(z),
            self.t.normalize(t),
        )

    def denormalize_point(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
    ) -> tuple[float, float, float, float]:
        """Map a normalized space-time point back to physical coordinates."""

        return (
            self.x.denormalize(x),
            self.y.denormalize(y),
            self.z.denormalize(z),
            self.t.denormalize(t),
        )

    def as_bounds(self) -> dict[AxisName, tuple[float, float]]:
        """Return coordinate bounds in a sampler-friendly dictionary."""

        return {
            "x": (self.x.minimum, self.x.maximum),
            "y": (self.y.minimum, self.y.maximum),
            "z": (self.z.minimum, self.z.maximum),
            "t": (self.t.minimum, self.t.maximum),
        }


def make_cartesian_wrf_domain(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    nt: int | None = None,
) -> CartesianWRFDomain:
    """Build a Cartesian WRF domain from scalar bounds and optional grid counts."""

    grid_counts = (nx, ny, nz)
    if any(count is not None for count in grid_counts):
        if nx is None or ny is None or nz is None:
            raise ValueError("nx, ny, and nz must be provided together.")
        grid = GridSpec(nx=nx, ny=ny, nz=nz, nt=nt)
    elif nt is not None:
        raise ValueError("nt requires nx, ny, and nz grid counts.")
    else:
        grid = None

    return CartesianWRFDomain(
        x=CoordinateRange(x_min, x_max),
        y=CoordinateRange(y_min, y_max),
        z=CoordinateRange(z_min, z_max),
        t=CoordinateRange(t_min, t_max),
        grid=grid,
    )
