"""Boundary-condition configuration for Cartesian WRF PINN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BoundaryName = Literal[
    "west",
    "east",
    "south",
    "north",
    "bottom",
    "top",
    "initial",
    "final",
]

BoundaryConditionType = Literal[
    "none",
    "periodic",
    "dirichlet",
    "neumann",
    "no_penetration",
    "open",
    "sponge",
    "data_forced",
]


BOUNDARY_NAMES: tuple[BoundaryName, ...] = (
    "west",
    "east",
    "south",
    "north",
    "bottom",
    "top",
    "initial",
    "final",
)


@dataclass(frozen=True)
class BoundaryConditionSpec:
    """Configuration for one named boundary condition."""

    name: BoundaryName
    condition_type: BoundaryConditionType
    variables: tuple[str, ...] = ()
    data_source: str | None = None
    weight: float = 1.0
    active: bool = True

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"Boundary weight must be nonnegative; got {self.weight}.")

        if self.condition_type == "data_forced" and self.data_source is None:
            msg = "data_forced boundary conditions require a data_source."
            raise ValueError(msg)


@dataclass(frozen=True)
class BoundaryConfig:
    """Collection of boundary-condition specs keyed by Cartesian domain face."""

    west: BoundaryConditionSpec = BoundaryConditionSpec("west", "none", active=False)
    east: BoundaryConditionSpec = BoundaryConditionSpec("east", "none", active=False)
    south: BoundaryConditionSpec = BoundaryConditionSpec("south", "none", active=False)
    north: BoundaryConditionSpec = BoundaryConditionSpec("north", "none", active=False)
    bottom: BoundaryConditionSpec = BoundaryConditionSpec("bottom", "none", active=False)
    top: BoundaryConditionSpec = BoundaryConditionSpec("top", "none", active=False)
    initial: BoundaryConditionSpec = BoundaryConditionSpec(
        "initial",
        "data_forced",
        data_source="initial_state",
    )
    final: BoundaryConditionSpec = BoundaryConditionSpec("final", "none", active=False)

    @property
    def names(self) -> tuple[BoundaryName, ...]:
        """Return all boundary names in canonical order."""

        return BOUNDARY_NAMES

    @property
    def active(self) -> tuple[BoundaryConditionSpec, ...]:
        """Return enabled boundary-condition specs."""

        return tuple(spec for spec in self.as_dict().values() if spec.active)

    def as_dict(self) -> dict[BoundaryName, BoundaryConditionSpec]:
        """Return boundary-condition specs keyed by boundary name."""

        return {
            "west": self.west,
            "east": self.east,
            "south": self.south,
            "north": self.north,
            "bottom": self.bottom,
            "top": self.top,
            "initial": self.initial,
            "final": self.final,
        }


DEFAULT_BOUNDARIES = BoundaryConfig()
