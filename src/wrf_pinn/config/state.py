"""State variable definitions for Cartesian WRF PINN experiments.

This module defines the variable contract shared by the model, data loaders,
physics residuals, and training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CoordinateName = Literal["x", "y", "z", "t"]

PrognosticName = Literal[
    "u",
    "v",
    "w",
    "theta",
    "p",
    "rho",
    "qv",
]

DiagnosticName = Literal[
    "temperature",
    "pressure_gradient_x",
    "pressure_gradient_y",
    "pressure_gradient_z",
    "divergence",
    "wind_speed",
    "virtual_temperature",
    "buoyancy",
]


@dataclass(frozen=True)
class StateSpec:
    """Names and ordering for the PINN input and output state."""

    coordinates: tuple[CoordinateName, ...] = ("x", "y", "z", "t")
    prognostic: tuple[PrognosticName, ...] = (
        "u",
        "v",
        "w",
        "theta",
        "p",
        "rho",
        "qv",
    )
    diagnostic: tuple[DiagnosticName, ...] = (
        "temperature",
        "pressure_gradient_x",
        "pressure_gradient_y",
        "pressure_gradient_z",
        "divergence",
        "wind_speed",
        "virtual_temperature",
        "buoyancy",
    )

    @property
    def input_dim(self) -> int:
        """Number of neural-network input coordinates."""

        return len(self.coordinates)

    @property
    def output_dim(self) -> int:
        """Number of neural-network prognostic outputs."""

        return len(self.prognostic)

    def prognostic_index(self, name: PrognosticName) -> int:
        """Return the output-column index for a prognostic variable."""

        return self.prognostic.index(name)

    def coordinate_index(self, name: CoordinateName) -> int:
        """Return the input-column index for a coordinate variable."""

        return self.coordinates.index(name)


DEFAULT_STATE = StateSpec()
