"""Physics configuration for simplified Cartesian WRF PINN residuals.

This file describes which physical terms are active. It does not implement the
residual equations; the physics package will consume this configuration later.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal


CoordinateSystem = Literal["local_cartesian"]
PhysicsVariable = Literal["u", "v", "w", "theta", "p_prime", "k_m"]
ResidualName = Literal["mass", "x_momentum", "y_momentum", "z_momentum", "potential_temperature"]

@dataclass(frozen=True)
class PhysicalConstants:
    """Standard atmospheric physical constants.

    These constants are available to future residual and diagnostic modules.
    They are not automatically active; ``PhysicsConfig`` switches decide which
    physical terms are included in a particular experiment.
    """

    gravity: float = 9.81
    dry_air_gas_constant: float = 287.0
    water_vapor_gas_constant: float = 461.6
    dry_air_specific_heat_cp: float = 1004.0
    dry_air_specific_heat_cv: float = 717.0
    reference_pressure: float = 100000.0
    earth_rotation_rate: float = 7.2921e-5
    earth_radius: float = 6370000.0
    eddy_viscosity_min: float = 0.0       # K_m lower bound [m^2/s]
    eddy_viscosity_max: float = 100.0     # K_m upper bound [m^2/s]
    eddy_viscosity_initial: float = 0.003853  # Initial guess only [m^2/s]
    von_karman_constant: float = 0.4 # kappa
    surface_roughness_length: float = 0.1  # z_0
    surface_reference_height: float = 7 # z_1
    # Check k_m positivity | numeric
    def __post_init__(self) -> None:
        values = {
            "eddy_viscosity_min": self.eddy_viscosity_min,
            "eddy_viscosity_max": self.eddy_viscosity_max,
            "eddy_viscosity_initial": self.eddy_viscosity_initial,
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value}.")

        if self.eddy_viscosity_min < 0.0:
            raise ValueError(
                "eddy_viscosity_min must be nonnegative; "
                f"got {self.eddy_viscosity_min}."
            )

        if self.eddy_viscosity_max <= self.eddy_viscosity_min:
            raise ValueError(
                "eddy_viscosity_max must be greater than "
                "eddy_viscosity_min."
            )

        if not (
            self.eddy_viscosity_min
            < self.eddy_viscosity_initial
            < self.eddy_viscosity_max
        ):
            raise ValueError(
                "eddy_viscosity_initial must lie between "
                "eddy_viscosity_min and eddy_viscosity_max."
            )

    @property
    def kappa(self) -> float:
        """Return R_d / c_p for dry air."""

        return self.dry_air_gas_constant / self.dry_air_specific_heat_cp

    @property
    def epsilon(self) -> float:
        """Return R_d / R_v for dry air and water vapor."""

        return self.dry_air_gas_constant / self.water_vapor_gas_constant


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for the dry neutral boundary layer model.

    Assumptions:
    - local Cartesian coordinates
    - zero external forcing
    - dry atmospheric thermodynamics
    - varying momentum eddy viscosity
    - density is derived from theta and p_prime
    - active state is u, v, w, theta, p_prime, k_m
    """

    coordinate_system: CoordinateSystem = "local_cartesian"
    active_variables: tuple[PhysicsVariable, ...] = ("u", "v", "w", "theta", "p_prime", "k_m")
    residuals: tuple[ResidualName, ...] = (
        "mass",
        "x_momentum",
        "y_momentum",
        "z_momentum",
        "potential_temperature",
    )
    include_coriolis: bool = False
    include_gravity: bool = True
    include_pressure_gradient: bool = True
    include_temperature: bool = True
    include_moisture: bool = False
    include_turbulence: bool = True
    include_microphysics: bool = False
    forcing_is_zero: bool = True
    constants: PhysicalConstants = PhysicalConstants()

    @property
    def state_dim(self) -> int:
        """Number of active prognostic variables required by this physics config."""

        return len(self.active_variables)

    @property
    def residual_dim(self) -> int:
        """Number of residual equations produced by this physics config."""

        return len(self.residuals)

    def variable_index(self, name: PhysicsVariable) -> int:
        """Return the active-state column index for a physics variable."""

        return self.active_variables.index(name)

    def residual_index(self, name: ResidualName) -> int:
        """Return the residual column index for a residual equation."""

        return self.residuals.index(name)


DEFAULT_PHYSICS = PhysicsConfig()
