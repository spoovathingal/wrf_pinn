"""Physics configuration for simplified Cartesian WRF PINN residuals.

This file describes which physical terms are active. It does not implement the
residual equations; the physics package will consume this configuration later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CoordinateSystem = Literal["local_cartesian"]
PhysicsVariable = Literal["u", "v", "w", "theta", "p_prime"]
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
    eddy_viscosity: float = 0.003853  # K_m [m^2/s]

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
    - constant momentum eddy viscosity
    - density is derived from theta and p_prime
    - active state is u, v, w, theta, and p_prime
    """

    coordinate_system: CoordinateSystem = "local_cartesian"
    active_variables: tuple[PhysicsVariable, ...] = ("u", "v", "w", "theta", "p_prime")
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
