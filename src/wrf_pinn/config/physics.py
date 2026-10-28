"""Physics configuration for simplified Cartesian WRF PINN residuals.

This file describes which physical terms are active. It does not implement the
residual equations; the physics package will consume this configuration later.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal


CoordinateSystem = Literal["local_cartesian"]
PhysicsVariable = Literal["u", "v", "w", "theta", "p_prime", "q_v", "e_sgs"]
ResidualName = Literal["mass", "x_momentum", "y_momentum", "z_momentum",
                       "potential_temperature","water_vapor"]

@dataclass(frozen=True)
class PhysicalConstants:
    """Standard atmospheric physical constants.

    These constants are available to future residual and diagnostic modules.
    They are not automatically active; ``PhysicsConfig`` switches decide which
    physical terms are included in a particular experiment.
    """

    gravity: float = 9.81
    dry_air_gas_constant: float = 287.04
    water_vapor_gas_constant: float = 461.6
    dry_air_specific_heat_cp: float = 1005.04
    dry_air_specific_heat_cv: float = 718.0
    reference_pressure: float = 100000.0
    earth_rotation_rate: float = 7.2921e-5
    earth_radius: float = 6370000.0
    # K_m training not active not but to be implemented in future
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
    def gamma(self) -> float:
        """Return c_p / c_v."""

        return (self.dry_air_specific_heat_cp / self.dry_air_specific_heat_cv)

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
    """Configuration for reduced moist Cartesian equations with SGS closure.

    Seven fields are predicted, including positive SGS TKE. Six PDE residuals
    constrain mass, momentum, potential temperature, and water vapor.

    TKE enters the diagnostic SGS stresses and diffusivities, but has no
    separate transport-equation residual.
    """

    coordinate_system: CoordinateSystem = "local_cartesian"
    active_variables: tuple[PhysicsVariable, ...] = ("u", "v", "w", "theta",
                                                      "p_prime", "q_v", "e_sgs")
    residuals: tuple[ResidualName, ...] = (
        "mass",
        "x_momentum",
        "y_momentum",
        "z_momentum",
        "potential_temperature",
        "water_vapor",
    )
    include_coriolis: bool = False
    include_gravity: bool = True
    include_pressure_gradient: bool = True
    include_temperature: bool = True
    include_moisture: bool = True
    include_turbulence: bool = True
    include_microphysics: bool = False
    forcing_is_zero: bool = True
    constants: PhysicalConstants = PhysicalConstants()

    # Fixed hydrostatic base state for the current FastEddy case.
    hydrostatic_reference_potential_temperature: float = 300.0  # K

    # Multiplier taking the data's q_v to kg/kg for the thermodynamics
    # (1e-3 when the data is g/kg, as FastEddy writes it).
    q_v_to_kg_per_kg: float = 1.0



    # Native FE horizontal grid spacing in metres, not tile/subsample spacing.
    sgs_dx: float = 15.0115
    sgs_dy: float = 15.0115

    # Native reference-level centre heights in metres, before terrain
    # compression. These already include vertical stretching. from FE_CSTL_subset_400800.nc
    sgs_reference_z: tuple[float, ...] = (10.125386238098145, 30.385427474975586,
        50.67327880859375, 71.00747680664062, 91.40655517578125, 111.8890609741211,
        132.47354125976562, 153.1785125732422, 174.02252197265625, 195.02410888671875,
        216.20181274414062, 237.57418823242188, 259.1597595214844, 280.97705078125,
        303.04461669921875, 325.3809814453125, 348.0047302246094, 370.9343566894531,
        394.1883850097656, 417.7854309082031, 441.7439270019531, 466.0824890136719,
        490.81964111328125, 515.9739379882812, 541.5638427734375, 567.6079711914062,
        594.1248779296875, 621.1329956054688, 648.6509399414062, 676.697265625,
        705.2904663085938, 734.4491577148438, 764.1917114257812, 794.536865234375,
        825.5030517578125, 857.1087646484375, 889.3726806640625, 922.313232421875,
        955.948974609375, 990.2984619140625, 1025.3802490234375, 1061.2127685546875,
        1097.814697265625, 1135.20458984375, 1173.40087890625, 1212.422119140625,
        1252.286865234375, 1293.013671875, 1334.62109375, 1377.127685546875,
        1420.5518798828125,)

    # Record the spacing rule in checkpoints. The cropped-top rule is approximate.
    # Interior spacing: (z[k+1] - z[k-1]) / 2. Bottom: z[1] - z[0],
    # Cropped top: (3*z[-1] - 4*z[-2] + z[-3]) / 2, an approximation.
    sgs_vertical_edge_policy: Literal[
        "linear_bottom_quadratic_top"] = "linear_bottom_quadratic_top"

    # FastEddy diagnostic SGS closure.
    sgs_mixing_coefficient: float = 0.10          # c_k
    stable_mixing_length_coefficient: float = 0.76
    minimum_mixing_length: float = 0.01           # m

    def __post_init__(self) -> None:
        for name in ("sgs_dx", "sgs_dy"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        if self.sgs_vertical_edge_policy != "linear_bottom_quadratic_top":
            raise ValueError("Unsupported sgs_vertical_edge_policy.")

        # Keep the frozen config immutable/hashable even if loaded from a list.
        reference_z = tuple(float(z) for z in self.sgs_reference_z)
        object.__setattr__(self, "sgs_reference_z", reference_z)

        if len(reference_z) < 3:
            raise ValueError("sgs_reference_z must contain at least 3 levels.")

        if any(not math.isfinite(z) or z < 0.0 for z in reference_z):
            raise ValueError(
                "sgs_reference_z must contain finite, nonnegative heights."
            )

        if any(b <= a for a, b in zip(reference_z, reference_z[1:])):
            raise ValueError("sgs_reference_z must be strictly increasing.")

        top_spacing = (3.0 * reference_z[-1] - 4.0 * reference_z[-2]
            + reference_z[-3]) / 2.0

        if not math.isfinite(top_spacing) or top_spacing <= 0.0:
            raise ValueError(
                "Estimated top reference spacing must be finite and positive."
            )

    @property
    def state_dim(self) -> int:
        """Number of predicted state variables, including SGS TKE"""
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
