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
                       "potential_temperature","water_vapor", "sgs_tke"]

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
    active_variables: tuple[PhysicsVariable, ...] = ("u", "v", "w", "theta",
                                                      "p_prime", "q_v", "e_sgs")
    residuals: tuple[ResidualName, ...] = (
        "mass",
        "x_momentum",
        "y_momentum",
        "z_momentum",
        "potential_temperature",
        "water_vapor",
        "sgs_tke",
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

    # "uniform" preserves the previous closure. The driver will explicitly
    # select "fe_cell_volume" once all residual paths support local widths.
    sgs_filter_width_mode: Literal["uniform", "fe_cell_volume"] = "uniform"
    uniform_filter_width: float = 15.0  # m; used only in uniform mode

    # Native FE horizontal grid spacing in metres, not tile/subsample spacing.
    sgs_dx: float = 15.0115
    sgs_dy: float = 15.0115

    # Native reference-level centre heights in metres, before terrain
    # compression. These already include vertical stretching.
    # The driver will populate this from zPos and save the values in checkpoints.
    sgs_reference_z: tuple[float, ...] = ()

    # Interior spacing: (z[k+1] - z[k-1]) / 2.
    # Bottom: z[1] - z[0], consistent with linear extension below the first level.
    # Cropped top: (3*z[-1] - 4*z[-2] + z[-3]) / 2, an approximation.
    sgs_vertical_edge_policy: Literal[
        "linear_bottom_quadratic_top"
    ] = "linear_bottom_quadratic_top"

    # FastEddy diagnostic SGS closure.
    sgs_mixing_coefficient: float = 0.10          # c_k
    stable_mixing_length_coefficient: float = 0.76
    minimum_mixing_length: float = 0.01           # m
    # Prognostic SGS TKE dissipation:
    # c_e = a*c_k + (c0 - a*c_k)*(l_sgs/Delta)
    tke_dissipation_coefficient: float = 0.93  # c0
    tke_dissipation_mixing_factor: float = 1.9  # a
    # T = c_T * div(rho_d * K_m * grad(e_sgs))
    tke_transport_coefficient: float = 2.0  # c_T

    def __post_init__(self) -> None:
        if self.sgs_filter_width_mode not in ("uniform", "fe_cell_volume"):
            raise ValueError(
                "sgs_filter_width_mode must be 'uniform' or 'fe_cell_volume'."
            )

        for name in ("uniform_filter_width", "sgs_dx", "sgs_dy"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        if self.sgs_vertical_edge_policy != "linear_bottom_quadratic_top":
            raise ValueError("Unsupported sgs_vertical_edge_policy.")

        # Keep the frozen config immutable/hashable even if loaded from a list.
        reference_z = tuple(float(z) for z in self.sgs_reference_z)
        object.__setattr__(self, "sgs_reference_z", reference_z)

        if self.sgs_filter_width_mode == "fe_cell_volume" and not reference_z:
            raise ValueError(
                "fe_cell_volume requires native reference heights in sgs_reference_z."
            )

        if reference_z:
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
