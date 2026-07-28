"""Scaling configuration for WRF PINN inputs, outputs, and residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wrf_pinn.config.physics import DEFAULT_PHYSICS


ScaleMode = Literal["none", "standard", "minmax", "reference"]


@dataclass(frozen=True)
class VariableScale:
    """Affine scale used to nondimensionalize one variable."""

    name: str
    reference: float = 1.0
    offset: float = 0.0
    mode: ScaleMode = "reference"

    def __post_init__(self) -> None:
        if self.reference <= 0.0:
            raise ValueError(
                f"{self.name} reference scale must be positive; got {self.reference}."
            )


@dataclass(frozen=True)
class ScalingConfig:
    """Scaling choices shared by models, data loaders, and residuals.

    The values are intentionally first-pass reference scales. They can be
    replaced later by dataset-derived statistics once WRF/HRRR readers exist.
    """

    coordinate_scales: tuple[VariableScale, ...] = (
        VariableScale("x", reference=1000.0),
        VariableScale("y", reference=1000.0),
        VariableScale("z", reference=1000.0),
        VariableScale("t", reference=3600.0),
    )
    state_scales: tuple[VariableScale, ...] = (
        VariableScale("u", reference=10.0),
        VariableScale("v", reference=10.0),
        VariableScale("w", reference=1.0),
        VariableScale("rho", reference=1.0),
    )
    residual_scales: tuple[VariableScale, ...] = (
        VariableScale("mass", reference=1.0),
        VariableScale("x_momentum", reference=1.0),
        VariableScale("y_momentum", reference=1.0),
        VariableScale("z_momentum", reference=1.0),
    )
    use_coordinate_scaling: bool = True
    use_state_scaling: bool = True
    use_residual_scaling: bool = False

    def __post_init__(self) -> None:
        state_names = tuple(scale.name for scale in self.state_scales)
        residual_names = tuple(scale.name for scale in self.residual_scales)

        if state_names != DEFAULT_PHYSICS.active_variables:
            msg = (
                "state_scales must match DEFAULT_PHYSICS.active_variables; "
                f"got {state_names}."
            )
            raise ValueError(msg)

        if residual_names != DEFAULT_PHYSICS.residuals:
            msg = (
                "residual_scales must match DEFAULT_PHYSICS.residuals; "
                f"got {residual_names}."
            )
            raise ValueError(msg)

    def coordinate_scale(self, name: str) -> VariableScale:
        """Return scaling config for one coordinate."""

        return self._get_scale(self.coordinate_scales, name)

    def state_scale(self, name: str) -> VariableScale:
        """Return scaling config for one state variable."""

        return self._get_scale(self.state_scales, name)

    def residual_scale(self, name: str) -> VariableScale:
        """Return scaling config for one residual."""

        return self._get_scale(self.residual_scales, name)

    @staticmethod
    def _get_scale(scales: tuple[VariableScale, ...], name: str) -> VariableScale:
        for scale in scales:
            if scale.name == name:
                return scale

        raise KeyError(f"Unknown scale name: {name}.")


DEFAULT_SCALING = ScalingConfig()
