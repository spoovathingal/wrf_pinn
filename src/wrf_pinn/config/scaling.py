"""Scaling metadata for physics residual evaluation.

The data pipeline is expected to normalize coordinates and state variables
before they enter the model. This module records the affine maps between the
normalized quantities used by the neural network and the physical quantities
needed by PDE residuals.

Convention:

    physical_value = offset + scale * normalized_value

The identity defaults preserve current behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CoordinateName = Literal["x", "y", "z", "t"]
StateVariableName = Literal["u", "v", "w", "theta", "p_prime"]
ScaledVariableName = CoordinateName | StateVariableName


@dataclass(frozen=True)
class VariableScale:
    """Affine scale for one normalized variable."""

    offset: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale == 0.0:
            raise ValueError("VariableScale.scale must be nonzero.")


@dataclass(frozen=True)
class ResidualScalingConfig:
    """Coordinate and state scaling used by PDE residuals."""

    x: VariableScale = VariableScale()
    y: VariableScale = VariableScale()
    z: VariableScale = VariableScale()
    t: VariableScale = VariableScale()
    u: VariableScale = VariableScale()
    v: VariableScale = VariableScale()
    w: VariableScale = VariableScale()
    theta: VariableScale = VariableScale()
    p_prime: VariableScale = VariableScale()

    def coordinate_scales(self) -> tuple[float, float, float, float]:
        """Return coordinate scale factors in x, y, z, t order."""

        return (self.x.scale, self.y.scale, self.z.scale, self.t.scale)

    def state_scales(self) -> tuple[float, float, float, float, float]:
        """Return state scale factors in u, v, w, theta, p' order."""

        return (self.u.scale, self.v.scale, self.w.scale,
                self.theta.scale, self.p_prime.scale)

    def scale_for(self, name: ScaledVariableName) -> VariableScale:
        """Return the scale object for a named coordinate or state variable."""

        return getattr(self, name)


DEFAULT_RESIDUAL_SCALING = ResidualScalingConfig()
