"""Model architecture configuration for WRF PINN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wrf_pinn.config.physics import DEFAULT_PHYSICS


ActivationName = Literal["tanh", "silu", "gelu", "relu"]
InitializerName = Literal["xavier_uniform", "xavier_normal", "kaiming_uniform"]
ModelFamily = Literal["mlp"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the first Cartesian WRF PINN neural network.

    The default is intentionally conservative: a standard fully connected MLP
    that maps continuous coordinates (x, y, z, t) to the reduced physics state
    (u, v, w, rho).
    """

    family: ModelFamily = "mlp"
    input_dim: int = 4
    output_dim: int = DEFAULT_PHYSICS.state_dim
    hidden_width: int = 128
    hidden_layers: int = 6
    activation: ActivationName = "tanh"
    initializer: InitializerName = "xavier_uniform"
    use_input_normalization: bool = True
    use_output_scaling: bool = True
    residual_connection: bool = False

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be positive; got {self.input_dim}.")

        if self.output_dim < 1:
            raise ValueError(f"output_dim must be positive; got {self.output_dim}.")

        if self.hidden_width < 1:
            raise ValueError(
                f"hidden_width must be positive; got {self.hidden_width}."
            )

        if self.hidden_layers < 1:
            raise ValueError(
                f"hidden_layers must be positive; got {self.hidden_layers}."
            )


DEFAULT_MODEL = ModelConfig()
