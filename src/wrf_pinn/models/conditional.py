"""Solution network for the conditional PINN: [coords | z] -> state. z is the POD
coefficients of the initial/boundary, supplied per case; encoding is done upstream
in the pre-processor, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from wrf_pinn.config.physics import DEFAULT_PHYSICS


@dataclass(frozen=True)
class ConditionalModelConfig:
    coord_dim: int = 4
    # full physics state (6 vars); first 4 are supervised, rest physics-only
    state_dim: int = DEFAULT_PHYSICS.state_dim
    hidden_width: int = 128
    hidden_layers: int = 4
    activation: str = "tanh"


class ConditionalModel(nn.Module):
    """Solution network: [coords | z] -> state. z is the POD coeffs, supplied per
    case; the model does not encode."""

    def __init__(self, latent_dim: int,
                 config: ConditionalModelConfig = ConditionalModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.net = self._build(config.coord_dim + latent_dim, config)

    @staticmethod
    def _build(in_features: int, config: ConditionalModelConfig) -> nn.Sequential:
        act = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[
            config.activation]
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(config.hidden_layers):
            layers += [nn.Linear(width, config.hidden_width), act()]
            width = config.hidden_width
        layers.append(nn.Linear(width, config.state_dim))
        return nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict state at coordinates; z (1, latent_dim) broadcasts to every point."""
        if z.shape[1] == 0:
            return self.net(coordinates)
        z_rows = z.expand(coordinates.shape[0], -1)
        return self.net(torch.cat([coordinates, z_rows], dim=1))
