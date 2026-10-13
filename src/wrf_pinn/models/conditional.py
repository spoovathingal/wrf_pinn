"""Solution network for the conditional PINN: [coords | z] -> state. z is the POD
coefficients of the initial/boundary, supplied per case; encoding is done upstream
in the pre-processor, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from wrf_pinn.config.physics import DEFAULT_PHYSICS


@dataclass(frozen=True)
class ConditionalModelConfig:
    coord_dim: int = 4
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

    def project_condition(self, z: torch.Tensor) -> torch.Tensor:
        """Project each case's POD vector into the first layer's output space."""
        first = self.net[0]
        latent_dim = first.in_features - self.config.coord_dim

        if z.ndim != 2 or z.shape[1] != latent_dim:
            raise ValueError(f"z must have shape (n, {latent_dim}).")

        return F.linear(z, first.weight[:, self.config.coord_dim:], bias=None)

    def forward_projected(self, coordinates: torch.Tensor, condition: torch.Tensor,
        *,
        case_sizes: torch.Tensor | None = None) -> torch.Tensor:
        """Predict using projected conditioning, optionally grouped by case."""
        first = self.net[0]

        if (coordinates.ndim != 2 or coordinates.shape[1] != self.config.coord_dim):
            raise ValueError("Invalid coordinate shape.")

        if (condition.ndim != 2 or condition.shape[1] != first.out_features):
            raise ValueError("Invalid projected conditioning shape.")

        if case_sizes is not None:
            if (case_sizes.ndim != 1 or case_sizes.shape[0] != condition.shape[0]):
                raise ValueError("case_sizes must contain one count per case.")

            # Repeat only the projected values: typically tilesize, not 15,235.
            condition = torch.repeat_interleave(condition, case_sizes,
                dim=0, output_size=coordinates.shape[0])
        elif condition.shape[0] not in (1, coordinates.shape[0]):
            raise ValueError("Supply one condition row, one per point, or case_sizes.")

        # Include the original first-layer bias exactly once.
        hidden = F.linear(coordinates, first.weight[:, :self.config.coord_dim], first.bias,
        ) + condition

        # Includes the original first activation and all remaining layers.
        for layer in self.net[1:]:
            hidden = layer(hidden)

        return hidden

    def forward(self, coordinates: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Existing API: one POD vector or one vector per coordinate row."""
        return self.forward_projected(coordinates, self.project_condition(z))