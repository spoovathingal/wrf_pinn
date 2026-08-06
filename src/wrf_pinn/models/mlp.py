"""Fully connected MLP for the first Cartesian WRF PINN."""

from __future__ import annotations

import torch
from torch import nn

from wrf_pinn.config.model import DEFAULT_MODEL, ModelConfig


class MLP(nn.Module):
    """Coordinate-to-state multilayer perceptron.

    The model maps continuous coordinates ``(x, y, z, t)`` to the atmospheric
    state ``(u, v, w, theta, p_prime)``. Inputs and targets are expected to be
    normalized before they reach this model.
    """

    def __init__(
        self,
        config: ModelConfig = DEFAULT_MODEL,
    ) -> None:
        super().__init__()
        self.config = config

        self.network = self._build_network(config)
        self._initialize(config)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate state predictions at coordinate inputs."""

        if coordinates.ndim != 2 or coordinates.shape[1] != self.config.input_dim:
            msg = (
                "coordinates must have shape "
                f"(n_points, {self.config.input_dim}); got {tuple(coordinates.shape)}."
            )
            raise ValueError(msg)

        return self.network(coordinates)

    @staticmethod
    def _build_network(config: ModelConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = config.input_dim

        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(in_features, config.hidden_width))
            layers.append(_activation(config.activation))
            in_features = config.hidden_width

        layers.append(nn.Linear(in_features, config.output_dim))
        return nn.Sequential(*layers)

    def _initialize(self, config: ModelConfig) -> None:
        for module in self.network:
            if not isinstance(module, nn.Linear):
                continue

            if config.initializer == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif config.initializer == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif config.initializer == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unsupported initializer: {config.initializer}.")

            nn.init.zeros_(module.bias)


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()

    raise ValueError(f"Unsupported activation: {name}.")
