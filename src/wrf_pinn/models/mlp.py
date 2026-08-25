"""Fully connected MLP for the first Cartesian WRF PINN."""

from __future__ import annotations
import math
import torch
from torch import nn
from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.config.model import DEFAULT_MODEL, ModelConfig


class MLP(nn.Module):
    """Coordinate-to-state multilayer perceptron.

    The model maps continuous coordinates ``(x, y, z, t)`` to the atmospheric
    state ``(u, v, w, theta, p_prime, k_m_unconstrained)``. First 5 outputs should've been
    normalized before they reach this model.
    """

    def __init__(
        self,
        config: ModelConfig = DEFAULT_MODEL,
        physics: PhysicsConfig = DEFAULT_PHYSICS,
    ) -> None:
        super().__init__()
        if config.output_dim != physics.state_dim:
            raise ValueError(
                "Model output dimension must match the configured physics state; "
                f"got output_dim={config.output_dim} and "
                f"physics.state_dim={physics.state_dim}."
            )

        self.config = config
        self.physics = physics

        self.network = self._build_network(config)
        self._initialize(config)
        self._initialize_eddy_viscosity_output(physics)

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

    def _initialize_eddy_viscosity_output(self, physics: PhysicsConfig) -> None:
        """Initialize K_m as a constant field at its configured initial value."""

        output_layer = self.network[-1]
        if not isinstance(output_layer, nn.Linear):
            raise RuntimeError("The final MLP module must be a linear output layer.")

        k_m_index = physics.variable_index("k_m")

        k_m_min = physics.constants.eddy_viscosity_min
        k_m_max = physics.constants.eddy_viscosity_max
        k_m_initial = physics.constants.eddy_viscosity_initial

        initial_fraction = (
            (k_m_initial - k_m_min)
            / (k_m_max - k_m_min)
        )
        initial_unconstrained = math.log(
            initial_fraction / (1.0 - initial_fraction)
        )

        with torch.no_grad():
            output_layer.weight[k_m_index].zero_()
            output_layer.bias[k_m_index].fill_(initial_unconstrained)

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
