"""Conditional encoder--decoder model: z = encode(fields), q = decode([coords | z]).

Losses are taken on the decoded output, never on z. Encoders (Null, FlattenMLP,
Dual) share the Encoder protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class Encoder(Protocol):
    """Maps a case's conditioning members to one latent vector per case."""

    latent_dim: int

    def __call__(self, initial: torch.Tensor, boundary: torch.Tensor,
                 terrain: torch.Tensor) -> torch.Tensor:
        ...


class NullEncoder(nn.Module):
    """Returns a zero-width latent: the decoder reduces to a coordinate MLP."""

    latent_dim = 0

    def forward(self, initial, boundary, terrain) -> torch.Tensor:
        # a (1, 0) latent broadcasts to every query with no added features
        return initial.new_zeros((1, 0))


class FlattenMLPEncoder(nn.Module):
    """Flatten all members, concatenate, one MLP to a latent vector."""

    def __init__(self, *, in_features: int, latent_dim: int = 16,
                 hidden: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, initial, boundary, terrain) -> torch.Tensor:
        parts = [initial.reshape(-1), boundary.reshape(-1), terrain.reshape(-1)]
        return self.net(torch.cat(parts)).unsqueeze(0)   # (1, latent_dim)


class _MLP(nn.Module):
    """Flatten inputs, one hidden-layer MLP to a latent vector."""

    def __init__(self, in_features: int, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, *parts: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([p.reshape(-1) for p in parts])
        return self.net(flat)


class DualEncoder(nn.Module):
    """Separate initial and boundary encoders, z = [z_I | z_B].

    Matches the plan's E_I[q_0], E_B[B]. Terrain is static domain info, so it
    rides with the initial encoder rather than a third encoder.
    """

    def __init__(self, *, initial_features: int, boundary_features: int,
                 latent_dim: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        half = latent_dim // 2
        self.initial = _MLP(initial_features, latent_dim - half, hidden)
        self.boundary = _MLP(boundary_features, half, hidden)

    def forward(self, initial, boundary, terrain) -> torch.Tensor:
        z_i = self.initial(initial, terrain)
        z_b = self.boundary(boundary)
        return torch.cat([z_i, z_b]).unsqueeze(0)        # (1, latent_dim)


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
    """encode(fields) -> z; decoder([coords | z]) -> state. Encoder is injected."""

    def __init__(self, encoder: nn.Module,
                 config: ConditionalModelConfig = ConditionalModelConfig()) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        decoder_in = config.coord_dim + encoder.latent_dim
        self.decoder = self._build_decoder(decoder_in, config)

    @staticmethod
    def _build_decoder(in_features: int, config: ConditionalModelConfig) -> nn.Sequential:
        act = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[
            config.activation]
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(config.hidden_layers):
            layers += [nn.Linear(width, config.hidden_width), act()]
            width = config.hidden_width
        layers.append(nn.Linear(width, config.state_dim))
        return nn.Sequential(*layers)

    def encode(self, initial, boundary, terrain) -> torch.Tensor:
        return self.encoder(initial, boundary, terrain)

    def forward(self, coordinates: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict state at coordinates; z (1, latent_dim) broadcasts to every point."""
        if z.shape[1] == 0:
            decoder_in = coordinates
        else:
            z_rows = z.expand(coordinates.shape[0], -1)
            decoder_in = torch.cat([coordinates, z_rows], dim=1)
        return self.decoder(decoder_in)
