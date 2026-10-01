"""Conditional encoder--decoder model for the conditional PINN (MWE).

The model is conditioned on a case's fields and predicts the interior state at
query coordinates:

    z        = encode(initial, boundary, terrain)        # one latent per case
    q_hat    = decoder([coords | z])                      # per query point

Losses are taken on ``q_hat`` (the decoded output), never on ``z`` -- that is the
rule that forces a decoder into the network.

Two encoder choices, both deliberately simple (YAGNI for the MWE):
  NullEncoder     returns a zero-width latent, so the decoder reduces to a plain
                  coordinate MLP (proves the pipeline runs unchanged).
  FlattenMLPEncoder   flattens each member, concatenates, one MLP -> latent
                  (proves conditioning is wired: changing phi/psi changes output).

The real architecture (the groupmate's encoder-decoder analysis) can replace the
encoder by satisfying the same ``Encoder`` protocol.
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
    """Flatten each member, concatenate, one MLP to a latent vector.

    Simplest thing that makes conditioning real. Not the final architecture.
    """

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


from wrf_pinn.config.physics import DEFAULT_PHYSICS


@dataclass(frozen=True)
class ConditionalModelConfig:
    coord_dim: int = 4
    # Full physics state (u,v,w,theta,p_prime,k_m) so the PDE and surface residuals
    # work unchanged. The 4 supervised vars (u,v,w,theta) are the first columns;
    # pressure and eddy-viscosity are unsupervised by data but constrained by physics.
    state_dim: int = DEFAULT_PHYSICS.state_dim
    hidden_width: int = 128
    hidden_layers: int = 4
    activation: str = "tanh"


class ConditionalModel(nn.Module):
    """encode(fields) -> z; decoder([coords | z]) -> state.

    The encoder is injected (NullEncoder or FlattenMLPEncoder, or a real one later).
    """

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
        """Predict state at ``coordinates`` (n_pts, coord_dim) given latent ``z``.

        ``z`` is (1, latent_dim); it is broadcast to every query point.
        """
        if z.shape[1] == 0:
            decoder_in = coordinates
        else:
            z_rows = z.expand(coordinates.shape[0], -1)
            decoder_in = torch.cat([coordinates, z_rows], dim=1)
        return self.decoder(decoder_in)
