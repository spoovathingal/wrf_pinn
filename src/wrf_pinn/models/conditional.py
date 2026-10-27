"""Solution network for the conditional PINN: [coords | z] -> state. z is the POD
coefficients of the initial/boundary (encoded upstream in the pre-processor), or the
raw fields, optionally passed through a learned RawEncoder first.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from wrf_pinn.config.physics import DEFAULT_PHYSICS


@dataclass(frozen=True)
class ConditionalModelConfig:
    coord_dim: int = 4
    state_dim: int = DEFAULT_PHYSICS.state_dim
    hidden_width: int = 128
    hidden_layers: int = 4
    activation: str = "tanh"


class RawEncoder(nn.Module):
    """Raw z (bnd_raw then ini_raw, flattened per case) -> latent. 3-D convs over the
    initial field (n_state, nz, nx, ny); 2-D convs over each side face, with times x
    state as channels (n_times*n_state, nz, face width); one linear head on both."""

    def __init__(self, ini_shape: tuple[int, int, int, int],
                 bnd_shape: tuple[int, int, int, int], out_dim: int = 256,
                 width: int = 32) -> None:
        super().__init__()
        self.ini_shape, self.bnd_shape, self.out_dim = tuple(ini_shape), tuple(bnd_shape), out_dim
        nv, nz, nx, ny = ini_shape
        nf, nt, fl, _ = bnd_shape
        self.n_bnd = nf * nt * fl * nv
        conv = lambda C, cin: nn.Sequential(
            C(cin, width, 3, 2, 1), nn.GELU(), C(width, 2 * width, 3, 2, 1), nn.GELU(),
            C(2 * width, 4 * width, 3, 2, 1), nn.GELU(), nn.Flatten())
        self.ini = conv(nn.Conv3d, nv)
        self.bnd = conv(nn.Conv2d, nt * nv)
        with torch.no_grad():
            n_ini = self.ini(torch.zeros(1, nv, nz, nx, ny)).shape[1]
            n_face = self.bnd(torch.zeros(1, nt * nv, nz, fl // nz)).shape[1]
        self.head = nn.Linear(n_ini + nf * n_face, out_dim)

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (k, n) -> initial (k, n_state, nz, nx, ny) and faces (k*n_faces,
        n_times*n_state, nz, face width); face rows are z-slow in bnd_raw."""
        k = z.shape[0]
        nf, nt, fl, nv = self.bnd_shape
        nz = self.ini_shape[1]
        bnd = (z[:, :self.n_bnd].reshape(k, nf, nt, nz, fl // nz, nv)
               .permute(0, 1, 2, 5, 3, 4).reshape(k * nf, nt * nv, nz, fl // nz))
        return z[:, self.n_bnd:].reshape(k, *self.ini_shape), bnd

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ini, bnd = self.split(z)
        return self.head(torch.cat([self.ini(ini), self.bnd(bnd).reshape(z.shape[0], -1)], dim=1))


class RawProjection(nn.Linear):
    """Raw z (flattened, as RawEncoder takes it) -> latent through one linear layer."""

    def __init__(self, in_dim: int, out_dim: int = 256) -> None:
        super().__init__(in_dim, out_dim)
        self.out_dim = out_dim


class ConditionalModel(nn.Module):
    """Solution network: [coords | z] -> state. z is supplied per case. With an
    encoder, callers pass encode(z) (the training step does this once per pass).
    The encoder output is LayerNormed so a wide raw input cannot saturate the first
    tanh layer (unnormalized, Adam grew it to ~100-1000 and the output went constant)."""

    def __init__(self, latent_dim: int,
                 config: ConditionalModelConfig = ConditionalModelConfig(),
                 encoder: "RawEncoder | RawProjection | None" = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        width = encoder.out_dim if encoder is not None else latent_dim
        self.latent_norm = nn.LayerNorm(width) if encoder is not None else None
        self.net = self._build(config.coord_dim + width, config)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """The case latent the net sees: LayerNorm(encoder(z)), or z itself."""
        return z if self.encoder is None else self.latent_norm(self.encoder(z))

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

    def forward(self, coordinates: torch.Tensor, z: torch.Tensor,
                sizes: "torch.Tensor | int | None" = None) -> torch.Tensor:
        """Predict state at coordinates.

        z is (1, latent_dim), broadcast to every point; (n_points, latent_dim), one
        row per point; or, with ``sizes``, (k, latent_dim), one row per case, where
        case i owns the next ``sizes[i]`` rows of ``coordinates`` (an int: the same
        count for every case). The per-case forms multiply z through the first layer
        once per case instead of once per point: same math, without building an
        (n_points, latent_dim) copy of z."""
        if z.shape[1] == 0:
            return self.net(coordinates)
        if sizes is None and z.shape[0] == coordinates.shape[0]:
            return self.net(torch.cat([coordinates, z], dim=1))
        return self.net[1:](first_layer(self.net[0], coordinates, z, sizes))


def first_layer(layer: nn.Linear, coordinates: torch.Tensor, z: torch.Tensor,
                sizes: "torch.Tensor | int | None") -> torch.Tensor:
    """``layer([coordinates | z per point])`` with z given per case (or broadcast).

    W [c; z] + b = W_c c + (W_z z + b): the z term is computed for the k cases and
    repeated to their rows, so its cost no longer scales with the number of points."""
    n_coord = coordinates.shape[1]
    per_case = nn.functional.linear(z, layer.weight[:, n_coord:], layer.bias)   # (k, width)
    if sizes is None:
        rows = per_case.expand(coordinates.shape[0], -1)
    else:
        rows = per_case.repeat_interleave(sizes, dim=0, output_size=coordinates.shape[0])
    return nn.functional.linear(coordinates, layer.weight[:, :n_coord]) + rows
