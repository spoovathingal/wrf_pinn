"""Solution network for the conditional PINN: [coords | z] -> state. z is the POD
coefficients of the initial/boundary (encoded upstream in the pre-processor), or the
raw fields, optionally passed through a learned RawEncoder first.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from wrf_pinn.config.physics import DEFAULT_PHYSICS

TKE_INDEX = DEFAULT_PHYSICS.variable_index("e_sgs")
RHO_D_INDEX = DEFAULT_PHYSICS.variable_index("rho_d")

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class AdaptiveTanh(nn.Module):
    """Jagtap, Kawaguchi and Karniadakis 2020: tanh(n a x) with one learned slope a per layer."""

    def __init__(self, n: float = 10.0):
        super().__init__()
        self.n = n
        self.a = nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, x):
        return torch.tanh(self.n * self.a * x)


ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU,
               "sin": Sine, "adaptive-tanh": AdaptiveTanh}


@dataclass(frozen=True)
class ConditionalModelConfig:
    coord_dim: int = 4
    state_dim: int = DEFAULT_PHYSICS.state_dim
    hidden_width: int = 128
    hidden_layers: int = 4
    activation: str = "tanh"
    initializer: str = "default"    # default (PyTorch) or xavier (uniform weights, zero bias)

    # Dataset normalization: e_physical = tke_offset + tke_scale * e_normalized.
    tke_offset: float = 0.0
    tke_scale: float = 1.0
    # Physical units: m^2/s^2.
    tke_floor: float = 1.0e-10
    tke_softplus_scale: float = 1.0
    tke_initial_value: float = 0.01
    # Dataset normalization:
    # rho_physical = rho_d_offset + rho_d_scale * rho_normalized.
    rho_d_offset: float = 0.0
    rho_d_scale: float = 1.0

    # Physical units: kg/m^3.
    rho_d_floor: float = 1.0e-6
    rho_d_softplus_scale: float = 1.0
    rho_d_initial_value: float = 1.0

    def __post_init__(self) -> None:
        if self.state_dim != DEFAULT_PHYSICS.state_dim:
            raise ValueError(
                f"state_dim must be {DEFAULT_PHYSICS.state_dim}, "
                "including the rho_d output."
            )

        for prefix in ("tke", "rho_d"):
            for suffix in ("offset", "scale", "floor","softplus_scale", "initial_value"):
                name = f"{prefix}_{suffix}"
                if not math.isfinite(getattr(self, name)):
                    raise ValueError(f"{name} must be finite.")

            scale = getattr(self, f"{prefix}_scale")
            floor = getattr(self, f"{prefix}_floor")
            softplus_scale = getattr(self, f"{prefix}_softplus_scale")
            initial_value = getattr(self, f"{prefix}_initial_value")

            if scale <= 0.0:
                raise ValueError(f"{prefix}_scale must be positive.")
            if softplus_scale <= 0.0:
                raise ValueError(f"{prefix}_softplus_scale must be positive.")
            if floor < 0.0:
                raise ValueError(f"{prefix}_floor must be nonnegative.")
            if initial_value <= floor:
                raise ValueError(f"{prefix}_initial_value must exceed {prefix}_floor.")

            initial_argument = (initial_value - floor) / softplus_scale
            if not math.isfinite(initial_argument) or initial_argument <= 0.0:
                raise ValueError(f"Initial {prefix}/Softplus scale ratio is invalid.")

        if self.rho_d_floor <= 0.0:
            raise ValueError("rho_d_floor must be strictly positive.")

def transform_state_output(raw_state: torch.Tensor,config: ConditionalModelConfig,
) -> torch.Tensor:
    """Convert raw outputs to normalized state, enforcing physical TKE positivity."""
    # Keep the affine conversion and Softplus at least float32, including when
    # the derivative-free network forwards use bf16 autocast.
    if raw_state.ndim != 2 or raw_state.shape[1] != config.state_dim:
        raise ValueError(f"Expected raw_state shape (n, {config.state_dim}); "
            f"got {tuple(raw_state.shape)}.")

    if raw_state.dtype in (torch.float16, torch.bfloat16):
        raw_state = raw_state.float()

    state = raw_state
    for index, prefix in ((TKE_INDEX, "tke"), (RHO_D_INDEX, "rho_d")):
        raw = raw_state[:, index:index + 1]
        physical = (getattr(config, f"{prefix}_floor")
            + getattr(config, f"{prefix}_softplus_scale")
            * F.softplus(raw, beta=1.0, threshold=20.0)
        )
        normalized = (
            physical - getattr(config, f"{prefix}_offset")
        ) / getattr(config, f"{prefix}_scale")

        state = torch.cat((state[:, :index], normalized, state[:, index + 1:]),
            dim=1)

    return state


def inverse_transform_state(state: torch.Tensor, config: ConditionalModelConfig) -> torch.Tensor:
    """Raw outputs that ``transform_state_output`` maps to ``state`` (n, state_dim).
    TKE and rho_d at or below their floor are clamped just above it."""
    raw_state = state
    for index, prefix in ((TKE_INDEX, "tke"), (RHO_D_INDEX, "rho_d")):
        physical = (getattr(config, f"{prefix}_offset")
                    + getattr(config, f"{prefix}_scale") * state[:, index:index + 1])
        y = ((physical - getattr(config, f"{prefix}_floor"))
             / getattr(config, f"{prefix}_softplus_scale")).clamp_min(1e-12)
        raw = torch.where(y > 20.0, y, torch.log(torch.expm1(torch.minimum(y, torch.full_like(y, 20.0)))))
        raw_state = torch.cat((raw_state[:, :index], raw, raw_state[:, index + 1:]), dim=1)
    return raw_state

class RawEncoder(nn.Module):
    """Raw z (bnd_raw then ini_raw, flattened per case) -> latent. 3-D convs over the
    initial field (n_state, nz, nx, ny); 2-D convs over each side face, with times x
    state as channels (n_times*n_state, nz, face width); one linear head on both.
    Inputs are first standardized per state variable and level (``in_mean``, ``in_std``,
    set by ``set_input_stats``; 0 and 1 until then, and for checkpoints saved before them).
    terrain=True: z ends with the ground height (m) on the grid (nx*ny, x slow), read by
    2-D convs as two channels: the shape (height minus the subdomain mean, over
    ``ter_stats[2]``) and the subdomain mean ((mean - ter_stats[0]) / ter_stats[1])."""

    def __init__(self, ini_shape: tuple[int, int, int, int],
                 bnd_shape: tuple[int, int, int, int], out_dim: int = 256,
                 width: int = 32, terrain: bool = False) -> None:
        super().__init__()
        self.ini_shape, self.bnd_shape, self.out_dim = tuple(ini_shape), tuple(bnd_shape), out_dim
        self.terrain = terrain
        nv, nz, nx, ny = ini_shape
        self.register_buffer("in_mean", torch.zeros(nv, nz))
        self.register_buffer("in_std", torch.ones(nv, nz))
        nf, nt, fl, _ = bnd_shape
        self.n_bnd = nf * nt * fl * nv
        self.n_ini = nv * nz * nx * ny
        conv = lambda C, cin: nn.Sequential(
            C(cin, width, 3, 2, 1), nn.GELU(), C(width, 2 * width, 3, 2, 1), nn.GELU(),
            C(2 * width, 4 * width, 3, 2, 1), nn.GELU(), nn.Flatten())
        self.ini = conv(nn.Conv3d, nv)
        self.bnd = conv(nn.Conv2d, nt * nv)
        with torch.no_grad():
            n_ini = self.ini(torch.zeros(1, nv, nz, nx, ny)).shape[1]
            n_face = self.bnd(torch.zeros(1, nt * nv, nz, fl // nz)).shape[1]
        n_ter = 0
        if terrain:
            self.register_buffer("ter_stats", torch.tensor([0.0, 1.0, 1.0]))
            self.ter = conv(nn.Conv2d, 2)
            with torch.no_grad():
                n_ter = self.ter(torch.zeros(1, 2, nx, ny)).shape[1]
        self.head = nn.Linear(n_ini + nf * n_face + n_ter, out_dim)

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (k, n) -> initial (k, n_state, nz, nx, ny) and faces (k*n_faces,
        n_times*n_state, nz, face width); face rows are z-slow in bnd_raw."""
        k = z.shape[0]
        nf, nt, fl, nv = self.bnd_shape
        nz = self.ini_shape[1]
        m, s = self.in_mean[..., None], self.in_std[..., None]                    # (nv, nz, 1)
        bnd = ((z[:, :self.n_bnd].reshape(k, nf, nt, nz, fl // nz, nv).permute(0, 1, 2, 5, 3, 4) - m) / s
               ).reshape(k * nf, nt * nv, nz, fl // nz)
        ini = (z[:, self.n_bnd:self.n_bnd + self.n_ini].reshape(k, *self.ini_shape) - m[..., None]) / s[..., None]
        return ini, bnd

    def split_terrain(self, z: torch.Tensor) -> torch.Tensor:
        """z (k, n) -> terrain channels (k, 2, nx, ny): shape and subdomain mean, standardized."""
        h = z[:, self.n_bnd + self.n_ini:].reshape(z.shape[0], *self.ini_shape[2:])
        mean = h.mean(dim=(1, 2), keepdim=True)
        level = ((mean - self.ter_stats[0]) / self.ter_stats[1]).expand_as(h)
        return torch.stack([(h - mean) / self.ter_stats[2], level], dim=1)

    def set_terrain_stats(self, mean: float, std: float, shape_std: float) -> None:
        """Mean and spread of subdomain-mean height, and spread of height about it (m)."""
        self.ter_stats.copy_(torch.tensor([mean, max(std, 1e-6), max(shape_std, 1e-6)]))

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Per (state variable, level) mean and spread of the raw fields, (n_state, nz)."""
        self.in_mean.copy_(mean)
        self.in_std.copy_(std.clamp_min(1e-6))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name, fill in (("in_mean", torch.zeros_like), ("in_std", torch.ones_like)):
            state_dict.setdefault(prefix + name, fill(getattr(self, name)))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ini, bnd = self.split(z)
        parts = [self.ini(ini), self.bnd(bnd).reshape(z.shape[0], -1)]
        if self.terrain:
            parts.append(self.ter(self.split_terrain(z)))
        return self.head(torch.cat(parts, dim=1))


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

        # Initialize TKE and density near their physical starting values.
        # Random output weights retain spatial and case-dependent variation.
        with torch.no_grad():
            for index, prefix in ((TKE_INDEX, "tke"), (RHO_D_INDEX, "rho_d")):
                
                target = (
                    getattr(config, f"{prefix}_initial_value")
                    - getattr(config, f"{prefix}_floor")
                ) / getattr(config, f"{prefix}_softplus_scale")

                bias = (
                    target if target > 20.0
                    else target + math.log(-math.expm1(-target))
                )
                self.net[-1].bias[index] = bias

    @staticmethod
    def _build(in_features: int, config: ConditionalModelConfig) -> nn.Sequential:
        act = ACTIVATIONS[config.activation]
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(config.hidden_layers):
            layers += [nn.Linear(width, config.hidden_width), act()]
            width = config.hidden_width
        layers.append(nn.Linear(width, config.state_dim))
        if config.initializer == "xavier":
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        elif config.initializer != "default":
            raise ValueError(f"Unknown initializer: {config.initializer}.")
        return nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor, z: torch.Tensor,
                sizes: "torch.Tensor | int | None" = None, raw: bool = False) -> torch.Tensor:
        """Predict normalized state with one latent row per point or per case.

        z is (1, latent_dim), broadcast to every point; (n_points, latent_dim), one
        row per point; or, with ``sizes``, (k, latent_dim), one row per case, where
        case i owns the next ``sizes[i]`` rows of ``coordinates`` (an int: the same
        count for every case). The per-case forms multiply z through the first layer
        once per case instead of once per point: same math, without building an
        (n_points, latent_dim) copy of z. ``raw``: skip the output transform."""
        if z.shape[1] == 0:
            raw_state = self.net(coordinates)
        elif sizes is None and z.shape[0] == coordinates.shape[0]:
            raw_state = self.net(torch.cat([coordinates, z], dim=1))
        else:
            raw_state = self.net[1:](
                first_layer(self.net[0], coordinates, z, sizes)
            )
        return raw_state if raw else transform_state_output(raw_state, self.config)


def first_layer(layer: nn.Linear, coordinates: torch.Tensor, z: torch.Tensor,
                sizes: "torch.Tensor | int | None") -> torch.Tensor:
    """``layer([coordinates | z per point])`` with z given per case (or broadcast).

    W [c; z] + b = W_c c + (W_z z + b): the z term is computed for the k cases and
    repeated to their rows, so its cost no longer scales with the number of points."""
    n_coord = coordinates.shape[1]
    per_case = F.linear(z, layer.weight[:, n_coord:], layer.bias)   # (k, width)
    if sizes is None:
        rows = per_case.expand(coordinates.shape[0], -1)
    else:
        rows = per_case.repeat_interleave(sizes, dim=0, output_size=coordinates.shape[0])
    return F.linear(coordinates, layer.weight[:, :n_coord]) + rows
