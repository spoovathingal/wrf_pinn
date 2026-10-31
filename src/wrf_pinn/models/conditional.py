"""Solution network for the conditional PINN: [coords | z] -> state. z is the POD
coefficients of the initial/boundary, supplied per case; encoding is done upstream
in the pre-processor, not here.
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

@dataclass(frozen=True)
class ConditionalModelConfig:
    coord_dim: int = 4
    state_dim: int = DEFAULT_PHYSICS.state_dim
    hidden_width: int = 128
    hidden_layers: int = 4
    activation: str = "tanh"

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

class ConditionalModel(nn.Module):
    """Solution network: [coords | z] -> state. z is the POD coeffs, supplied per
    case; the model does not encode."""

    def __init__(self, latent_dim: int,
                 config: ConditionalModelConfig = ConditionalModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.net = self._build(config.coord_dim + latent_dim, config)

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
        """Predict normalized state with one latent row per point or per case.

        z is (1, latent_dim), broadcast to every point; (n_points, latent_dim), one
        row per point; or, with ``sizes``, (k, latent_dim), one row per case, where
        case i owns the next ``sizes[i]`` rows of ``coordinates`` (an int: the same
        count for every case). The per-case forms multiply z through the first layer
        once per case instead of once per point: same math, without building an
        (n_points, latent_dim) copy of z."""
        if z.shape[1] == 0:
            raw_state = self.net(coordinates)
        elif sizes is None and z.shape[0] == coordinates.shape[0]:
            raw_state = self.net(torch.cat([coordinates, z], dim=1))
        else:
            raw_state = self.net[1:](
                first_layer(self.net[0], coordinates, z, sizes)
            )
        return transform_state_output(raw_state, self.config)


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
