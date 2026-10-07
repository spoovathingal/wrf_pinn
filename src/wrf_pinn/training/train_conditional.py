"""Train one shared ConditionalModel across a batch of cases on the five losses."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch
from torch import nn

from wrf_pinn.data.conditional_case import (
    ConditionalCase, COORD_NAMES, STATE_VARS,
)
from wrf_pinn.config.physics import DEFAULT_PHYSICS
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig
from wrf_pinn.physics.residuals_pde import cartesian_zero_forcing_residuals
from wrf_pinn.physics.residuals_boundary import no_slip_wall_residuals

#: the five losses of the conditional PINN, in order.
LOSS_NAMES: tuple[str, ...] = ("data", "initial", "flow_boundary", "pde", "surface")


@dataclass
class ConditionalHistory:
    """Per-epoch loss history (one list per loss name)."""

    total: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)
    initial: list[float] = field(default_factory=list)
    flow_boundary: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    surface: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ConditionalTrainConfig:
    epochs: int = 2000
    batch_cases: int = 4          # cases drawn per step
    learning_rate: float = 1e-3
    weight_data: float = 1.0
    weight_initial: float = 1.0
    weight_flow_boundary: float = 1.0
    weight_pde: float = 1.0
    weight_surface: float = 1.0
    n_collocation: int = 2048     # PDE collocation points per case per step
    n_wall: int = 512             # surface/wall points per case per step
    grad_clip_norm: float = 1.0   # clip grads (PINN PDE gradients can spike)
    #: affine scaling (physical = offset + scale*normalized) for the residuals,
    #: built from the pre-processor's normalization recipe. Identity if None.
    scaling: "ResidualScalingConfig | None" = None
    log_every: int = 100
    device: str = "auto"
    seed: int = 0


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _initial_targets(case: ConditionalCase) -> torch.Tensor:
    """Flatten the decoded initial field to (n_ic, n_state), matching initial_coords."""
    fields = [case.initial[v].transpose(0, 1).reshape(-1) for v in range(case.initial.shape[0])]
    return torch.stack(fields, dim=1)


def _boundary_targets(case: ConditionalCase) -> torch.Tensor:
    """Flatten the decoded boundary field to (n_points, n_state), matching boundary_coords."""
    ns = case.boundary.shape[0]
    return torch.stack([case.boundary[v].reshape(-1) for v in range(ns)], dim=1)


def _case_losses(model: nn.Module, case: ConditionalCase,
                 cfg: "ConditionalTrainConfig") -> dict[str, torch.Tensor]:
    """The five losses for one case; the model conditions on the case's latent z."""
    n_coord = len(COORD_NAMES)
    n_obs = len(STATE_VARS)
    z = case.z.unsqueeze(0)

    pred = model(case.interior, z)[:, :n_obs]
    err = (pred - case.targets) * case.target_mask
    data = err.square().sum() / case.target_mask.sum().clamp_min(1.0)

    initial = (model(case.initial_coords, z)[:, :n_obs] - _initial_targets(case)).square().mean()

    bc_coords = case.boundary_coords.reshape(-1, n_coord)
    flow = (model(bc_coords, z)[:, :n_obs] - _boundary_targets(case)).square().mean()

    scaling = cfg.scaling if cfg.scaling is not None else DEFAULT_RESIDUAL_SCALING
    zero = torch.zeros((), device=case.interior.device, dtype=case.interior.dtype)

    if cfg.weight_pde != 0.0:
        coll = _sample_box(case.interior, cfg.n_collocation).requires_grad_(True)
        coll_state = model(coll, z)
        pde_res = cartesian_zero_forcing_residuals(
            coll, coll_state, physics=DEFAULT_PHYSICS, scaling=scaling)
        pde = torch.stack([r.square().mean() for r in pde_res.values()]).mean()
    else:
        pde = zero

    if cfg.weight_surface != 0.0:
        wall = _sample_box(case.interior, cfg.n_wall)
        wall = wall.clone(); wall[:, 2] = case.interior[:, 2].min()   # z -> domain floor
        wall_state = model(wall, z)
        wall_res = no_slip_wall_residuals(wall_state, scaling=scaling)
        surface = torch.stack([r.square().mean() for r in wall_res.values()]).mean()
    else:
        surface = zero

    return {"data": data, "initial": initial, "flow_boundary": flow,
            "pde": pde, "surface": surface}


def _sample_box(interior: torch.Tensor, n: int) -> torch.Tensor:
    """Sample n random coordinates uniformly in the bounding box of the interior
    query points (same normalized coordinate ranges as the case)."""
    lo = interior.min(dim=0).values
    hi = interior.max(dim=0).values
    u = torch.rand((n, interior.shape[1]), device=interior.device,
                   dtype=interior.dtype)
    return lo + u * (hi - lo)


def train_conditional(
    model: nn.Module,
    cases: list[ConditionalCase],
    config: ConditionalTrainConfig = ConditionalTrainConfig(),
) -> ConditionalHistory:
    """Train the shared model across ``cases``; return the loss history."""
    if not cases:
        raise ValueError("train_conditional needs at least one case.")

    device = _resolve_device(config.device)
    model.to(device)
    cases = [c.as_torch(device=device) for c in cases]

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = torch.Generator().manual_seed(config.seed)
    history = ConditionalHistory()
    weights = {"data": config.weight_data, "initial": config.weight_initial,
               "flow_boundary": config.weight_flow_boundary,
               "pde": config.weight_pde, "surface": config.weight_surface}

    for epoch in range(1, config.epochs + 1):
        k = min(config.batch_cases, len(cases))
        idx = torch.randperm(len(cases), generator=rng)[:k].tolist()

        optimizer.zero_grad()
        batch = {name: torch.zeros((), device=device) for name in weights}
        for i in idx:
            for name, value in _case_losses(model, cases[i], config).items():
                batch[name] = batch[name] + value
        for name in batch:
            batch[name] = batch[name] / k
        total = sum(weights[name] * batch[name] for name in weights)

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
        total.backward()
        # Clip gradients: PDE residual gradients (2nd derivatives via autograd) can
        # spike and freeze training. Standard PINN safeguard.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        history.total.append(float(total.detach().cpu()))
        for name in weights:
            getattr(history, name).append(float(batch[name].detach().cpu()))

        if epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0:
            parts = " ".join(f"{n}={float(batch[n].detach()):.4e}" for n in weights)
            # stderr + flush so progress is live in the SLURM log, never buffered
            print(f"epoch {epoch}/{config.epochs} "
                  f"total={float(total.detach()):.4e} {parts}",
                  file=sys.stderr, flush=True)

    return history


def split_cases(cases: list[ConditionalCase], chi: float, seed: int = 0
                ) -> tuple[list[ConditionalCase], list[ConditionalCase]]:
    """Split a case family into a train fraction ``chi`` and a held-out remainder."""
    import numpy as np

    order = np.random.default_rng(seed).permutation(len(cases))
    n_train = max(1, round(len(cases) * chi))
    train_idx = set(order[:n_train].tolist())
    train = [c for i, c in enumerate(cases) if i in train_idx]
    held = [c for i, c in enumerate(cases) if i not in train_idx]
    return train, held
