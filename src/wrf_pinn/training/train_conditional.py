"""Train one shared ConditionalModel across a batch of cases on the five losses."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from wrf_pinn.data.conditional_case import (
    ConditionalCase, COORD_NAMES, STATE_VARS,
)
from wrf_pinn.config.physics import DEFAULT_PHYSICS
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig
from wrf_pinn.physics.residuals_pde import cartesian_zero_forcing_residuals
from wrf_pinn.physics.residuals_boundary import no_penetration_z_wall_residuals

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
    #: flux (fricVel, htFlux) offset/scale from the recipe, to un-normalize
    #: case.surface back to physical for the surface-layer residual.
    flux_offset: tuple[float, float] = (0.0, 0.0)
    flux_scale: tuple[float, float] = (1.0, 1.0)
    surface_stress_scale: float = 3.8273
    surface_heat_flux_scale: float = 1.0
    log_every: int = 100
    device: str = "auto"
    seed: int = 0
    #: write a resumable checkpoint here every ``checkpoint_every`` epochs; resume
    #: from it on start if it exists. None disables checkpointing.
    checkpoint_path: "str | None" = None
    checkpoint_every: int = 100
    #: when True, time forward/backward/step per epoch (CUDA-synced) for profiling.
    profile: bool = False


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


def _surface_loss(model, case, z, cfg, scaling) -> torch.Tensor:
    """Surface-layer residual on the case's bottom face, using its stored fluxes.

    Wall coords come from case.surface (x,y,t normalized; z at the floor); fluxes
    are un-normalized to physical and fed to the no-penetration-z residual, which
    builds its own z1 reference pair."""
    surf = case.surface.reshape(-1, case.surface.shape[-1])          # (N, 5)
    dev, dt = surf.device, surf.dtype
    z_floor = (0.0 - scaling.z.offset) / scaling.z.scale
    wall = torch.stack([surf[:, 0], surf[:, 1],
                        torch.full((surf.shape[0],), z_floor, device=dev, dtype=dt),
                        surf[:, 2]], dim=1).requires_grad_(True)
    wall_state = model(wall, z)

    reference = wall.detach().clone()
    z1 = (DEFAULT_PHYSICS.constants.surface_reference_height - scaling.z.offset) / scaling.z.scale
    reference[:, 2] = z1
    reference_state = model(reference, z)

    f_off = torch.tensor(cfg.flux_offset, device=dev, dtype=dt)
    f_scale = torch.tensor(cfg.flux_scale, device=dev, dtype=dt)
    phys = surf[:, 3:] * f_scale + f_off                             # un-normalize fluxes
    fric_vel, ht_flux = phys[:, 0:1], phys[:, 1:2]

    res = no_penetration_z_wall_residuals(
        coordinates=wall, state=wall_state,
        reference_coordinates=reference, reference_state=reference_state,
        physics=DEFAULT_PHYSICS, scaling=scaling,
        fric_vel=fric_vel, ht_flux=ht_flux,
        surface_stress_scale=cfg.surface_stress_scale,
        surface_heat_flux_scale=cfg.surface_heat_flux_scale)
    return torch.stack([r.square().mean() for r in res.values()]).mean()


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
        surface = _surface_loss(model, case, z, cfg, scaling)
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

    start_epoch = _maybe_resume(model, optimizer, history, config, device)

    prof = {"fwd": 0.0, "bwd": 0.0, "step": 0.0} if config.profile else None
    cuda = device.type == "cuda"

    for epoch in range(start_epoch, config.epochs + 1):
        k = min(config.batch_cases, len(cases))
        idx = torch.randperm(len(cases), generator=rng)[:k].tolist()

        if prof is not None and cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
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
        if prof is not None:
            if cuda:
                torch.cuda.synchronize()
            prof["fwd"] += time.perf_counter() - t; t = time.perf_counter()
        total.backward()
        # Clip gradients: PDE residual gradients (2nd derivatives via autograd) can
        # spike and freeze training. Standard PINN safeguard.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        if prof is not None:
            if cuda:
                torch.cuda.synchronize()
            prof["bwd"] += time.perf_counter() - t; t = time.perf_counter()
        optimizer.step()
        if prof is not None:
            if cuda:
                torch.cuda.synchronize()
            prof["step"] += time.perf_counter() - t

        history.total.append(float(total.detach().cpu()))
        for name in weights:
            getattr(history, name).append(float(batch[name].detach().cpu()))

        if epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0:
            parts = " ".join(f"{n}={float(batch[n].detach()):.4e}" for n in weights)
            # stderr + flush so progress is live in the SLURM log, never buffered
            print(f"epoch {epoch}/{config.epochs} "
                  f"total={float(total.detach()):.4e} {parts}",
                  file=sys.stderr, flush=True)
            if prof is not None:
                tot = sum(prof.values()) or 1.0
                print(f"  profile/epoch(avg over {epoch}): "
                      f"fwd={prof['fwd']/epoch*1e3:.1f}ms ({prof['fwd']/tot*100:.0f}%) "
                      f"bwd={prof['bwd']/epoch*1e3:.1f}ms ({prof['bwd']/tot*100:.0f}%) "
                      f"step={prof['step']/epoch*1e3:.1f}ms ({prof['step']/tot*100:.0f}%)",
                      file=sys.stderr, flush=True)

        if config.checkpoint_path and (epoch % config.checkpoint_every == 0
                                       or epoch == config.epochs):
            _save_checkpoint(model, optimizer, history, epoch, config)

    return history


def _save_checkpoint(model, optimizer, history, epoch, config) -> None:
    """Atomically write a resumable checkpoint (model, optimizer, epoch, history)."""
    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": vars(history)}, tmp)
    tmp.replace(path)
    print(f"checkpoint @ epoch {epoch} -> {path}", file=sys.stderr, flush=True)


def _maybe_resume(model, optimizer, history, config, device) -> int:
    """Load a checkpoint if present; return the epoch to start from (1 if none)."""
    if not config.checkpoint_path or not Path(config.checkpoint_path).is_file():
        return 1
    ckpt = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    for name, vals in ckpt["history"].items():
        getattr(history, name).extend(vals)
    print(f"resumed from {config.checkpoint_path} at epoch {ckpt['epoch']}",
          file=sys.stderr, flush=True)
    return ckpt["epoch"] + 1

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
