"""Conditional-PINN training loop (minimum working example).

Trains one shared ConditionalModel across a family of cases. Each step:
  1. draw a batch of cases,
  2. encode each case's (initial, boundary, terrain) -> z,
  3. predict the interior at the case's query coordinates,
  4. accumulate the data + initial + flow-boundary losses,
  5. average over the batch and step.

Losses implemented here (the data-driven ones the MWE needs to show a result):
  data           predicted interior vs. targets (masked)
  initial        predicted state at tau=0 vs. the initial field phi
  flow_boundary  predicted state on the open faces vs. the boundary history psi

PDE and surface-boundary losses are physics terms that need the residual machinery
and a fixed surface; they are deferred for the MWE and slot in via the same per-case
accumulation. Everything is intentionally small: this proves the pipeline runs and
the loss decreases, with figures to follow.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch
from torch import nn

from wrf_pinn.data.conditional_case import ConditionalCase, COORD_NAMES, STATE_VARS


@dataclass
class ConditionalHistory:
    """Per-epoch loss history."""

    total: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)
    initial: list[float] = field(default_factory=list)
    flow_boundary: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ConditionalTrainConfig:
    epochs: int = 2000
    batch_cases: int = 4          # cases drawn per step
    learning_rate: float = 1e-3
    weight_data: float = 1.0
    weight_initial: float = 1.0
    weight_flow_boundary: float = 1.0
    log_every: int = 100
    device: str = "auto"
    seed: int = 0


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _case_losses(model: nn.Module, case: ConditionalCase) -> dict[str, torch.Tensor]:
    """The three data-driven losses for one case, on the decoded output."""
    z = model.encode(case.initial, case.boundary, case.terrain)
    n_coord, n_state = len(COORD_NAMES), len(STATE_VARS)

    # data: interior query coordinates -> targets (masked)
    pred = model(case.interior, z)
    err = (pred - case.targets) * case.target_mask
    measured = case.target_mask.sum().clamp_min(1.0)
    data = err.square().sum() / measured

    # initial: phi = coords (tau=0) then state; predict at those coords, match state
    ic_coords = case.initial[:, :n_coord]
    ic_state = case.initial[:, n_coord:]
    initial = (model(ic_coords, z) - ic_state).square().mean()

    # flow boundary: psi (faces, times, len, state). Build face query coords is a
    # future refinement; for the MWE we match the model's state statistics to the
    # face history at the interior query points' times. Minimal, data-driven term.
    flow = (model(case.interior, z).mean(0) - case.boundary.mean(dim=(0, 1, 2))
            ).square().mean()

    return {"data": data, "initial": initial, "flow_boundary": flow}


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
               "flow_boundary": config.weight_flow_boundary}

    for epoch in range(1, config.epochs + 1):
        k = min(config.batch_cases, len(cases))
        idx = torch.randperm(len(cases), generator=rng)[:k].tolist()

        optimizer.zero_grad()
        batch = {name: torch.zeros((), device=device) for name in weights}
        for i in idx:
            for name, value in _case_losses(model, cases[i]).items():
                batch[name] = batch[name] + value
        for name in batch:
            batch[name] = batch[name] / k
        total = sum(weights[name] * batch[name] for name in weights)

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
        total.backward()
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
