"""Training loop for dense flow-field data plus PDE residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from wrf_pinn.config.conditions import DEFAULT_CONDITIONS, ConditionsConfig
from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.config.training import DEFAULT_TRAINING, OptimizerConfig
from wrf_pinn.config.training import TrainingConfig
from wrf_pinn.data.flow_field import FlowFieldData
from wrf_pinn.physics.residuals import cartesian_zero_forcing_residuals
from wrf_pinn.training.losses import LossBreakdown, assemble_pinn_loss


@dataclass(frozen=True)
class TrainingHistory:
    """Scalar loss history recorded during training."""

    total: list[float]
    pde: list[float]
    data: list[float]


def train_dense_flow_field(
    *,
    model: nn.Module,
    flow_data: FlowFieldData,
    training: TrainingConfig = DEFAULT_TRAINING,
    conditions: ConditionsConfig = DEFAULT_CONDITIONS,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
) -> TrainingHistory:
    """Train a model using dense flow-field data and PDE residuals.

    This first trainer assumes all coordinates and targets are already
    normalized/scaled before they enter the codebase. It uses the full dataset as
    one batch.
    """

    device = _resolve_device(training.device)
    model.to(device)

    coordinates, targets = flow_data.as_torch(device=device)
    optimizer = build_optimizer(model, training.optimizer)

    history = TrainingHistory(total=[], pde=[], data=[])
    for epoch in range(training.epochs):
        coordinates_pde = coordinates.detach().clone().requires_grad_(True)
        predictions = model(coordinates_pde)

        data_errors = {
            "flow_field": predictions - targets,
        }
        pde_residuals = cartesian_zero_forcing_residuals(
            coordinates=coordinates_pde,
            state=predictions,
            physics=physics,
        )
        loss = assemble_pinn_loss(
            pde_residuals=pde_residuals,
            data_errors=data_errors,
            conditions=conditions,
        )

        optimizer.zero_grad()
        loss.total.backward()
        if training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training.gradient_clip_norm,
            )
        optimizer.step()

        _record_history(history, loss)
        if _should_log(epoch, training):
            _print_progress(epoch, training, loss)

    return history


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build the optimizer configured for training."""

    if config.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=config.learning_rate,
        )

    raise ValueError(f"Unsupported optimizer: {config.name}.")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def _record_history(history: TrainingHistory, loss: LossBreakdown) -> None:
    history.total.append(float(loss.total.detach().cpu()))
    history.pde.append(float(loss.terms["pde"].detach().cpu()))
    history.data.append(float(loss.terms["data"].detach().cpu()))


def _should_log(epoch: int, training: TrainingConfig) -> bool:
    return epoch == 0 or (epoch + 1) % training.log_every == 0


def _print_progress(
    epoch: int,
    training: TrainingConfig,
    loss: LossBreakdown,
) -> None:
    total = float(loss.total.detach().cpu())
    pde = float(loss.terms["pde"].detach().cpu())
    data = float(loss.terms["data"].detach().cpu())
    print(
        f"epoch={epoch + 1}/{training.epochs} "
        f"total={total:.6e} pde={pde:.6e} data={data:.6e}"
    )
