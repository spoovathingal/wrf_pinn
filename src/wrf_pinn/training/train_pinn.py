"""General WRF PINN training loop.

This module is the first general training entry point for the WRF PINN package.
It keeps the objective assembly centralized in ``training.losses`` and uses
``config.conditions`` as the only place that decides which loss modes are active
and how strongly they are weighted.

PDE residual points are generated from ``config.domain`` and
``config.sampling``. Dense flow-field data, sensor data, and boundary data stay
separate from the collocation points used to enforce the governing equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from wrf_pinn.config.boundary_data import DEFAULT_BOUNDARIES, BoundaryConfig
from wrf_pinn.config.conditions import DEFAULT_CONDITIONS, ConditionsConfig
from wrf_pinn.config.domain import CartesianWRFDomain
from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.config.sampling import DEFAULT_SAMPLING, SamplingConfig
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig
from wrf_pinn.config.training import DEFAULT_TRAINING, OptimizerConfig, TrainingConfig
from wrf_pinn.data.case import Case, SRC_INLET, SRC_SIM, SRC_SENSOR
from wrf_pinn.physics.residuals_boundary import no_penetration_z_wall_residuals
from wrf_pinn.physics.residuals_boundary import no_slip_wall_residuals
from wrf_pinn.physics.residuals_pde import cartesian_zero_forcing_residuals
from wrf_pinn.sampling import sample_collocation_points
from wrf_pinn.sampling import sample_wall_boundary_points
from wrf_pinn.training.losses import (
    DEFAULT_PDE_RESIDUAL_SCALES, LossBreakdown, PDEResidualScales,
    assemble_pinn_loss,
)


#: Component loss names recorded in ``TrainingHistory`` (excludes ``total``).
COMPONENT_LOSS_NAMES: tuple[str, ...] = (
    "pde",
    "boundary",
    "inlet",
    "simulation",
    "sensor",
)


@dataclass
class TrainingHistory:
    """Loss history recorded during training."""

    total: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    boundary: list[float] = field(default_factory=list)
    inlet: list[float] = field(default_factory=list)
    simulation: list[float] = field(default_factory=list)
    sensor: list[float] = field(default_factory=list)


class TrainingMonitor(Protocol):
    """Optional live monitor called during training.

    Any object providing these methods can be passed as ``monitor`` to stream
    progress (for example, an incremental loss plot). ``update`` receives the
    current epoch, the total loss, and a dict of the component losses named by
    ``COMPONENT_LOSS_NAMES``. The concrete ``LiveTrainingMonitor`` in
    ``wrf_pinn.evaluation.live_monitor`` implements this protocol.
    """

    def update(self, *, epoch: int, total: float, components: dict[str, float]) -> None:
        ...

    def finalize(self) -> None:
        ...


@dataclass(frozen=True)
class TrainingSetup:
    """Everything a training run needs, assembled once and validated once.

    Data enters as one normalized ``Case`` (from the pre-processor); its rows
    carry a per-source tag so the data loss is weighted per source.
    """

    domain: CartesianWRFDomain | None = None
    case: Case | None = None
    boundaries: BoundaryConfig = DEFAULT_BOUNDARIES
    conditions: ConditionsConfig = DEFAULT_CONDITIONS
    sampling: SamplingConfig = DEFAULT_SAMPLING
    physics: PhysicsConfig = DEFAULT_PHYSICS
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING
    pde_residual_scales: PDEResidualScales = DEFAULT_PDE_RESIDUAL_SCALES
    training: TrainingConfig = DEFAULT_TRAINING


def train_pinn(
    model: nn.Module,
    setup: TrainingSetup = TrainingSetup(),
    *,
    monitor: TrainingMonitor | None = None,
) -> TrainingHistory:
    """Train a WRF PINN using the globally active condition modes.

    Parameters
    ----------
    model:
        Neural network mapping normalized ``(x, y, z, t)`` inputs to normalized
        ``(u, v, w, rho)`` outputs.
    setup:
        The bundled run configuration and data (domain, flow/sensor data,
        boundary/conditions/sampling/physics/scaling/training configs). See
        ``TrainingSetup``.
    monitor:
        Optional live monitor. When given, its ``update`` method is called on
        the logging cadence with the current epoch's total and component losses,
        and ``finalize`` is called once training ends. This lets a live plot
        stream progress during long runs without affecting the training loop.

    Returns
    -------
    TrainingHistory
        Per-epoch total and component losses.
    """

    _validate_active_inputs(setup)

    device = _resolve_device(setup.training.device)
    model.to(device)

    optimizer = build_optimizer(model, setup.training.optimizer)
    history = TrainingHistory()
    prepared = _PreparedData.from_setup(setup, device=device)

    for epoch in range(1, setup.training.epochs + 1):
        loss = _training_step(model, optimizer, setup, prepared)
        _record_history(history, loss)

        if _should_log(epoch, setup.training):
            _print_progress(epoch, setup.training.epochs, loss)
            _notify_monitor(monitor, epoch=epoch, loss=loss)

    if monitor is not None:
        monitor.finalize()

    return history


#: Condition name -> the source tag whose rows feed that data objective.
_DATA_SOURCE = {"inlet": SRC_INLET, "simulation": SRC_SIM, "sensor": SRC_SENSOR}


@dataclass
class _PreparedData:
    """Per-source (coordinates, targets, target_mask) tensors, split once by source tag."""

    device: torch.device
    by_source: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @classmethod
    def from_setup(cls, setup: TrainingSetup, *, device: torch.device) -> "_PreparedData":
        prepared = cls(device=device)
        if setup.case is None:
            return prepared
        coordinates, targets, target_mask = setup.case.as_torch(device=device)
        source = torch.as_tensor(setup.case.source, device=device)
        for code in torch.unique(source).tolist():
            rows = source == code
            prepared.by_source[int(code)] = (
                coordinates[rows], targets[rows], target_mask[rows]
            )
        return prepared


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    setup: TrainingSetup,
    prepared: _PreparedData,
) -> LossBreakdown:
    """Run one optimizer step and return the loss breakdown for the epoch."""

    optimizer.zero_grad()
    objectives = _evaluate_active_objectives(model, setup, prepared)
    loss = assemble_pinn_loss(
        conditions=setup.conditions,
        pde_residual_scales=setup.pde_residual_scales,
        **objectives,
    )
    if not bool(torch.isfinite(loss.total)):
        raise FloatingPointError("Non-finite total loss during training.")

    loss.total.backward()
    _raise_on_nonfinite_gradients(model)
    if setup.training.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            setup.training.gradient_clip_norm,
        )
    optimizer.step()
    return loss


def _raise_on_nonfinite_gradients(model: nn.Module) -> None:
    """Fail fast if any parameter gradient is non-finite (catches blowups early)."""

    bad = [
        name for name, p in model.named_parameters()
        if p.grad is not None and not bool(torch.isfinite(p.grad).all())
    ]
    if bad:
        raise FloatingPointError(f"Non-finite gradients in: {bad}.")


def _evaluate_active_objectives(
    model: nn.Module,
    setup: TrainingSetup,
    prepared: _PreparedData,
) -> dict[str, dict[str, torch.Tensor] | None]:
    """Evaluate residuals/errors for each active objective.

    Returns a mapping in the keyword shape ``assemble_pinn_loss`` expects, with
    ``None`` for inactive objectives.
    """

    conditions = setup.conditions
    pde_residuals: dict[str, torch.Tensor] | None = None
    boundary_residuals: dict[str, torch.Tensor] | None = None

    if conditions.pde.active:
        pde_coordinates = _sample_pde_coordinates(
            domain=setup.domain,
            sampling=setup.sampling,
            device=prepared.device,
        )
        pde_state = model(pde_coordinates)
        pde_residuals = cartesian_zero_forcing_residuals(
            pde_coordinates,
            pde_state,
            physics=setup.physics,
            scaling=setup.scaling,
        )

    if conditions.boundary.active:
        boundary_residuals = _boundary_residuals(model, setup, prepared.device)

    objectives: dict[str, dict[str, torch.Tensor] | None] = {
        "pde_residuals": pde_residuals,
        "boundary_residuals": boundary_residuals,
    }
    # One data term per source category (inlet / simulation / sensor), each fed
    # by that source's rows via the case's source tag.
    for name, code in _DATA_SOURCE.items():
        spec = getattr(conditions, name)
        errors = None
        masks = None
        if spec.active:
            coordinates, targets, target_mask = _require_source_rows(prepared, code, name)
            errors = {name: _data_errors(model, coordinates, targets, target_mask)}
            masks = {name: target_mask}
        objectives[f"{name}_errors"] = errors
        objectives[f"{name}_masks"] = masks
    return objectives


def _boundary_residuals(
    model: nn.Module, setup: TrainingSetup, device: torch.device
) -> dict[str, torch.Tensor]:
    """Wall residuals for the configured wall condition."""

    wall_coordinates = sample_wall_boundary_points(
        boundary=setup.boundaries.no_slip_wall,
        sampling=setup.sampling.boundary,
        case=setup.case,
        seed=setup.sampling.seed,
        device=device,
    )
    wall_state = model(wall_coordinates)

    if setup.boundaries.no_slip_wall.condition == "no_penetration_z":
        return no_penetration_z_wall_residuals(wall_state, scaling=setup.scaling)
    return no_slip_wall_residuals(wall_state, scaling=setup.scaling)


def _data_errors(
    model: nn.Module,
    coordinates: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked prediction-minus-target error (unmeasured entries zeroed)."""

    predictions = model(coordinates)
    error = predictions[:, : targets.shape[1]] - targets
    return error * target_mask


def _notify_monitor(
    monitor: TrainingMonitor | None, *, epoch: int, loss: LossBreakdown
) -> None:
    """Stream the epoch's losses to the monitor, if one is attached."""

    if monitor is None:
        return
    monitor.update(
        epoch=epoch,
        total=float(loss.total.detach().cpu()),
        components={
            name: float(loss.terms[name].detach().cpu())
            for name in COMPONENT_LOSS_NAMES
        },
    )


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build an optimizer from training configuration."""

    parameters = model.parameters()

    if config.name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {config.name}.")


def _validate_active_inputs(setup: TrainingSetup) -> None:
    """Fail early when an active objective has no corresponding data source."""

    conditions = setup.conditions

    if conditions.pde.active and setup.domain is None:
        raise ValueError("conditions.pde is active, but domain is None.")

    data_active = any(getattr(conditions, name).active for name in _DATA_SOURCE)
    if data_active and setup.case is None:
        raise ValueError("A data condition is active, but setup.case is None.")

    if conditions.boundary.active:
        if not setup.boundaries.no_slip_wall.surface.path:
            raise ValueError(
                "conditions.boundary is active, but no no-slip wall surface "
                "path is configured."
            )
        if setup.case is None:
            raise ValueError(
                "conditions.boundary is active, but no case was provided to "
                "supply boundary times."
            )


def _sample_pde_coordinates(
    *,
    domain: CartesianWRFDomain | None,
    sampling: SamplingConfig,
    device: torch.device,
) -> torch.Tensor:
    """Generate autograd-ready collocation coordinates for PDE residuals."""

    if domain is None:
        raise ValueError("Cannot sample PDE coordinates because domain is None.")

    coordinates = sample_collocation_points(
        domain=domain,
        sampling=sampling.collocation,
        seed=sampling.seed,
        device=device,
    )
    return coordinates.requires_grad_(True)


def _require_source_rows(
    prepared: "_PreparedData", code: int, name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (coordinates, targets, target_mask) for a source, or raise if none."""

    rows = prepared.by_source.get(code)
    if rows is None:
        raise ValueError(
            f"conditions.{name} is active, but the case has no rows tagged "
            f"'{name}' (source {code})."
        )
    return rows


def _resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def _record_history(history: TrainingHistory, loss: LossBreakdown) -> None:
    """Append scalar loss values to history."""

    history.total.append(float(loss.total.detach().cpu()))
    for name in COMPONENT_LOSS_NAMES:
        getattr(history, name).append(float(loss.terms[name].detach().cpu()))


def _should_log(epoch: int, training: TrainingConfig) -> bool:
    """Return true when this epoch should be printed."""

    return epoch == 1 or epoch == training.epochs or epoch % training.log_every == 0


def _print_progress(epoch: int, epochs: int, loss: LossBreakdown) -> None:
    """Print one concise training progress line."""

    parts = [
        f"epoch={epoch}/{epochs}",
        f"total={float(loss.total.detach().cpu()):.6e}",
    ]
    parts += [
        f"{name}={float(loss.terms[name].detach().cpu()):.6e}"
        for name in COMPONENT_LOSS_NAMES
    ]
    # Per-residual raw vs scaled MSE (the PDE residual scaling diagnostics).
    for name, raw_mse in loss.pde_raw_mse.items():
        scaled_mse = loss.pde_scaled_mse[name]
        parts.append(f"{name}_raw_mse={float(raw_mse.detach().cpu()):.6e}")
        parts.append(f"{name}_scaled_mse={float(scaled_mse.detach().cpu()):.6e}")
    print(" | ".join(parts))
