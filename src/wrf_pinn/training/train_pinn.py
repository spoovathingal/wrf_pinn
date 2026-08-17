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
from wrf_pinn.data.flow_field import FlowFieldData
from wrf_pinn.data.sensors import SensorData
from wrf_pinn.physics.residuals_boundary import no_penetration_z_wall_residuals
from wrf_pinn.physics.residuals_boundary import no_slip_wall_residuals
from wrf_pinn.physics.residuals_pde import cartesian_zero_forcing_residuals
from wrf_pinn.sampling import sample_collocation_points
from wrf_pinn.sampling import sample_wall_boundary_points
from wrf_pinn.training.losses import LossBreakdown, assemble_pinn_loss


#: Component loss names recorded in ``TrainingHistory`` (excludes ``total``).
COMPONENT_LOSS_NAMES: tuple[str, ...] = (
    "pde",
    "boundary",
    "sensor_data",
    "flow_field_data",
)


@dataclass
class TrainingHistory:
    """Loss history recorded during training."""

    total: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    boundary: list[float] = field(default_factory=list)
    sensor_data: list[float] = field(default_factory=list)
    flow_field_data: list[float] = field(default_factory=list)


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

    Grouping the per-run configuration and data here keeps ``train_pinn`` down
    to three parameters (model, setup, monitor). Callers already build these
    objects individually, so bundling them is free at the call site.

    Notes
    -----
    ``flow_field_data`` and ``sensor_data`` are the current, format-specific data
    inputs. They are expected to be replaced by a single normalized ``Case`` seam
    (see the Change 2 roadmap), so no abstraction is layered over them here.
    """

    domain: CartesianWRFDomain | None = None
    flow_field_data: FlowFieldData | None = None
    sensor_data: SensorData | None = None
    boundaries: BoundaryConfig = DEFAULT_BOUNDARIES
    conditions: ConditionsConfig = DEFAULT_CONDITIONS
    sampling: SamplingConfig = DEFAULT_SAMPLING
    physics: PhysicsConfig = DEFAULT_PHYSICS
    scaling: ResidualScalingConfig = DEFAULT_RESIDUAL_SCALING
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


@dataclass
class _PreparedData:
    """The resolved device plus the data tensors placed on it.

    Materialized once before the epoch loop. Carrying the device here keeps it
    with the tensors it applies to, so the per-epoch helpers do not have to pass
    it around separately.
    """

    device: torch.device
    flow_coordinates: torch.Tensor | None = None
    flow_targets: torch.Tensor | None = None
    sensor_coordinates: torch.Tensor | None = None
    sensor_targets: torch.Tensor | None = None

    @classmethod
    def from_setup(cls, setup: TrainingSetup, *, device: torch.device) -> "_PreparedData":
        prepared = cls(device=device)
        if setup.flow_field_data is not None:
            prepared.flow_coordinates, prepared.flow_targets = (
                setup.flow_field_data.as_torch(device=device)
            )
        if setup.sensor_data is not None:
            prepared.sensor_coordinates, prepared.sensor_targets = (
                setup.sensor_data.as_torch(device=device)
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
    loss = assemble_pinn_loss(conditions=setup.conditions, **objectives)

    loss.total.backward()
    if setup.training.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            setup.training.gradient_clip_norm,
        )
    optimizer.step()
    return loss


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
    sensor_data_errors: dict[str, torch.Tensor] | None = None
    flow_field_data_errors: dict[str, torch.Tensor] | None = None

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

    if conditions.sensor_data.active:
        coordinates = _require_tensor(
            prepared.sensor_coordinates, "sensor_data.coordinates"
        )
        targets = _require_tensor(prepared.sensor_targets, "sensor_data.targets")
        sensor_data_errors = {
            "sensor_velocity": _data_errors(model, coordinates, targets),
        }

    if conditions.flow_field_data.active:
        coordinates = _require_tensor(
            prepared.flow_coordinates, "flow_field_data.coordinates"
        )
        targets = _require_tensor(prepared.flow_targets, "flow_field_data.targets")
        flow_field_data_errors = {
            "flow_field": _data_errors(model, coordinates, targets),
        }

    return {
        "pde_residuals": pde_residuals,
        "boundary_residuals": boundary_residuals,
        "sensor_data_errors": sensor_data_errors,
        "flow_field_data_errors": flow_field_data_errors,
    }


def _boundary_residuals(
    model: nn.Module, setup: TrainingSetup, device: torch.device
) -> dict[str, torch.Tensor]:
    """Wall residuals for the configured wall condition."""

    wall_coordinates = sample_wall_boundary_points(
        boundary=setup.boundaries.no_slip_wall,
        sampling=setup.sampling.boundary,
        flow_field_data=setup.flow_field_data,
        sensor_data=setup.sensor_data,
        seed=setup.sampling.seed,
        device=device,
    )
    wall_state = model(wall_coordinates)

    if setup.boundaries.no_slip_wall.condition == "no_penetration_z":
        return no_penetration_z_wall_residuals(wall_state, scaling=setup.scaling)
    return no_slip_wall_residuals(wall_state, scaling=setup.scaling)


def _data_errors(
    model: nn.Module, coordinates: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Prediction-minus-target error for a data objective."""

    predictions = model(coordinates)
    return predictions[:, : targets.shape[1]] - targets


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

    if conditions.flow_field_data.active and setup.flow_field_data is None:
        raise ValueError(
            "conditions.flow_field_data is active, but flow_field_data is None."
        )

    if conditions.sensor_data.active and setup.sensor_data is None:
        raise ValueError("conditions.sensor_data is active, but sensor_data is None.")

    if conditions.boundary.active:
        if not setup.boundaries.no_slip_wall.surface.path:
            raise ValueError(
                "conditions.boundary is active, but no no-slip wall surface "
                "path is configured."
            )

        if setup.flow_field_data is None and setup.sensor_data is None:
            raise ValueError(
                "conditions.boundary is active, but no flow_field_data or "
                "sensor_data was provided to supply boundary times."
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


def _require_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    """Return a tensor or raise a descriptive error."""

    if value is None:
        raise ValueError(f"Required tensor is missing: {name}.")

    return value


def _resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def _record_history(history: TrainingHistory, loss: LossBreakdown) -> None:
    """Append scalar loss values to history."""

    history.total.append(float(loss.total.detach().cpu()))
    history.pde.append(float(loss.terms["pde"].detach().cpu()))
    history.boundary.append(float(loss.terms["boundary"].detach().cpu()))
    history.sensor_data.append(float(loss.terms["sensor_data"].detach().cpu()))
    history.flow_field_data.append(
        float(loss.terms["flow_field_data"].detach().cpu()),
    )


def _should_log(epoch: int, training: TrainingConfig) -> bool:
    """Return true when this epoch should be printed."""

    return epoch == 1 or epoch == training.epochs or epoch % training.log_every == 0


def _print_progress(epoch: int, epochs: int, loss: LossBreakdown) -> None:
    """Print one concise training progress line."""

    parts = [
        f"epoch={epoch}/{epochs}",
        f"total={float(loss.total.detach().cpu()):.6e}",
        f"pde={float(loss.terms['pde'].detach().cpu()):.6e}",
        f"boundary={float(loss.terms['boundary'].detach().cpu()):.6e}",
        f"sensor_data={float(loss.terms['sensor_data'].detach().cpu()):.6e}",
        f"flow_field_data={float(loss.terms['flow_field_data'].detach().cpu()):.6e}",
    ]
    print(" | ".join(parts))
