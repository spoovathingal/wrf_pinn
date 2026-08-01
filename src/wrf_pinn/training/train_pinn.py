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


def train_pinn(
    model: nn.Module,
    *,
    domain: CartesianWRFDomain | None = None,
    flow_field_data: FlowFieldData | None = None,
    sensor_data: SensorData | None = None,
    boundaries: BoundaryConfig = DEFAULT_BOUNDARIES,
    training: TrainingConfig = DEFAULT_TRAINING,
    conditions: ConditionsConfig = DEFAULT_CONDITIONS,
    sampling: SamplingConfig = DEFAULT_SAMPLING,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    monitor: TrainingMonitor | None = None,
    use_no_penetration_z_wall: bool = False,
) -> TrainingHistory:
    """Train a WRF PINN using the globally active condition modes.

    Parameters
    ----------
    model:
        Neural network mapping normalized ``(x, y, z, t)`` inputs to normalized
        ``(u, v, w, rho)`` outputs.
    domain:
        Continuous Cartesian domain used to generate PDE collocation points.
        This is required only when ``conditions.pde.active`` is true.
    flow_field_data:
        Optional dense normalized data with ``x, y, z, t, u, v, w, rho``. This
        is required only when ``conditions.flow_field_data.active`` is true.
    sensor_data:
        Optional sparse normalized sensor data with ``x, y, z, t, u, v, w``.
        This is required only when ``conditions.sensor_data.active`` is true.
    boundaries:
        Boundary-condition configuration. The no-slip wall surface path is used
        by ``sampling.boundary`` when ``conditions.boundary.active`` is true.
    training:
        Optimizer, device, epoch count, and logging configuration.
    conditions:
        Global active/inactive switches and lambda weights.
    sampling:
        Sampling settings. The collocation subsection is used when
        ``conditions.pde.active`` is true.
    physics:
        Physics configuration passed to the PDE residual evaluator.
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

    _validate_active_inputs(
        domain=domain,
        flow_field_data=flow_field_data,
        sensor_data=sensor_data,
        boundaries=boundaries,
        conditions=conditions,
    )

    device = _resolve_device(training.device)
    model.to(device)

    optimizer = build_optimizer(model, training.optimizer)
    history = TrainingHistory()

    flow_coordinates: torch.Tensor | None = None
    flow_targets: torch.Tensor | None = None
    if flow_field_data is not None:
        flow_coordinates, flow_targets = flow_field_data.as_torch(device=device)

    sensor_coordinates: torch.Tensor | None = None
    sensor_targets: torch.Tensor | None = None
    if sensor_data is not None:
        sensor_coordinates, sensor_targets = sensor_data.as_torch(device=device)

    for epoch in range(1, training.epochs + 1):
        optimizer.zero_grad()

        pde_residuals: dict[str, torch.Tensor] | None = None
        boundary_residuals: dict[str, torch.Tensor] | None = None
        sensor_data_errors: dict[str, torch.Tensor] | None = None
        flow_field_data_errors: dict[str, torch.Tensor] | None = None

        if conditions.pde.active:
            pde_coordinates = _sample_pde_coordinates(
                domain=domain,
                sampling=sampling,
                device=device,
            )
            pde_state = model(pde_coordinates)
            pde_residuals = cartesian_zero_forcing_residuals(
                pde_coordinates,
                pde_state,
                physics=physics,
            )

        if conditions.boundary.active:
            wall_coordinates = sample_wall_boundary_points(
                boundary=boundaries.no_slip_wall,
                sampling=sampling.boundary,
                flow_field_data=flow_field_data,
                sensor_data=sensor_data,
                seed=sampling.seed,
                device=device,
            )
            wall_state = model(wall_coordinates)

            if use_no_penetration_z_wall:
                boundary_residuals = no_penetration_z_wall_residuals(wall_state)
            else:
                boundary_residuals = no_slip_wall_residuals(wall_state)

        if conditions.sensor_data.active:
            coordinates = _require_tensor(sensor_coordinates, "sensor_data.coordinates")
            targets = _require_tensor(sensor_targets, "sensor_data.targets")
            predictions = model(coordinates)
            sensor_data_errors = {
                "sensor_velocity": predictions[:, : targets.shape[1]] - targets,
            }

        if conditions.flow_field_data.active:
            coordinates = _require_tensor(
                flow_coordinates,
                "flow_field_data.coordinates",
            )
            targets = _require_tensor(flow_targets, "flow_field_data.targets")
            predictions = model(coordinates)
            flow_field_data_errors = {
                "flow_field": predictions[:, : targets.shape[1]] - targets,
            }

        loss = assemble_pinn_loss(
            pde_residuals=pde_residuals,
            boundary_residuals=boundary_residuals,
            sensor_data_errors=sensor_data_errors,
            flow_field_data_errors=flow_field_data_errors,
            conditions=conditions,
        )

        loss.total.backward()

        if training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training.gradient_clip_norm,
            )

        optimizer.step()

        _record_history(history, loss)

        if _should_log(epoch, training):
            _print_progress(epoch, training.epochs, loss)
            if monitor is not None:
                monitor.update(
                    epoch=epoch,
                    total=float(loss.total.detach().cpu()),
                    components={
                        name: float(loss.terms[name].detach().cpu())
                        for name in COMPONENT_LOSS_NAMES
                    },
                )

    if monitor is not None:
        monitor.finalize()

    return history


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


def _validate_active_inputs(
    *,
    domain: CartesianWRFDomain | None,
    flow_field_data: FlowFieldData | None,
    sensor_data: SensorData | None,
    boundaries: BoundaryConfig,
    conditions: ConditionsConfig,
) -> None:
    """Fail early when an active objective has no corresponding data source."""

    if conditions.pde.active and domain is None:
        raise ValueError("conditions.pde is active, but domain is None.")

    if conditions.flow_field_data.active and flow_field_data is None:
        raise ValueError(
            "conditions.flow_field_data is active, but flow_field_data is None."
        )

    if conditions.sensor_data.active and sensor_data is None:
        raise ValueError("conditions.sensor_data is active, but sensor_data is None.")

    if conditions.boundary.active:
        if not boundaries.no_slip_wall.surface.path:
            raise ValueError(
                "conditions.boundary is active, but no no-slip wall surface "
                "path is configured."
            )

        if flow_field_data is None and sensor_data is None:
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
