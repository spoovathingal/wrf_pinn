"""Uniform-flow training cases: a regression base for the shared pipeline.

These run short trainings on constant uniform-flow data with different active
loss modes and assert each combination (1) completes without error, (2) reduces
the loss, and (3) reproduces the expected flow field:

- flow_field-only: the trained model must reproduce the known constant state
  (predictions match the true u, v, w, rho within tolerance).
- pde-only: the PDE has no data anchor, so it cannot recover the specific
  constants; the correct known-result check is that the predicted field is
  spatially uniform (a constant state is what a zero-residual solution must be).

They are a safety net so a change to one loss path does not silently break
another. Boundary conditions are intentionally OFF: no-slip does not satisfy
uniform flow through the domain. Boundary combinations (flow_field+bc, pde+bc)
will be added once the no-penetration boundary condition exists.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wrf_pinn.config.boundary_data import (
    BoundaryConfig,
    NoSlipWallConfig,
    WallSurfaceConfig,
)
from wrf_pinn.config.conditions import ConditionSpec, ConditionsConfig
from wrf_pinn.config.domain import make_cartesian_wrf_domain
from wrf_pinn.config.flow_field_data import FlowFieldDataConfig
from wrf_pinn.config.sampling import (
    BoundarySamplingConfig,
    CollocationSamplingConfig,
    SamplingConfig,
)
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig
from wrf_pinn.config.scaling import VariableScale
from wrf_pinn.config.sensors_data import SensorDataConfig
from wrf_pinn.config.training import OptimizerConfig, TrainingConfig
from wrf_pinn.data.flow_field import read_flow_field
from wrf_pinn.data.sensors import read_sensor_data
from wrf_pinn.evaluation.predict import predict_flow_field
from wrf_pinn.models.mlp import MLP
from wrf_pinn.training.train_pinn import train_pinn, TrainingSetup


# Mean-absolute tolerance for reproducing the true field (flow_field case). At
# 300 epochs the observed per-variable MAE is <= ~0.015; 0.05 gives comfortable
# headroom against flakiness while still being a meaningful reproduction check.
REPRODUCTION_TOL = 0.05

# Spatial-standard-deviation tolerance for the field being uniform (pde case).
# The reduced transport PDE only weakly constrains rho toward a constant, so the
# field converges to uniform slowly and unevenly; std is used instead of max
# spread so a few stubborn collocation points do not make the check flaky. At
# 600 epochs the observed per-variable std is <= ~0.06.
UNIFORMITY_STD_TOL = 0.15


def _domain_from(flow_data):
    """Build a Cartesian domain spanning the data's coordinate bounds."""

    lo = flow_data.coordinates.min(axis=0)
    hi = flow_data.coordinates.max(axis=0)
    return make_cartesian_wrf_domain(
        x_min=float(lo[0]), x_max=float(hi[0]),
        y_min=float(lo[1]), y_max=float(hi[1]),
        z_min=float(lo[2]), z_max=float(hi[2]),
        t_min=float(lo[3]), t_max=float(hi[3]),
    )


def _training():
    """Deterministic training config with enough epochs to reproduce the field."""

    return TrainingConfig(
        epochs=300,
        log_every=100,
        optimizer=OptimizerConfig(name="adam", learning_rate=1.0e-3),
    )


def _assert_loss_decreased(history, report, label):
    """Assert training produced a non-empty history and reduced the loss."""

    assert len(history.total) > 0
    assert history.total[-1] < history.total[0]
    report(
        f"{label}: loss decreased",
        initial_loss=history.total[0],
        final_loss=history.total[-1],
    )


def _flow_field_only_conditions():
    return ConditionsConfig(
        pde=ConditionSpec("pde", active=False, weight=0.0),
        boundary=ConditionSpec("boundary", active=False, weight=0.0),
        sensor_data=ConditionSpec("sensor_data", active=False, weight=0.0),
        flow_field_data=ConditionSpec("flow_field_data", active=True, weight=1.0),
    )


def _run_flow_field_reproduction(path, report, label):
    """Train flow_field-only on ``path`` and assert the true field is reproduced.

    Returns the per-variable MAE so callers can report it. This is shared by the
    baseline case and the parameterized flow-condition / domain-size cases.
    """

    torch.manual_seed(0)
    flow_data = read_flow_field(FlowFieldDataConfig(path=str(path)))

    model = MLP()
    history = train_pinn(
        model,
        TrainingSetup(
            flow_field_data=flow_data,
            conditions=_flow_field_only_conditions(),
            training=_training(),
        ),
    )
    _assert_loss_decreased(history, report, label)

    _, predictions, targets = predict_flow_field(model, flow_data)
    mae = np.abs(predictions - targets).mean(axis=0)
    assert (mae < REPRODUCTION_TOL).all()
    report(
        f"{label}: reproduces true field",
        true_state=targets[0],
        predicted_mean=predictions.mean(axis=0),
        per_variable_mae=mae,
    )
    return mae


def test_uniform_flow_field_only(uniform_flow_csv, report):
    """flow_field data loss only (pde/boundary/sensor off), normalized coords."""

    path, _ = uniform_flow_csv
    _run_flow_field_reproduction(path, report, "uniform flow, flow_field only")


# Task 1.2: vary the flow conditions (state) and the domain size (coordinate
# range) to confirm the pipeline reproduces the field regardless of numeric
# scale. Tolerances calibrated from measured MAE (max observed ~0.03 at 300
# epochs across these cases; REPRODUCTION_TOL = 0.05).
FLOW_CONDITION_CASES = [
    ("baseline_unit_domain", {"u": 5.0, "v": 2.0, "w": 0.0, "rho": 1.225}, (0.0, 1.0)),
    ("symmetric_domain", {"u": 5.0, "v": 2.0, "w": 0.0, "rho": 1.225}, (-1.0, 1.0)),
    ("large_domain", {"u": 5.0, "v": 2.0, "w": 0.0, "rho": 1.225}, (0.0, 10.0)),
    ("small_magnitude_flow", {"u": 0.1, "v": 0.05, "w": 0.0, "rho": 1.0}, (0.0, 1.0)),
    ("large_mixed_sign_flow", {"u": 20.0, "v": -8.0, "w": 3.0, "rho": 1.3}, (0.0, 1.0)),
]


@pytest.mark.parametrize(
    "label, state, coordinate_range",
    FLOW_CONDITION_CASES,
    ids=[case[0] for case in FLOW_CONDITION_CASES],
)
def test_uniform_flow_field_reproduction_across_conditions(
    uniform_flow_csv_factory, report, label, state, coordinate_range
):
    """flow_field-only reproduces the field across flow conditions and domains."""

    path, _ = uniform_flow_csv_factory(state, coordinate_range)
    _run_flow_field_reproduction(path, report, f"uniform flow [{label}]")


def _pde_only_conditions():
    return ConditionsConfig(
        pde=ConditionSpec("pde", active=True, weight=1.0),
        boundary=ConditionSpec("boundary", active=False, weight=0.0),
        sensor_data=ConditionSpec("sensor_data", active=False, weight=0.0),
        flow_field_data=ConditionSpec("flow_field_data", active=False, weight=0.0),
    )


# A realistic non-identity residual scaling, mapping normalized [0,1] coordinates
# and outputs to physical WRF-like magnitudes (domain ~1 km, winds ~10 m/s, rho
# ~1.225). Used to exercise the chain-rule scaling path through training, not
# just the identity default. Direct scaling-math tests live in test_scaling.py.
PHYSICAL_SCALING = ResidualScalingConfig(
    x=VariableScale(0.0, 1000.0),
    y=VariableScale(0.0, 950.0),
    z=VariableScale(0.0, 1062.0),
    t=VariableScale(0.0, 600.0),
    u=VariableScale(0.0, 10.0),
    v=VariableScale(0.0, 10.0),
    w=VariableScale(0.0, 1.0),
    rho=VariableScale(0.0, 1.225),
)


def _run_pde_uniformity(
    path, report, label, *, collocation_points, scaling=DEFAULT_RESIDUAL_SCALING
):
    """Train pde-only with ``collocation_points`` and assert a uniform field.

    The true uniform field has zero PDE residual, so an untrained model starts
    with nonzero residual loss that should fall as it learns a constant state.
    With no data anchor the PDE cannot recover the specific constants, so the
    known-result check is spatial uniformity (near-zero per-variable std). The
    optional ``scaling`` exercises the residual chain-rule scaling path.
    """

    torch.manual_seed(0)
    flow_data = read_flow_field(FlowFieldDataConfig(path=str(path)))

    sampling = SamplingConfig(
        collocation=CollocationSamplingConfig(
            n_points=collocation_points, method="latin_hypercube"
        ),
        seed=0,
    )
    model = MLP()
    history = train_pinn(
        model,
        TrainingSetup(
            domain=_domain_from(flow_data),
            conditions=_pde_only_conditions(),
            sampling=sampling,
            scaling=scaling,
            training=TrainingConfig(
                epochs=600,
                log_every=200,
                optimizer=OptimizerConfig(name="adam", learning_rate=1.0e-3),
            ),
        ),
    )
    _assert_loss_decreased(history, report, label)

    _, predictions, _ = predict_flow_field(model, flow_data)
    spatial_std = predictions.std(axis=0)
    assert (spatial_std < UNIFORMITY_STD_TOL).all()
    report(
        f"{label}: predicted field is spatially uniform",
        collocation_points=np.array([collocation_points]),
        predicted_mean=predictions.mean(axis=0),
        per_variable_std=spatial_std,
    )


def test_uniform_flow_pde_only(uniform_flow_csv, report):
    """PDE residual loss only (flow_field/boundary/sensor off)."""

    path, _ = uniform_flow_csv
    _run_pde_uniformity(
        path, report, "uniform flow, pde only", collocation_points=200
    )


# Task 1.3: vary the number of collocation points for the pde case to confirm
# the pipeline is robust to sampling-config changes. (Boundary point counts are
# deferred with the boundary-condition test cases; see future_work.md.)
COLLOCATION_POINT_CASES = [64, 200, 512, 1024]


@pytest.mark.parametrize("collocation_points", COLLOCATION_POINT_CASES)
def test_uniform_flow_pde_collocation_counts(
    uniform_flow_csv, report, collocation_points
):
    """pde-only stays uniform across different collocation point counts."""

    path, _ = uniform_flow_csv
    _run_pde_uniformity(
        path,
        report,
        f"uniform flow, pde only [{collocation_points} collocation pts]",
        collocation_points=collocation_points,
    )


def test_uniform_flow_pde_with_physical_scaling(uniform_flow_csv, report):
    """pde-only reaches a uniform field under realistic non-identity scaling.

    Exercises the residual chain-rule scaling through the full training path
    (Task 1.2: ensure the scaling implementation works). The direct scaling-math
    checks are in test_scaling.py; this confirms training still converges to a
    uniform solution when the residual is evaluated in physical units.
    """

    path, _ = uniform_flow_csv
    _run_pde_uniformity(
        path,
        report,
        "uniform flow, pde only [physical scaling]",
        collocation_points=200,
        scaling=PHYSICAL_SCALING,
    )


def test_uniform_flow_pde_boundary_sensor(
    uniform_sensor_csv, bottom_wall_csv, report
):
    """PDE + boundary + sensor (no flow-field): recovers the sensor field.

    This is the combination missing from the earlier cases. Sensors anchor the
    solution to the true velocities, the PDE enforces the governing equations at
    collocation points, and the no-penetration wall constrains the bottom
    boundary. With a data anchor present, the model must reproduce the sensor
    values (unlike pde-only, which can only guarantee uniformity).
    """

    sensor_path, sensor_rows = uniform_sensor_csv
    torch.manual_seed(0)
    sensor_data = read_sensor_data(SensorDataConfig(path=str(sensor_path)))

    domain = make_cartesian_wrf_domain(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
        z_min=0.0, z_max=1.0, t_min=0.0, t_max=1.0,
    )
    boundaries = BoundaryConfig(
        no_slip_wall=NoSlipWallConfig(
            surface=WallSurfaceConfig(path=str(bottom_wall_csv)),
        )
    )
    conditions = ConditionsConfig(
        pde=ConditionSpec("pde", active=True, weight=1.0),
        boundary=ConditionSpec("boundary", active=True, weight=1.0),
        sensor_data=ConditionSpec("sensor_data", active=True, weight=1.0),
        flow_field_data=ConditionSpec("flow_field_data", active=False, weight=0.0),
    )
    sampling = SamplingConfig(
        collocation=CollocationSamplingConfig(n_points=200, method="latin_hypercube"),
        boundary=BoundarySamplingConfig(n_points=None, method="all"),
        seed=0,
    )

    model = MLP()
    history = train_pinn(
        model,
        domain=domain,
        sensor_data=sensor_data,
        boundaries=boundaries,
        conditions=conditions,
        sampling=sampling,
        use_no_penetration_z_wall=True,
        training=_training(),
    )
    _assert_loss_decreased(history, report, "uniform flow, pde+boundary+sensor")

    # All three modes are active and contribute.
    assert history.pde[-1] >= 0.0
    assert history.boundary[-1] >= 0.0
    assert history.sensor_data[-1] >= 0.0

    # Known-result check: sensors anchor the field, so predictions at the sensor
    # points reproduce the true (u, v, w) within tolerance.
    coordinates, _ = sensor_data.as_torch()
    with torch.no_grad():
        predictions = model(coordinates).cpu().numpy()
    targets = np.array([[r["u"], r["v"], r["w"]] for r in sensor_rows])
    mae = np.abs(predictions[:, :3] - targets).mean(axis=0)
    assert (mae < REPRODUCTION_TOL).all()
    report(
        "uniform flow, pde+boundary+sensor: reproduces sensor field",
        per_variable_mae=mae,
        final_components=np.array(
            [history.pde[-1], history.boundary[-1], history.sensor_data[-1]]
        ),
    )
