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
import torch

from wrf_pinn.config.conditions import ConditionSpec, ConditionsConfig
from wrf_pinn.config.domain import make_cartesian_wrf_domain
from wrf_pinn.config.flow_field_data import FlowFieldDataConfig
from wrf_pinn.config.sampling import CollocationSamplingConfig, SamplingConfig
from wrf_pinn.config.training import OptimizerConfig, TrainingConfig
from wrf_pinn.data.flow_field import read_flow_field
from wrf_pinn.evaluation.predict import predict_flow_field
from wrf_pinn.models.mlp import MLP
from wrf_pinn.training.train_pinn import train_pinn


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


def test_uniform_flow_field_only(uniform_flow_csv, report):
    """flow_field data loss only (pde/boundary/sensor off)."""

    path, _ = uniform_flow_csv
    torch.manual_seed(0)
    flow_data = read_flow_field(FlowFieldDataConfig(path=str(path)))

    conditions = ConditionsConfig(
        pde=ConditionSpec("pde", active=False, weight=0.0),
        boundary=ConditionSpec("boundary", active=False, weight=0.0),
        sensor_data=ConditionSpec("sensor_data", active=False, weight=0.0),
        flow_field_data=ConditionSpec("flow_field_data", active=True, weight=1.0),
    )

    model = MLP()
    history = train_pinn(
        model,
        flow_field_data=flow_data,
        conditions=conditions,
        training=_training(),
    )
    _assert_loss_decreased(history, report, "uniform flow, flow_field only")

    # Known-result check: the trained model reproduces the true constant field.
    _, predictions, targets = predict_flow_field(model, flow_data)
    mae = np.abs(predictions - targets).mean(axis=0)
    assert (mae < REPRODUCTION_TOL).all()
    report(
        "uniform flow, flow_field only: reproduces true field",
        true_state=targets[0],
        predicted_mean=predictions.mean(axis=0),
        per_variable_mae=mae,
    )


def test_uniform_flow_pde_only(uniform_flow_csv, report):
    """PDE residual loss only (flow_field/boundary/sensor off).

    The true uniform field has zero PDE residual, so an untrained model starts
    with nonzero residual loss that should fall as it learns a constant state.
    """

    path, _ = uniform_flow_csv
    torch.manual_seed(0)
    flow_data = read_flow_field(FlowFieldDataConfig(path=str(path)))

    conditions = ConditionsConfig(
        pde=ConditionSpec("pde", active=True, weight=1.0),
        boundary=ConditionSpec("boundary", active=False, weight=0.0),
        sensor_data=ConditionSpec("sensor_data", active=False, weight=0.0),
        flow_field_data=ConditionSpec("flow_field_data", active=False, weight=0.0),
    )
    sampling = SamplingConfig(
        collocation=CollocationSamplingConfig(n_points=200, method="latin_hypercube"),
        seed=0,
    )

    model = MLP()
    history = train_pinn(
        model,
        domain=_domain_from(flow_data),
        conditions=conditions,
        sampling=sampling,
        training=TrainingConfig(
            epochs=600,
            log_every=200,
            optimizer=OptimizerConfig(name="adam", learning_rate=1.0e-3),
        ),
    )
    _assert_loss_decreased(history, report, "uniform flow, pde only")

    # Known-result check: with no data anchor the PDE cannot recover the specific
    # constants, but a zero-residual solution must be spatially uniform. Assert
    # the predicted field has near-zero spatial std per variable (std, not max
    # spread, so a few stubborn collocation points do not make the check flaky).
    _, predictions, _ = predict_flow_field(model, flow_data)
    spatial_std = predictions.std(axis=0)
    assert (spatial_std < UNIFORMITY_STD_TOL).all()
    report(
        "uniform flow, pde only: predicted field is spatially uniform",
        predicted_mean=predictions.mean(axis=0),
        per_variable_std=spatial_std,
    )
