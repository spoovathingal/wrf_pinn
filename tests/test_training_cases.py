"""Training-case regression tests on ``.npy`` cases.

Short trainings on constant uniform-flow cases with different active loss modes.
Each asserts the pipeline (1) runs, (2) reduces the loss, and where a data anchor
is present (3) reproduces the known constant (u, v, w). Boundary is off (no-slip
does not satisfy uniform flow).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wrf_pinn.config.conditions import ConditionSpec, ConditionsConfig
from wrf_pinn.config.domain import make_cartesian_wrf_domain
from wrf_pinn.config.sampling import CollocationSamplingConfig, SamplingConfig
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING
from wrf_pinn.config.training import OptimizerConfig, TrainingConfig
from wrf_pinn.data.case import SRC_INLET, SRC_SIM, SRC_SENSOR
from wrf_pinn.evaluation.predict import predict_flow_field
from wrf_pinn.models.mlp import MLP
from wrf_pinn.training.train_pinn import train_pinn, TrainingSetup

REPRODUCTION_TOL = 0.05
UNIFORMITY_STD_TOL = 0.15


def _training(epochs=300, log_every=100):
    return TrainingConfig(
        epochs=epochs, log_every=log_every,
        optimizer=OptimizerConfig(name="adam", learning_rate=1.0e-3),
    )


def _off(name):
    return ConditionSpec(name, active=False, weight=0.0)


def _data_only(active_source: str) -> ConditionsConfig:
    """Conditions with pde/boundary off and only ``active_source`` on."""

    data = {n: _off(n) for n in ("inlet", "simulation", "sensor")}
    data[active_source] = ConditionSpec(active_source, active=True, weight=1.0)
    return ConditionsConfig(pde=_off("pde"), boundary=_off("boundary"), **data)


def _pde_only() -> ConditionsConfig:
    return ConditionsConfig(
        pde=ConditionSpec("pde", active=True, weight=1.0),
        boundary=_off("boundary"),
        inlet=_off("inlet"), simulation=_off("simulation"), sensor=_off("sensor"),
    )


def _domain_from(case):
    lo = case.coordinates.min(axis=0)
    hi = case.coordinates.max(axis=0)
    return make_cartesian_wrf_domain(
        x_min=float(lo[0]), x_max=float(hi[0]),
        y_min=float(lo[1]), y_max=float(hi[1]),
        z_min=float(lo[2]), z_max=float(hi[2]),
        t_min=float(lo[3]), t_max=float(hi[3]),
    )


def _assert_loss_decreased(history, report, label):
    assert len(history.total) > 0
    assert history.total[-1] < history.total[0]
    report(f"{label}: loss decreased",
           initial_loss=history.total[0], final_loss=history.total[-1])


# --- data-driven reproduction ------------------------------------------------
FLOW_CASES = [
    ("baseline", (5.0, 2.0, 0.0), (0.0, 1.0)),
    ("symmetric_domain", (5.0, 2.0, 0.0), (-1.0, 1.0)),
    ("small_magnitude", (0.1, 0.05, 0.0), (0.0, 1.0)),
    ("mixed_sign", (2.0, -0.8, 0.3), (0.0, 1.0)),
]


@pytest.mark.parametrize("label, state, coord_range", FLOW_CASES,
                         ids=[c[0] for c in FLOW_CASES])
def test_data_only_reproduces_uniform_field(uniform_case_factory, report,
                                            label, state, coord_range):
    """A simulation-tagged case reproduces its constant (u,v,w)."""

    torch.manual_seed(0)
    case = uniform_case_factory(state=state, coordinate_range=coord_range,
                                source=SRC_SIM)
    model = MLP()
    history = train_pinn(model, TrainingSetup(
        case=case, conditions=_data_only("simulation"), training=_training()))
    _assert_loss_decreased(history, report, f"data-only [{label}]")

    _, predictions, targets = predict_flow_field(model, case)
    mae = np.abs(predictions[:, : targets.shape[1]] - targets).mean(axis=0)
    assert (mae < REPRODUCTION_TOL).all()
    report(f"data-only [{label}]: reproduces field",
           true_state=targets[0], per_variable_mae=mae)


def test_each_source_tag_trains(uniform_case_factory, report):
    """The pipeline runs with each of the three source tags as the data term."""

    for source, name in ((SRC_INLET, "inlet"), (SRC_SIM, "simulation"),
                         (SRC_SENSOR, "sensor")):
        torch.manual_seed(0)
        case = uniform_case_factory(source=source)
        model = MLP()
        history = train_pinn(model, TrainingSetup(
            case=case, conditions=_data_only(name), training=_training(epochs=50)))
        _assert_loss_decreased(history, report, f"source={name}")


def test_multi_source_weighted_terms(multi_source_case, report):
    """A case with all three tags produces three distinct, active data terms."""

    torch.manual_seed(0)
    conditions = ConditionsConfig(
        pde=_off("pde"), boundary=_off("boundary"),
        inlet=ConditionSpec("inlet", active=True, weight=1.0),
        simulation=ConditionSpec("simulation", active=True, weight=1.0),
        sensor=ConditionSpec("sensor", active=True, weight=1.0),
    )
    model = MLP()
    history = train_pinn(model, TrainingSetup(
        case=multi_source_case, conditions=conditions, training=_training(epochs=50)))
    _assert_loss_decreased(history, report, "multi-source")
    for name in ("inlet", "simulation", "sensor"):
        assert getattr(history, name)[-1] >= 0.0
    report("multi-source: three data terms",
           final=np.array([history.inlet[-1], history.simulation[-1], history.sensor[-1]]))


# --- pde-only uniformity -----------------------------------------------------
@pytest.mark.parametrize("collocation_points", [64, 200, 512])
def test_pde_only_runs_and_reduces_loss(uniform_case, report, collocation_points):
    """pde-only path runs end to end and drives the residual loss down.

    (The old 'uniform flow -> uniform field' check no longer holds: the NBL
    governing equations are stratified/hydrostatic, so a uniform field is not a
    zero-residual solution. A proper NBL known-result check is future work.)
    """

    torch.manual_seed(0)
    sampling = SamplingConfig(
        collocation=CollocationSamplingConfig(
            n_points=collocation_points, method="latin_hypercube"),
        seed=0,
    )
    model = MLP()
    history = train_pinn(model, TrainingSetup(
        domain=_domain_from(uniform_case), conditions=_pde_only(),
        sampling=sampling, scaling=DEFAULT_RESIDUAL_SCALING,
        training=_training(epochs=600, log_every=200)))
    _assert_loss_decreased(history, report, f"pde-only [{collocation_points} pts]")
