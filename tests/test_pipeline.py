"""End-to-end reading pipeline: .npy case -> tensors -> model.

Walks the path a training run takes to get data into the model, with a value
checkpoint before the model so we know it is fed correct inputs:

    .npy case  ->  read_case  ->  as_torch tensors  ->  MLP(coords)
"""

from __future__ import annotations

import numpy as np
import torch

from wrf_pinn.data.case import read_case, SRC_SENSOR, SRC_SIM
from wrf_pinn.models.mlp import MLP


def test_nan_targets_read_without_raising(nan_target_case, report):
    """A widened case with NaN theta/p' reads; the mask marks measured entries."""

    npy, meta = nan_target_case
    case = read_case(npy, meta, targets=("u", "v", "w", "theta", "p_prime"))

    assert np.isfinite(case.targets).all()          # NaN was zero-filled
    sim = case.source == SRC_SIM
    sensor = case.source == SRC_SENSOR
    assert case.target_mask[sim, 3:5].all()          # sim measures theta/p'
    assert not case.target_mask[sensor, 3:5].any()   # sensors do not
    assert case.target_mask[:, :3].all()             # everyone has u,v,w

    report("masked widened targets",
           mask_sim=case.target_mask[sim][0], mask_sensor=case.target_mask[sensor][0])


def test_masked_loss_ignores_unmeasured_targets(nan_target_case):
    """Zeroed NaN entries never reach the loss; measured entries drive it."""

    from wrf_pinn.config.conditions import ConditionSpec
    from wrf_pinn.training.losses import _reduce_tensor

    npy, meta = nan_target_case
    case = read_case(npy, meta, targets=("u", "v", "w", "theta", "p_prime"))
    _, targets, mask = case.as_torch()

    # a constant-1 error everywhere, but unmeasured entries are masked to 0
    error = (targets * 0.0 + 1.0) * mask
    spec = ConditionSpec("sensor", active=True, weight=1.0, reduction="mean")

    # masked mean = 1.0 (every measured entry has squared error 1); a plain mean
    # would be < 1 because it would divide by the unmeasured zeros too.
    assert float(_reduce_tensor(error, spec, mask)) == 1.0
    assert float(_reduce_tensor(error, spec)) < 1.0


def test_pipeline_pre_model_values(uniform_case, report):
    """After reading, the tensors fed to the model hold the correct values."""

    case = uniform_case
    coordinates, targets, _ = case.as_torch()

    assert coordinates.shape == (case.n_points, 4)   # x, y, z, t
    assert targets.shape == (case.n_points, 3)        # u, v, w
    assert coordinates.dtype == torch.float32
    assert targets.dtype == torch.float32

    # values match what the case holds
    np.testing.assert_allclose(coordinates.numpy(), case.coordinates)
    np.testing.assert_allclose(targets.numpy(), case.targets)

    report(f"pre-model tensors ({case.n_points} points) fed to model",
           coordinates=coordinates.numpy(), targets=targets.numpy())


def test_pipeline_model_consumes_inputs(uniform_case, report):
    """The model accepts the read coordinates and returns a finite state."""

    coordinates, _, _ = uniform_case.as_torch()
    model = MLP()
    with torch.no_grad():
        predictions = model(coordinates)

    assert predictions.shape[0] == uniform_case.n_points
    assert torch.isfinite(predictions).all()

    report(f"post-pipeline model output ({uniform_case.n_points} points)",
           prediction_shape=np.array(predictions.shape),
           all_finite=np.array([bool(torch.isfinite(predictions).all())]))
