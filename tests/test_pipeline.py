"""End-to-end reading pipeline: .npy case -> tensors -> model.

Walks the path a training run takes to get data into the model, with a value
checkpoint before the model so we know it is fed correct inputs:

    .npy case  ->  read_case  ->  as_torch tensors  ->  MLP(coords)
"""

from __future__ import annotations

import numpy as np
import torch

from wrf_pinn.models.mlp import MLP


def test_pipeline_pre_model_values(uniform_case, report):
    """After reading, the tensors fed to the model hold the correct values."""

    case = uniform_case
    coordinates, targets = case.as_torch()

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

    coordinates, _ = uniform_case.as_torch()
    model = MLP()
    with torch.no_grad():
        predictions = model(coordinates)

    assert predictions.shape[0] == uniform_case.n_points
    assert torch.isfinite(predictions).all()

    report(f"post-pipeline model output ({uniform_case.n_points} points)",
           prediction_shape=np.array(predictions.shape),
           all_finite=np.array([bool(torch.isfinite(predictions).all())]))
