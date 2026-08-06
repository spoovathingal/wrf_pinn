"""End-to-end reading pipeline: generate -> read -> tensors -> model.

This walks the full path a training run takes to get data into the model, with a
value checkpoint *before* the model so we know the model is fed correct inputs:

    known CSV  ->  read_flow_field  ->  as_torch tensors  ->  MLP(coords)

The CSV is generated inside the test (self-contained, mirroring the uniform-flow
data) so nothing depends on the out-of-repo generator script and nothing
persists: the file lives under pytest ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import torch

from wrf_pinn.config.flow_field_data import FlowFieldDataConfig
from wrf_pinn.data.flow_field import read_flow_field
from wrf_pinn.models.mlp import MLP


def test_pipeline_pre_model_values(flow_csv, report):
    """After reading, the tensors fed to the model hold the correct values.

    This is the "before feed model should have correct values" checkpoint: the
    torch tensors match the known written data in shape, dtype, and value.
    """

    path, rows = flow_csv
    data = read_flow_field(FlowFieldDataConfig(path=str(path)))
    coordinates, targets = data.as_torch()

    expected_coordinates = np.array(
        [[r["x"], r["y"], r["z"], r["t"]] for r in rows], dtype=np.float32
    )
    expected_targets = np.array(
        [[r["u"], r["v"], r["w"], r["theta"], r["p_prime"]] for r in rows], dtype=np.float32
    )

    # shape and dtype are what the model expects
    assert coordinates.shape == (len(rows), 4)
    assert targets.shape == (len(rows), 5)
    assert coordinates.dtype == torch.float32
    assert targets.dtype == torch.float32

    # values survived the CSV -> numpy -> torch trip unchanged
    np.testing.assert_allclose(coordinates.numpy(), expected_coordinates)
    np.testing.assert_allclose(targets.numpy(), expected_targets)

    report(
        f"pre-model tensors ({len(rows)} points) fed to model",
        coordinates=coordinates.numpy(),
        targets=targets.numpy(),
    )


def test_pipeline_model_consumes_inputs(flow_csv, report):
    """The model accepts the read coordinates and returns a well-formed state.

    This is the "post pipeline" checkpoint: a forward pass runs on the real read
    inputs and produces finite outputs of the expected (n, 4) shape.
    """

    path, rows = flow_csv
    data = read_flow_field(FlowFieldDataConfig(path=str(path)))
    coordinates, _ = data.as_torch()

    model = MLP()
    with torch.no_grad():
        predictions = model(coordinates)

    assert predictions.shape == (len(rows), 5)  # (u, v, w, theta, p_prime)
    assert torch.isfinite(predictions).all()

    report(
        f"post-pipeline model output ({len(rows)} points)",
        prediction_shape=np.array(predictions.shape),
        all_finite=np.array([bool(torch.isfinite(predictions).all())]),
    )
