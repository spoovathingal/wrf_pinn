"""Evaluate a trained model to plain NumPy predictions.

This helper runs a forward pass without tracking gradients and returns the
coordinates, predictions, and targets as NumPy arrays ready for serialization by
``wrf_pinn.evaluation.results_writer``.
"""

from __future__ import annotations

import numpy as np

from wrf_pinn.data.case import Case


def predict_flow_field(model, case: Case, *, device: str = "cpu"):
    """Return coordinates, predictions, and targets as NumPy arrays.

    PyTorch is imported lazily so callers that only serialize existing arrays do
    not pay the import cost. The model is evaluated in ``eval`` mode with
    gradients disabled.
    """

    import torch

    torch_device = torch.device(device)
    model.to(torch_device)

    was_training = model.training
    model.eval()
    try:
        coordinates, targets = case.as_torch(device=torch_device)
        with torch.no_grad():
            predictions = model(coordinates)
    finally:
        model.train(was_training)

    return (
        coordinates.detach().cpu().numpy(),
        predictions.detach().cpu().numpy(),
        targets.detach().cpu().numpy(),
    )


def _as_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
