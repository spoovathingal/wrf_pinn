"""Checkpoint helpers for trained WRF PINN models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from wrf_pinn.training.train_pinn import TrainingHistory


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    history: TrainingHistory | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a trained model checkpoint.

    This is intentionally save-only for now. The caller owns the model
    architecture and training workflow; this helper simply writes the trained
    model weights plus optional history and metadata.

    Parameters
    ----------
    path:
        Output ``.pt`` file path.
    model:
        Trained PyTorch model to serialize via ``state_dict``.
    history:
        Optional training history returned by ``train_pinn``.
    metadata:
        Optional script-level metadata such as data paths, run notes, or config
        summaries.

    Returns
    -------
    Path
        Path to the checkpoint file that was written.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata) if metadata is not None else {},
    }

    if history is not None:
        payload["history"] = _history_to_dict(history)

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _history_to_dict(history: TrainingHistory) -> dict[str, list[float]]:
    """Convert ``TrainingHistory`` to plain Python containers."""

    return {
        "total": list(history.total),
        "pde": list(history.pde),
        "boundary": list(history.boundary),
        "simulation": list(history.simulation),
        "sensor": list(history.sensor),
    }
