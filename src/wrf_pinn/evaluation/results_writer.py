"""Write training results in plain, script-readable formats.

This module produces an organized set of output files that a separate plotting
script can consume. It intentionally depends only on the standard library and
NumPy so it stays decoupled from the training loop and from PyTorch.

Three artifacts are written into a run directory:

``loss_history.csv``
    One row per recorded epoch with the total, PDE, and data loss values.
    Enables a training-curve plot.

``predictions.csv``
    One row per flow-field point with the coordinate columns followed by
    predicted and true values for each state variable (``u_pred``/``u_true``,
    ...). Enables field and scatter comparison plots.

``run_metadata.json``
    A self-describing header: the configuration used, dataset shape, final loss
    values, and a manifest of the files above so a plotter can locate them.

The functions here only serialize already-computed values. Model evaluation,
loss assembly, and gradient work happen elsewhere.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOSS_HISTORY_FILENAME = "loss_history.csv"
PREDICTIONS_FILENAME = "predictions.csv"
METADATA_FILENAME = "run_metadata.json"


@dataclass(frozen=True)
class ResultsManifest:
    """Paths to the files written for one training run."""

    run_dir: Path
    loss_history: Path
    predictions: Path
    metadata: Path


def write_loss_history(
    path: str | Path,
    *,
    total: Sequence[float],
    pde: Sequence[float],
    data: Sequence[float],
) -> Path:
    """Write per-epoch loss values as CSV with a header row.

    The three sequences must have equal length. Epochs are numbered from 1.
    """

    if not (len(total) == len(pde) == len(data)):
        raise ValueError(
            "loss sequences must have equal length; got "
            f"total={len(total)}, pde={len(pde)}, data={len(data)}."
        )

    csv_path = Path(path)
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "total", "pde", "data"))
        for epoch, (total_value, pde_value, data_value) in enumerate(
            zip(total, pde, data), start=1
        ):
            writer.writerow((epoch, total_value, pde_value, data_value))

    return csv_path


def write_predictions(
    path: str | Path,
    *,
    coordinates: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    coordinate_names: Sequence[str],
    target_names: Sequence[str],
) -> Path:
    """Write coordinates with predicted and true state values as CSV.

    Columns are the coordinate names followed by ``<name>_pred`` and
    ``<name>_true`` pairs for each state variable.
    """

    coordinates = np.asarray(coordinates)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    _validate_prediction_shapes(
        coordinates=coordinates,
        predictions=predictions,
        targets=targets,
        coordinate_names=coordinate_names,
        target_names=target_names,
    )

    header = list(coordinate_names)
    for name in target_names:
        header.append(f"{name}_pred")
        header.append(f"{name}_true")

    csv_path = Path(path)
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row_index in range(coordinates.shape[0]):
            row: list[float] = list(coordinates[row_index])
            for column in range(len(target_names)):
                row.append(float(predictions[row_index, column]))
                row.append(float(targets[row_index, column]))
            writer.writerow(row)

    return csv_path


def write_metadata(path: str | Path, metadata: dict) -> Path:
    """Write run metadata as indented JSON."""

    json_path = Path(path)
    with json_path.open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    return json_path


def write_training_results(
    run_dir: str | Path,
    *,
    total: Sequence[float],
    pde: Sequence[float],
    data: Sequence[float],
    coordinates: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    coordinate_names: Sequence[str],
    target_names: Sequence[str],
    metadata: dict | None = None,
) -> ResultsManifest:
    """Write loss history, predictions, and metadata into a run directory.

    Returns a manifest with the paths written. Any ``metadata`` supplied by the
    caller is augmented with the dataset shape, final loss values, and the
    output filenames so the JSON header fully describes the run.
    """

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)

    loss_path = write_loss_history(
        directory / LOSS_HISTORY_FILENAME,
        total=total,
        pde=pde,
        data=data,
    )
    predictions_path = write_predictions(
        directory / PREDICTIONS_FILENAME,
        coordinates=coordinates,
        predictions=predictions,
        targets=targets,
        coordinate_names=coordinate_names,
        target_names=target_names,
    )

    full_metadata = dict(metadata) if metadata is not None else {}
    full_metadata.update(
        {
            "n_points": int(np.asarray(coordinates).shape[0]),
            "coordinate_names": list(coordinate_names),
            "target_names": list(target_names),
            "epochs_recorded": len(total),
            "final_loss": {
                "total": float(total[-1]) if total else None,
                "pde": float(pde[-1]) if pde else None,
                "data": float(data[-1]) if data else None,
            },
            "files": {
                "loss_history": LOSS_HISTORY_FILENAME,
                "predictions": PREDICTIONS_FILENAME,
                "metadata": METADATA_FILENAME,
            },
        }
    )
    metadata_path = write_metadata(directory / METADATA_FILENAME, full_metadata)

    return ResultsManifest(
        run_dir=directory,
        loss_history=loss_path,
        predictions=predictions_path,
        metadata=metadata_path,
    )


def _validate_prediction_shapes(
    *,
    coordinates: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    coordinate_names: Sequence[str],
    target_names: Sequence[str],
) -> None:
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape (n_points, n_coordinates).")

    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have the same shape; got "
            f"{predictions.shape} and {targets.shape}."
        )

    if predictions.ndim != 2 or predictions.shape[0] != coordinates.shape[0]:
        raise ValueError(
            "predictions must have shape (n_points, n_variables) matching "
            "coordinates row count."
        )

    if coordinates.shape[1] != len(coordinate_names):
        raise ValueError(
            "coordinate_names length must match coordinate columns; got "
            f"{len(coordinate_names)} names for {coordinates.shape[1]} columns."
        )

    if predictions.shape[1] != len(target_names):
        raise ValueError(
            "target_names length must match prediction columns; got "
            f"{len(target_names)} names for {predictions.shape[1]} columns."
        )
