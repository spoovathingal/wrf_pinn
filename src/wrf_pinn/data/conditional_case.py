"""The conditional-PINN case: the data contract the pre-processor writes and the
training loop reads.

A case is one sub-domain over one time window, uniquely identified by (space, time).
Unlike the flat point-cloud ``Case`` in ``case.py``, this names what it holds, so it
is agnostic to the source (FastEddy, LASSO, WRF) and carries the conditioning
fields the model is conditioned on:

    initial   phi   the full interior state at tau=0 (sensors folded in here)
    boundary  psi   the time history of the sub-domain's open faces
    terrain         the fixed surface geometry for the sub-domain (static)
    interior        supervised interior samples (coordinates + targets) at tau>0

The model encodes (initial, boundary, terrain) -> z and the decoder predicts the
interior; the losses compare the decoded prediction to these members. There is no
integer source tag and no HRRR anchor: a case names its parts.

Serialized as one ``.npz`` per case (the members have different shapes), plus a
shared ``metadata.json`` sidecar (normalization recipe, variable order). Arrays are
numpy on the host read path and are moved to torch tensors by ``as_torch``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Variables carried in the conditioning fields and predicted by the model.
STATE_VARS: tuple[str, ...] = ("u", "v", "w", "theta")
#: Interior sample coordinate columns.
COORD_NAMES: tuple[str, ...] = ("x", "y", "z", "t")
#: The four open side faces of a sub-domain, in a fixed order.
FACE_NAMES: tuple[str, ...] = ("west", "east", "south", "north")


@dataclass(frozen=True)
class ConditionalCase:
    """One sub-domain + time window, with its conditioning fields and targets.

    Shapes (host numpy, float32 unless noted):
      initial    (n_pts_ic, len(COORD_NAMES) + len(STATE_VARS))
                 interior points at tau=0: coordinates then state. phi.
      boundary   (n_faces, n_times, face_len, len(STATE_VARS))
                 per-face time history of the open faces. psi.
      terrain    (n_terr, 3)  surface coordinates (x, y, elevation). static.
      interior   (n_pts, len(COORD_NAMES))         query coordinates (tau > 0)
      targets    (n_pts, len(STATE_VARS))          matched interior state
      target_mask(n_pts, len(STATE_VARS))          1 where target measured, else 0

    ``name`` is the case id; ``times`` are the window's snapshot times (len n_times).
    """

    name: str
    initial: object       # np.ndarray | torch.Tensor
    boundary: object
    terrain: object
    interior: object
    targets: object
    target_mask: object
    times: object
    state_vars: tuple[str, ...] = STATE_VARS
    coord_names: tuple[str, ...] = COORD_NAMES
    face_names: tuple[str, ...] = FACE_NAMES

    @property
    def n_interior(self) -> int:
        return self.interior.shape[0]

    def as_torch(self, *, dtype: object | None = None, device: object | None = None):
        """Return the case members as torch tensors on ``device``.

        Returns a new ``ConditionalCase`` holding tensors; names/times are kept.
        """
        import torch

        td = dtype if dtype is not None else torch.float32

        def t(a):
            return torch.as_tensor(a, dtype=td, device=device)

        return ConditionalCase(
            name=self.name,
            initial=t(self.initial),
            boundary=t(self.boundary),
            terrain=t(self.terrain),
            interior=t(self.interior),
            targets=t(self.targets),
            target_mask=t(self.target_mask),
            times=t(self.times),
            state_vars=self.state_vars,
            coord_names=self.coord_names,
            face_names=self.face_names,
        )

    def save(self, path: str | Path) -> Path:
        """Write this case to one ``.npz`` (members by name). Returns the path."""
        path = Path(path)
        np.savez(
            path,
            initial=np.asarray(self.initial, dtype=np.float32),
            boundary=np.asarray(self.boundary, dtype=np.float32),
            terrain=np.asarray(self.terrain, dtype=np.float32),
            interior=np.asarray(self.interior, dtype=np.float32),
            targets=np.asarray(self.targets, dtype=np.float32),
            target_mask=np.asarray(self.target_mask, dtype=np.float32),
            times=np.asarray(self.times, dtype=np.float32),
        )
        return path if path.suffix else path.with_suffix(".npz")


def read_conditional_case(npz_path: str | Path) -> ConditionalCase:
    """Load one ``.npz`` case written by ``ConditionalCase.save`` or the pre-processor.

    Validates member presence and that coordinates are finite; optional targets may
    be NaN (recorded in ``target_mask`` and zero-filled), matching the flat reader.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Conditional case not found: {path}.")

    with np.load(path) as blob:
        needed = ("initial", "boundary", "terrain", "interior", "targets",
                  "target_mask", "times")
        missing = [k for k in needed if k not in blob.files]
        if missing:
            raise ValueError(f"Case {path} missing members: {missing}.")
        members = {k: blob[k].astype(np.float32) for k in needed}

    interior = members["interior"]
    if not np.isfinite(interior).all():
        raise ValueError(f"Non-finite interior coordinates in {path}.")

    # optional targets may be NaN: mask the measured entries, zero-fill the rest
    targets = members["targets"]
    mask = np.isfinite(targets).astype(np.float32)
    targets = np.where(mask > 0.0, targets, 0.0).astype(np.float32)

    return ConditionalCase(
        name=path.stem,
        initial=members["initial"],
        boundary=members["boundary"],
        terrain=members["terrain"],
        interior=interior,
        targets=targets,
        target_mask=mask,
        times=members["times"],
    )


@dataclass(frozen=True)
class ConditionalMetadata:
    """Shared sidecar: variable order and the normalization recipe (if any)."""

    state_vars: tuple[str, ...]
    coord_names: tuple[str, ...]
    normalization: dict

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalMetadata":
        raw = json.loads(Path(path).read_text())
        schema = raw.get("schema", {})
        return cls(
            state_vars=tuple(schema.get("state_vars", STATE_VARS)),
            coord_names=tuple(schema.get("coord_names", COORD_NAMES)),
            normalization=raw.get("normalization", {}),
        )
