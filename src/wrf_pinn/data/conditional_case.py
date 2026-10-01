"""The conditional-PINN case: one sub-domain over one time window, holding its
conditioning fields (initial, boundary, terrain) and interior targets. One .npz
per case plus a shared metadata.json sidecar.
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
    """One sub-domain + time window: conditioning fields and targets, as numpy or
    torch arrays."""

    name: str
    initial: object             # phi: (n_ic, n_coord + n_state), coords then state
    boundary: object            # psi: (n_faces, n_times, face_len, n_state)
    boundary_coords: object     # psi coords: (n_faces, n_times, face_len, n_coord)
    terrain: object             # (n_terr, 3): x, y, elevation (static)
    interior: object            # (n_pts, n_coord): query coords, tau > 0
    targets: object             # (n_pts, n_state): matched interior state
    target_mask: object         # (n_pts, n_state): 1 where measured
    times: object               # (n_times,): window snapshot times
    state_vars: tuple[str, ...] = STATE_VARS
    coord_names: tuple[str, ...] = COORD_NAMES
    face_names: tuple[str, ...] = FACE_NAMES

    @property
    def n_interior(self) -> int:
        return self.interior.shape[0]

    def as_torch(self, *, dtype: object | None = None, device: object | None = None):
        """Return a new ConditionalCase with members as torch tensors on device."""
        import torch

        td = dtype if dtype is not None else torch.float32

        def t(a):
            return torch.as_tensor(a, dtype=td, device=device)

        return ConditionalCase(
            name=self.name,
            initial=t(self.initial),
            boundary=t(self.boundary),
            boundary_coords=t(self.boundary_coords),
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
            boundary_coords=np.asarray(self.boundary_coords, dtype=np.float32),
            terrain=np.asarray(self.terrain, dtype=np.float32),
            interior=np.asarray(self.interior, dtype=np.float32),
            targets=np.asarray(self.targets, dtype=np.float32),
            target_mask=np.asarray(self.target_mask, dtype=np.float32),
            times=np.asarray(self.times, dtype=np.float32),
        )
        return path if path.suffix else path.with_suffix(".npz")


def read_conditional_case(npz_path: str | Path) -> ConditionalCase:
    """Load one .npz case; validate members and coords, mask NaN targets."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Conditional case not found: {path}.")

    with np.load(path) as blob:
        needed = ("initial", "boundary", "boundary_coords", "terrain", "interior",
                  "targets", "target_mask", "times")
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
        boundary_coords=members["boundary_coords"],
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
