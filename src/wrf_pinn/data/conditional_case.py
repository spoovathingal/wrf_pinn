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
    initial_coords: object      # (n_ic, n_coord): where to enforce the initial field
    z_initial: object           # (k_i,): POD coeffs of the initial state
    z_boundary: object          # (k_b,): POD coeffs of the boundary state
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
            initial_coords=t(self.initial_coords),
            z_initial=t(self.z_initial),
            z_boundary=t(self.z_boundary),
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


#: Case members written by the generator and read back here.
_CASE_MEMBERS: tuple[str, ...] = (
    "initial_coords", "z_initial", "z_boundary", "boundary_coords", "terrain",
    "interior", "targets", "target_mask", "times",
)


def read_conditional_case(npz_path: str | Path) -> ConditionalCase:
    """Load one POD-encoded .npz case; validate members, mask NaN targets."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Conditional case not found: {path}.")

    with np.load(path) as blob:
        missing = [k for k in _CASE_MEMBERS if k not in blob.files]
        if missing:
            raise ValueError(f"Case {path} missing members: {missing}.")
        members = {k: blob[k].astype(np.float32) for k in _CASE_MEMBERS}

    if not np.isfinite(members["interior"]).all():
        raise ValueError(f"Non-finite interior coordinates in {path}.")

    mask = np.isfinite(members["targets"]).astype(np.float32)
    members["targets"] = np.where(mask > 0.0, members["targets"], 0.0).astype(np.float32)
    members["target_mask"] = mask

    return ConditionalCase(name=path.stem, **members)


@dataclass
class PODBasis:
    """Per-variable POD modes for decoding z back to physical state fields.

    means/modes are dicts keyed by state variable; each field (initial, boundary)
    has its own set. z for a field is the per-variable coeffs concatenated in
    STATE_VARS order.
    """

    initial: dict               # {"means": {var: (n,)}, "modes": {var: (n, k)}}
    boundary: dict

    @classmethod
    def load(cls, path: str | Path) -> "PODBasis":
        with np.load(Path(path)) as b:
            fields = {}
            for field in ("initial", "boundary"):
                fields[field] = {
                    "means": {v: b[f"{field}_mean_{v}"] for v in STATE_VARS},
                    "modes": {v: b[f"{field}_modes_{v}"] for v in STATE_VARS},
                }
        return cls(initial=fields["initial"], boundary=fields["boundary"])

    def as_torch(self, *, device=None):
        import torch
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        conv = lambda f: {"means": {v: t(f["means"][v]) for v in STATE_VARS},
                          "modes": {v: t(f["modes"][v]) for v in STATE_VARS}}
        return PODBasis(initial=conv(self.initial), boundary=conv(self.boundary))

    @staticmethod
    def _decode(z, basis):
        """Split z per variable, decode each column, stack to (n_points, n_state)."""
        import torch
        cols, i = [], 0
        for var in STATE_VARS:
            V = basis["modes"][var]
            k = V.shape[1]
            cols.append(basis["means"][var] + V @ z[i:i + k])
            i += k
        return torch.stack(cols, dim=1)

    def decode_initial(self, z):
        return self._decode(z, self.initial)

    def decode_boundary(self, z):
        return self._decode(z, self.boundary)


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
