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
STATE_VARS: tuple[str, ...] = ("u", "v", "w", "theta", "p_prime", "q_v", "e_sgs")
#: Interior sample coordinate columns.
COORD_NAMES: tuple[str, ...] = ("x", "y", "z", "t")
#: The four open side faces of a sub-domain, in a fixed order.
FACE_NAMES: tuple[str, ...] = ("west", "east", "south", "north")


@dataclass(frozen=True)
class ConditionalCase:
    """One sub-domain + time window: conditioning fields and targets, as numpy or
    torch arrays."""

    name: str
    z: object                   # (k,): concatenated POD coeffs; the conditioning latent
    initial_coords: object      # (n_ic, n_coord): where to enforce the initial field
    initial: object             # (n_state, nz, nx*ny): decoded initial field
    boundary: object            # (n_state, n_faces, n_times, face_len): decoded boundary
    boundary_coords: object     # psi coords: (n_faces, n_times, face_len, n_coord)
    surface: object             # (n_times, n_face_pts, 5): x,y,t,fricVel,htFlux (bottom)
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
            z=t(self.z),
            initial_coords=t(self.initial_coords),
            initial=t(self.initial),
            boundary=t(self.boundary),
            boundary_coords=t(self.boundary_coords),
            surface=t(self.surface),
            terrain=t(self.terrain),
            interior=t(self.interior),
            targets=t(self.targets),
            target_mask=t(self.target_mask),
            times=t(self.times),
            state_vars=self.state_vars,
            coord_names=self.coord_names,
            face_names=self.face_names,
        )


#: Non-POD members written by the generator and read back directly. The initial
#: and boundary fields are reconstructed from their per-subdomain POD pieces.
_CASE_MEMBERS: tuple[str, ...] = (
    "initial_coords", "boundary_coords", "surface", "terrain", "interior",
    "targets", "times",
)


def read_conditional_case(npz_path: str | Path) -> ConditionalCase:
    """Load one POD-encoded .npz case; decode initial/boundary, mask NaN targets."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Conditional case not found: {path}.")

    with np.load(path) as blob:
        missing = [k for k in _CASE_MEMBERS if k not in blob.files]
        if missing:
            raise ValueError(f"Case {path} missing members: {missing}.")
        members = {k: blob[k].astype(np.float32) for k in _CASE_MEMBERS}
        bnd = decode_boundary(blob)
        ini = decode_initial(blob)
        members["z"] = latent_vector(blob)

    members["boundary"] = np.stack([bnd[v] for v in STATE_VARS], axis=0).astype(np.float32)
    members["initial"] = np.stack([ini[v] for v in STATE_VARS], axis=0).astype(np.float32)

    if not np.isfinite(members["interior"]).all():
        raise ValueError(f"Non-finite interior coordinates in {path}.")

    mask = np.isfinite(members["targets"]).astype(np.float32)
    members["targets"] = np.where(mask > 0.0, members["targets"], 0.0).astype(np.float32)
    members["target_mask"] = mask

    return ConditionalCase(name=path.stem, **members)


def decode_boundary(blob) -> dict:
    """Reconstruct the boundary state from a case's per-face POD pieces.

    Returns {var: (n_faces, n_times, face_len)} in normalized units. blob is an
    np.load handle (or dict) of a case .npz written by the preprocessor."""
    out = {}
    face = 0
    nf = sum(1 for k in blob.files if k.startswith(f"bnd_modes_{STATE_VARS[0]}_"))
    for var in STATE_VARS:
        faces = []
        for f in range(nf):
            mean = blob[f"bnd_mean_{var}_{f}"]       # (face_len,)
            modes = blob[f"bnd_modes_{var}_{f}"]     # (face_len, k)
            coeffs = blob[f"bnd_coeffs_{var}_{f}"]   # (n_times, k)
            faces.append(mean + coeffs @ modes.T)    # (n_times, face_len)
        out[var] = np.stack(faces, axis=0)           # (n_faces, n_times, face_len)
    return out


def decode_initial(blob) -> dict:
    """Reconstruct the initial state from a case's spatial POD pieces.

    Returns {var: (nz, nx*ny)} in normalized units (z-levels x horizontal)."""
    out = {}
    for var in STATE_VARS:
        mean = blob[f"ini_mean_{var}"]               # (nx*ny,)
        modes = blob[f"ini_modes_{var}"]             # (nx*ny, k)
        coeffs = blob[f"ini_coeffs_{var}"]           # (nz, k)
        out[var] = mean + coeffs @ modes.T           # (nz, nx*ny)
    return out


def latent_vector(blob) -> np.ndarray:
    """Concatenate the case's POD coeffs into the conditioning latent z. Order is
    fixed (boundary faces then initial, by STATE_VARS) so z is consistent across cases."""
    nf = sum(1 for k in blob.files if k.startswith(f"bnd_modes_{STATE_VARS[0]}_"))
    parts = []
    for var in STATE_VARS:
        for f in range(nf):
            parts.append(blob[f"bnd_coeffs_{var}_{f}"].reshape(-1))
    for var in STATE_VARS:
        parts.append(blob[f"ini_coeffs_{var}"].reshape(-1))
    return np.concatenate(parts).astype(np.float32)


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
