"""The conditional-PINN case: one sub-domain over one time window, holding its
conditioning fields (initial, boundary, terrain) and interior targets. One .npz
per case plus a shared metadata.json sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import torch
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
    initial_raw: object = None  # (n_state, nz, nx*ny): exact initial field (--keep-raw cases)
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
            initial_raw=None if self.initial_raw is None else t(self.initial_raw),
            state_vars=self.state_vars,
            coord_names=self.coord_names,
            face_names=self.face_names,
        )


#: Non-POD members written by the generator and read back directly. The initial
#: and boundary fields are reconstructed from their per-subdomain POD pieces;
#: ``interior`` is rebuilt (``interior_from_grid``) when a case omits it.
_CASE_MEMBERS: tuple[str, ...] = (
    "initial_coords", "boundary_coords", "surface", "terrain",
    "targets", "times",
)


def interior_from_grid(initial_coords: np.ndarray, boundary_coords: np.ndarray) -> np.ndarray:
    """Interior query coords: the initial grid at each lead time (lead-major, initial
    point order), the leads' times taken from the boundary coords' snapshot times."""
    t = boundary_coords[0, 1:, 0, 3]
    xyz = np.tile(initial_coords[:, :3], (len(t), 1))
    return np.concatenate([xyz, np.repeat(t, len(initial_coords))[:, None]], 1).astype(np.float32)


def read_conditional_case(npz_path: str | Path, latent: str = "pod") -> ConditionalCase:
    """Load one POD-encoded .npz case; decode initial/boundary, mask NaN targets.
    latent picks z: 'pod' (POD coeffs) or 'raw' (the full raw fields, see latent_vector)."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Conditional case not found: {path}.")

    with np.load(path) as blob:
        missing = [k for k in _CASE_MEMBERS if k not in blob.files]
        if missing:
            raise ValueError(f"Case {path} missing members: {missing}.")
        members = {k: blob[k].astype(np.float32) for k in _CASE_MEMBERS}
        members["interior"] = (blob["interior"].astype(np.float32) if "interior" in blob.files
                               else interior_from_grid(members["initial_coords"],
                                                       members["boundary_coords"]))
        bnd = decode_boundary(blob)
        ini = decode_initial(blob)
        members["z"] = latent_vector(blob, latent)
        if "ini_raw" in blob.files:
            members["initial_raw"] = blob["ini_raw"].astype(np.float32)

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


def terrain_grid(terrain) -> np.ndarray:
    """Terrain rows (x, y, elevation) -> elevations (nx*ny,), x slow like ini_raw."""
    t = np.asarray(terrain, dtype=np.float64)
    return t[np.lexsort((t[:, 1], t[:, 0])), 2]


def latent_vector(blob, latent: str = "pod") -> np.ndarray:
    """Concatenate the case's POD coeffs into the conditioning latent z. Order is
    fixed (boundary faces then initial, by STATE_VARS) so z is consistent across cases.
    latent='raw': the normalized raw fields instead, bnd_raw (n_faces, n_times,
    face_len, n_state) then ini_raw (n_state, nz, nx*ny), flattened (needs --keep-raw).
    latent='raw_terrain': the same, then the ground height (m) on the grid (terrain_grid)."""
    if latent in ("raw", "raw_terrain"):
        parts = [blob["bnd_raw"].reshape(-1), blob["ini_raw"].reshape(-1)]
        if latent == "raw_terrain":
            parts.append(terrain_grid(blob["terrain"]))
        return np.concatenate(parts).astype(np.float32)
    if latent != "pod":
        raise ValueError(f"latent must be 'pod', 'raw' or 'raw_terrain', got {latent!r}.")
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

def prepare_terrain(terrain):
    """Build sorted terrain axes and heights once on CPU.

    Input columns are physical x, y, elevation in metres.
    Returns x_axis, y_axis, height[x_index, y_index].
    """
    terrain = torch.as_tensor(terrain, device="cpu").detach()
    if not terrain.is_floating_point():
        terrain = terrain.float()

    if (terrain.ndim != 2 or terrain.shape[1] != 3 or not torch.isfinite(terrain).all()):
        raise ValueError("terrain must contain finite (x, y, elevation) rows")

    x_axis = torch.unique(terrain[:, 0], sorted=True)
    y_axis = torch.unique(terrain[:, 1], sorted=True)
    nx, ny = x_axis.numel(), y_axis.numel()

    if nx < 2 or ny < 2:
        raise ValueError("terrain requires at least two x and y values")

    ix = torch.searchsorted(x_axis, terrain[:, 0].contiguous())
    iy = torch.searchsorted(y_axis, terrain[:, 1].contiguous())

    if (len(terrain) != nx * ny or torch.unique(ix * ny + iy).numel() != nx * ny):
        raise ValueError("terrain must be a complete grid without duplicate rows")

    height = terrain.new_empty((nx, ny))
    height[ix, iy] = terrain[:, 2]
    return x_axis, y_axis, height


def terrain_at(xq, yq, x_axis, y_axis, height):
    """Return physical terrain height and slopes dh/dx, dh/dy.

    Single grid: axes (nx,), (ny,), height (nx, ny).
    Batched grids: axes (k, nx), (k, ny), height (k, nx, ny);
    queries must then have shape (k, n).

    Queries and grid tensors must use physical metres and the same device.
    Boundary cells extrapolate linearly beyond the sampled terrain.
    """

    ix = torch.searchsorted(x_axis, xq.contiguous(), right=True).clamp(1, x_axis.shape[-1] - 1) - 1
    iy = torch.searchsorted(y_axis, yq.contiguous(), right=True).clamp(1, y_axis.shape[-1] - 1) - 1

    batch = ()
    if x_axis.ndim == 2:
        batch = (torch.arange(height.shape[0], device=height.device)[:, None],)

    x0 = x_axis[batch + (ix,)]
    x1 = x_axis[batch + (ix + 1,)]
    y0 = y_axis[batch + (iy,)]
    y1 = y_axis[batch + (iy + 1,)]

    h00 = height[batch + (ix, iy)]
    h10 = height[batch + (ix + 1, iy)]
    h01 = height[batch + (ix, iy + 1)]
    h11 = height[batch + (ix + 1, iy + 1)]

    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)

    a = h00 + tx * (h10 - h00)
    b = h01 + tx * (h11 - h01)

    h = a + ty * (b - a)
    hx = ( (1 - ty) * (h10 - h00) + ty * (h11 - h01) ) / (x1 - x0)
    hy = (b - a) / (y1 - y0)

    return h, hx, hy