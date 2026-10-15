"""JSON-compatible provenance shared by conditional checkpoints and run logs."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import torch


def run_manifest(model, config, *, case_paths=None, cases=None):
    """Capture configuration once; metadata hashes identify files, not case contents."""
    from wrf_pinn.data.conditional_case import STATE_VARS, COORD_NAMES

    root = Path(__file__).resolve().parents[3]
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    paths = [Path(p) for p in case_paths] if case_paths is not None else []
    datasets = []
    for path in sorted({p.parent.parent / "metadata.json" for p in paths}):
        if path.is_file():
            data = path.read_bytes()
            raw = json.loads(data)
            datasets.append({
                "path": str(path.resolve()), "metadata_sha256": hashlib.sha256(data).hexdigest(),
                **{k: raw[k] for k in ("schema_version", "dataset_id", "schema", "normalization",
                                       "provenance", "pod_modes") if k in raw},
            })
    manifest = {
        "schema_version": 1,
        "model": {"type": "conditional", "config": asdict(model.config),
                  "latent_dim": model.net[0].in_features - model.config.coord_dim},
        "state_vars": list(STATE_VARS), "coord_names": list(COORD_NAMES),
        "training": asdict(config),
        "data": {"datasets": datasets,
                 "rank_case_ids": [p.stem for p in paths] if paths else [c.name for c in cases or []],
                 "case_identity_scope": "this rank; metadata hashes do not hash case contents"},
        "code": {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")},
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "torch_initial_seed": torch.initial_seed(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
    }
    # Fail here rather than silently saving an unportable object in a checkpoint.
    json.dumps(manifest, allow_nan=False)
    return manifest


def write_event(path, event):
    if path is not None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
