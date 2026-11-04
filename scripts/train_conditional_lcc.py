"""Conditional FastEddy PINN driver: one GPU or torchrun with multiple GPUs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta

import logging
import os
from pathlib import Path
import sys
import time
import math

import numpy as np
import torch
import torch.distributed as dist

from wrf_pinn.config.physics import DEFAULT_PHYSICS
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.data.conditional_case import COORD_NAMES, STATE_VARS, ConditionalMetadata, read_conditional_case
from wrf_pinn.models.conditional import (
    RHO_D_INDEX, TKE_INDEX, ConditionalModel, ConditionalModelConfig, RawEncoder, RawProjection)
from wrf_pinn.training.train_conditional import (
    ConditionalTrainConfig, train_conditional,
)

log = logging.getLogger("train_conditional_lcc")
# Physical-altitude normalization for the network, in metres; not a ceiling.
MODEL_Z_SCALE_METRES = 3000.0

def _scaling_from_recipe(recipe: dict, grid_spacing: float = 1.0,
                         step_seconds: float = 1.0) -> ResidualScalingConfig:
    """Recipe scaling; x, y stored as grid indices and t as solver steps are converted to
    metres and seconds by grid_spacing and step_seconds (normalized coordinates unchanged)."""
    if tuple(recipe["coord_names"]) != COORD_NAMES:
        raise ValueError(f"Normalization must use coordinates {COORD_NAMES}.")

    if tuple(recipe["state_vars"]) != STATE_VARS:
        raise ValueError(f"Normalization must include exactly {STATE_VARS}.")

    unit = {"x": grid_spacing, "y": grid_spacing, "t": step_seconds}
    values = {}
    for prefix, names in (("coord", COORD_NAMES), ("state", STATE_VARS)):
        offsets = recipe[f"{prefix}_offset"]
        scales = recipe[f"{prefix}_scale"]

        if len(offsets) != len(names) or len(scales) != len(names):
            raise ValueError(
                f"{prefix} normalization must contain {len(names)} offsets and scales.")

        for name, offset, scale in zip(names, offsets, scales, strict=True):
            if (not math.isfinite(offset) or not math.isfinite(scale) or scale <= 0.0):
                raise ValueError(
                    f"Invalid normalization for {name}: offset must be finite and scale finite and positive.")

            u = unit.get(name, 1.0) if prefix == "coord" else 1.0
            values[name] = VariableScale(offset=offset * u, scale=scale * u)

    return ResidualScalingConfig(**values)

def _raw_input_stats(paths: list[Path], read, n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and spread per (state variable, level) of the raw initial field over n training
    cases spread evenly through the sorted list (the same on every rank). ``read`` loads a case."""
    pick = paths[::max(1, len(paths) // n)][:n]
    ini = np.stack([read(p).initial_raw.astype(np.float64) for p in pick])    # (n, nv, nz, nxy)
    mean, std = ini.mean(axis=(0, 3)), ini.std(axis=(0, 3))
    log.info("encoder input stats from %d cases: std per variable (level mean) %s",
             len(pick), np.array2string(std.mean(1), precision=4))
    return torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)


def _terrain_stats(paths: list[Path], n: int = 256) -> tuple[float, float, float]:
    """Mean and spread of subdomain-mean ground height, and spread of height about each
    subdomain's mean (m), over the same n training cases as _raw_input_stats."""
    pick = paths[::max(1, len(paths) // n)][:n]
    h = np.stack([np.load(p)["terrain"][:, 2].astype(np.float64) for p in pick])     # (n, nx*ny)
    mean = h.mean(axis=1)
    stats = float(mean.mean()), float(mean.std()), float((h - mean[:, None]).std())
    log.info("terrain stats from %d cases: subdomain mean %.0f +- %.0f m, shape spread %.1f m", len(pick), *stats)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases_dir", type=Path)
    ap.add_argument(
        "--n-cases", type=int,
        help="Global case limit before splitting between GPUs")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument(
        "--batch-cases", type=int, default=4,
        help="Cases per GPU per update")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument(
        "--q-v-to-kg-per-kg", type=float, default=1e-3,
        help="FastEddy g/kg: 0.001; data already in kg/kg: 1")
    ap.add_argument(
        "--local-rank", "--local_rank", type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")))
    ap.add_argument(
        "--cases-per-pass", type=int, default=0,
        help="Cases per forward/backward pass per GPU; 0 uses the whole batch")
    ap.add_argument(
        "--readers", type=int, default=8,
        help="Reader threads per process in lazy mode")
    ap.add_argument(
        "--loader-workers", type=int, default=0,
        help="Worker processes per GPU in lazy mode; 0 uses reader threads")
    ap.add_argument(
        "--bf16-data", action="store_true",
        help="Use bf16 for data/initial/boundary forwards; physics stays fp32")
    ap.add_argument(
        "--compile", dest="compile_model", action="store_true",
        help="Enable torch.compile; initial steps include compilation time")
    ap.add_argument(
        "--events", default=None,
        help="Append rank-zero training events to this JSONL file")
    ap.add_argument("--terrain-top-height",type=float, default=4027.5,
        help=("FE terrain-compression height H in metres; "
            "4027.5 is the inferred CSTL value"),
    )
    ap.add_argument("--fe-surface-reference-z", type=float, default=10.125386238098145,
        help=("First FE reference-level height before terrain compression, "
            "in metres; its local height above ground is z_ref*(1-h/H)"),
    )
    ap.add_argument(
        "--model-z-max", type=float, default=MODEL_Z_SCALE_METRES,
        help="Physical-height normalization scale (m); not a model ceiling")
    ap.add_argument("--lr", type=float, default=1e-3, help="(peak) learning rate")
    ap.add_argument(
        "--lr-min", type=float, default=None,
        help="cosine-decay the learning rate to this by the last epoch; omit for constant")
    ap.add_argument(
        "--boundary-weight", type=float, default=1.0,
        help="weight of the flow-boundary (side faces) loss")
    ap.add_argument(
        "--pde-weight", type=float, default=1.0,
        help="weight of the PDE loss (e.g. ~2e-6 with --predict-change, whose PDE starts ~3e5x its data loss)")
    ap.add_argument(
        "--latent", choices=("pod", "raw", "encoder"), default="pod",
        help="z: POD coeffs; the full raw fields through one linear layer; or through a learned conv encoder")
    ap.add_argument(
        "--predict-change", action="store_true",
        help="state = initial field + (t - t0)/T * network (needs --keep-raw cases)")
    ap.add_argument("--encoder-dim", type=int, default=256, help="encoder latent size")
    ap.add_argument("--grid-spacing", type=float, default=1.0,
                    help="metres per stored x, y unit: 1 for cases in metres (build_fe_cases.py); "
                         "15.0115 for grid-index cases (carve_encode_gpu.py)")
    ap.add_argument("--step-seconds", type=float, default=1.0,
                    help="seconds per stored t unit: 1 for cases in seconds; 0.025 for FastEddy-step cases")
    ap.add_argument("--initial-from-targets", action="store_true",
                    help="start each window at its first lead, all raw: no POD enters z or any loss "
                         "(POD-only cases such as cstldense128_5; window loses one snapshot)")
    ap.add_argument("--terrain-input", action="store_true",
                    help="the encoder also reads each subdomain's ground height map (--latent encoder)")
    args = ap.parse_args()

    if min(args.epochs, args.batch_cases, args.log_every, args.checkpoint_every) < 1:
        ap.error(
            "epochs, batch-cases, log-every and checkpoint-every must be positive"
        )

    if args.n_cases is not None and args.n_cases < 1:
        ap.error("n-cases must be positive")

    physics = replace(DEFAULT_PHYSICS, q_v_to_kg_per_kg=args.q_v_to_kg_per_kg)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = 0, 1
    device = args.device

    if distributed:
        if args.device == "cpu" or not torch.cuda.is_available():
            ap.error("This distributed driver requires CUDA GPUs")

        torch.cuda.set_device(args.local_rank)
        device = f"cuda:{args.local_rank}"
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=1),
        )
        rank, world = dist.get_rank(), dist.get_world_size()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format=f"%(asctime)s rank={rank}: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        metadata = ConditionalMetadata.load(args.cases_dir / "metadata.json")
        recipe = metadata.normalization

        source_scaling = _scaling_from_recipe(recipe, args.grid_spacing, args.step_seconds)
        log.info("coordinate units: x, y x %g m, t x %g s (physics and terrain slopes in m and s)",
                 args.grid_spacing, args.step_seconds)
        scaling = replace(source_scaling, z=VariableScale(offset=0.0, scale=args.model_z_max))
        # Preprocessing preserves FastEddy qv in g/kg.
        # Convert to kg/kg inside the physics calculations.

        all_paths = sorted((args.cases_dir / "train").glob("*.npz"))
        if args.n_cases is not None:
            all_paths = all_paths[:args.n_cases]

        if len(all_paths) < world:
            raise ValueError(
                f"Need at least {world} training cases; "
                f"found {len(all_paths)}"
            )

        # All ranks construct an identical model.
        if args.terrain_input and args.latent != "encoder":
            raise ValueError("--terrain-input needs --latent encoder")
        read_latent = "pod" if args.latent == "pod" else "raw_terrain" if args.terrain_input else "raw"
        read = lambda p: read_conditional_case(p, read_latent, args.initial_from_targets)
        first = read(all_paths[0])
        z_dim = first.z.shape[0]
        encoder = None
        if args.latent == "encoder":
            with np.load(all_paths[0]) as blob:
                nx, ny, nz = (int(v) for v in blob["initial_grid"])
            nv = first.initial_raw.shape[0]
            bnd_shape = tuple(np.moveaxis(first.boundary, 0, -1).shape)   # (face, time, face_len, var)
            encoder = RawEncoder((nv, nz, nx, ny), bnd_shape, args.encoder_dim, terrain=args.terrain_input)
            encoder.set_input_stats(*_raw_input_stats(all_paths, read))
            if args.terrain_input:
                encoder.set_terrain_stats(*_terrain_stats(all_paths))
        elif args.latent == "raw":
            encoder = RawProjection(z_dim, args.encoder_dim)

        # Each rank loads only its own portion of the dataset.
        paths = all_paths[rank::world]
        batch_cases = min(args.batch_cases, len(all_paths) // world)

        model_config = ConditionalModelConfig(
            tke_offset=scaling.e_sgs.offset, tke_scale=scaling.e_sgs.scale,
            rho_d_offset=scaling.rho_d.offset, rho_d_scale=scaling.rho_d.scale,
            )

        model = ConditionalModel(z_dim, model_config, encoder)
        if args.predict_change:   # the network is a change: start TKE and rho_d at no change
            with torch.no_grad():
                model.net[-1].bias[[TKE_INDEX, RHO_D_INDEX]] = 0.0
        cfg = ConditionalTrainConfig(
            epochs=args.epochs, batch_cases=batch_cases,
            cases_per_pass=args.cases_per_pass, readers=args.readers,
            loader_workers=args.loader_workers, device=device,
            log_every=args.log_every,
            learning_rate=args.lr, learning_rate_min=args.lr_min,
            weight_flow_boundary=args.boundary_weight, weight_pde=args.pde_weight,
            source_scaling=source_scaling, scaling=scaling,
            terrain_xy_scale=args.grid_spacing,
            terrain_top_height=args.terrain_top_height, fe_surface_reference_z=args.fe_surface_reference_z,
            physics=physics,
            flux_offset=tuple(recipe["flux_offset"]),
            flux_scale=tuple(s or 1.0 for s in recipe["flux_scale"]),
            profile=args.profile, bf16_data=args.bf16_data,
            compile=args.compile_model, events_path=args.events,
            checkpoint_path=args.checkpoint, checkpoint_every=args.checkpoint_every,
            latent=read_latent, predict_change=args.predict_change,
            initial_from_targets=args.initial_from_targets,
        )

        log.info(
            "geometry: FE terrain-following; H=%.6f m;  first reference z=%.9f m; "
            "model z offset=%.6f m; model z scale=%.6f m",
            cfg.terrain_top_height, cfg.fe_surface_reference_z,
            scaling.z.offset, scaling.z.scale,
        )

        log.info(
            "device=%s; local cases=%d/%d; latent dim=%d; "
            "batch/GPU=%d; global batch=%d; q_v conversion=%g; mode=%s; "
            "lr=%g->%s; boundary weight=%g; pde weight=%g; latent=%s; params=%d",
            device, len(paths), len(all_paths), z_dim,
            batch_cases, batch_cases * world,
            args.q_v_to_kg_per_kg,
            "lazy" if args.lazy else "resident",
            args.lr, args.lr_min, args.boundary_weight, args.pde_weight,
            args.latent, sum(p.numel() for p in model.parameters()),
        )

        log.info(
            "TKE output: softplus; offset=%g; scale=%g; floor=%g; "
            "softplus scale=%g; initial value=%g [physical TKE: m^2/s^2]",
            model_config.tke_offset, model_config.tke_scale,
            model_config.tke_floor, model_config.tke_softplus_scale,
            model_config.tke_initial_value,
        )

        log.info(
            "SGS width: FE cell volume; native dx=%g m; native dy=%g m; "
            "reference levels=%d; reference z=%g..%g m; edges=%s",
            physics.sgs_dx, physics.sgs_dy, len(physics.sgs_reference_z),
            physics.sgs_reference_z[0], physics.sgs_reference_z[-1],
            physics.sgs_vertical_edge_policy,
        )

        log.info("PDE: %d residuals; TKE transport equation disabled",
            physics.residual_dim)

        if args.lazy:
            hist = train_conditional(
                model, config=cfg, case_paths=paths,
            )
        else:
            t0 = time.time()
            cases = [read(path) for path in paths]

            if any(case.z.shape[0] != z_dim for case in cases):
                raise ValueError(
                    "Cases have inconsistent conditioning-vector sizes"
                )

            log.info(
                "loaded %d resident cases in %.0fs",
                len(cases), time.time() - t0,
            )
            hist = train_conditional(model, cases, cfg)

        if rank == 0:
            log.info(
                "done. total loss: first=%.4e last=%.4e",
                hist.total[0], hist.total[-1],
            )
        return 0

    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())