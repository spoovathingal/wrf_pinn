"""Conditional FastEddy PINN driver: one GPU or torchrun with multiple GPUs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from wrf_pinn.config.physics import DEFAULT_PHYSICS
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.data.conditional_case import read_conditional_case
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig, RawEncoder, RawProjection
from wrf_pinn.training.train_conditional import (
    ConditionalTrainConfig, train_conditional,
)

log = logging.getLogger("train_conditional_lcc")


def _scaling_from_recipe(recipe: dict) -> ResidualScalingConfig:
    names = recipe["coord_names"] + recipe["state_vars"]
    offsets = recipe["coord_offset"] + recipe["state_offset"]
    scales = recipe["coord_scale"] + recipe["state_scale"]
    return ResidualScalingConfig(**{
        name: VariableScale(offset=offset, scale=scale or 1.0)
        for name, offset, scale in zip(names, offsets, scales)
    })


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
    ap.add_argument("--source-z-mode",
        choices=("fe_terrain_following", "reference_plus_terrain",
        "height_above_ground", "physical_altitude"),
        default="fe_terrain_following",
        help=(" Meaning of the saved vertical coordinate: "
            "fe_terrain_following uses reference heights before terrain "
            "compression; physical_altitude uses already-converted heights"))
    ap.add_argument("--terrain-top-height",type=float, default=4027.5,
        help=("FE terrain-compression height H in metres; "
            "4027.5 is the inferred CSTL value"),
    )
    ap.add_argument("--fe-surface-reference-z", type=float, default=10.125386238098145,
        help=("First FE reference-level height before terrain compression, "
            "in metres; its local height above ground is z_ref*(1-h/H)"),
    )
    ap.add_argument(
        "--model-z-max", type=float, default=None,
        help="Physical-height normalization scale; not a model ceiling")
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
        "--tke-weight", type=float, default=1.0,
        help="Relative SGS TKE weight within the PDE loss; 0 disables this term")
    ap.add_argument(
        "--tke-residual-scale", type=float, default=1.0,
        help="Conservative TKE residual scale [kg m^-1 s^-3]")
    ap.add_argument(
        "--latent", choices=("pod", "raw", "encoder"), default="pod",
        help="z: POD coeffs; the full raw fields through one linear layer; or through a learned conv encoder")
    ap.add_argument(
        "--predict-change", action="store_true",
        help="state = initial field + (t - t0)/T * network (needs --keep-raw cases)")
    ap.add_argument("--encoder-dim", type=int, default=256, help="encoder latent size")
    args = ap.parse_args()

    if not 0.0 < args.terrain_top_height < float("inf"):
        ap.error("--terrain-top-height must be finite and positive")
    if not (0.0 < args.fe_surface_reference_z < args.terrain_top_height):
        ap.error("--fe-surface-reference-z must be finite and between "
            "zero and --terrain-top-height")
    if (args.model_z_max is not None and not 0.0 < args.model_z_max < float("inf")):
        ap.error("--model-z-max must be finite and positive")
    if min(args.epochs, args.batch_cases, args.log_every, args.checkpoint_every) < 1:
        ap.error(
            "epochs, batch-cases, log-every and checkpoint-every must be positive")
    if args.n_cases is not None and args.n_cases < 1:
        ap.error("n-cases must be positive")

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
        recipe = json.loads(
            (args.cases_dir / "metadata.json").read_text()
        )["normalization"]

        source_scaling = _scaling_from_recipe(recipe)
        scaling = source_scaling
        if args.model_z_max is not None:
            scaling = replace(source_scaling, z=VariableScale(offset=0.0, scale=args.model_z_max))
        # Preprocessing preserves FastEddy qv in g/kg.
        # Convert to kg/kg inside the physics calculations.
        physics = replace(DEFAULT_PHYSICS, q_v_to_kg_per_kg=args.q_v_to_kg_per_kg)

        all_paths = sorted((args.cases_dir / "train").glob("*.npz"))
        if args.n_cases is not None:
            all_paths = all_paths[:args.n_cases]

        if len(all_paths) < world:
            raise ValueError(
                f"Need at least {world} training cases; "
                f"found {len(all_paths)}"
            )

        # All ranks construct an identical model.
        read_latent = "pod" if args.latent == "pod" else "raw"
        z_dim = read_conditional_case(all_paths[0], read_latent).z.shape[0]
        encoder = None
        if args.latent == "encoder":
            with np.load(all_paths[0]) as blob:
                nx, ny, nz = (int(v) for v in blob["initial_grid"])
                nv = blob["ini_raw"].shape[0]
                bnd_shape = blob["bnd_raw"].shape
            encoder = RawEncoder((nv, nz, nx, ny), bnd_shape, args.encoder_dim)
        elif args.latent == "raw":
            encoder = RawProjection(z_dim, args.encoder_dim)

        # Each rank loads only its own portion of the dataset.
        paths = all_paths[rank::world]
        batch_cases = min(args.batch_cases, len(all_paths) // world)

        model = ConditionalModel(z_dim, ConditionalModelConfig(), encoder)
        cfg = ConditionalTrainConfig(
            epochs=args.epochs, batch_cases=batch_cases, cases_per_pass=args.cases_per_pass,
            readers=args.readers, loader_workers=args.loader_workers, device=device,
            log_every=args.log_every,
            learning_rate=args.lr, learning_rate_min=args.lr_min,
            weight_flow_boundary=args.boundary_weight, weight_pde=args.pde_weight,
            source_scaling=source_scaling, source_z_mode=args.source_z_mode, scaling=scaling,
            terrain_top_height=args.terrain_top_height, fe_surface_reference_z=args.fe_surface_reference_z,
            physics=physics, tke_weight=args.tke_weight, tke_residual_scale=args.tke_residual_scale,
            flux_offset=tuple(recipe["flux_offset"]), flux_scale=tuple(s or 1.0 for s in recipe["flux_scale"]),
            profile=args.profile, bf16_data=args.bf16_data,
            compile=args.compile_model, events_path=args.events,
            checkpoint_path=args.checkpoint, checkpoint_every=args.checkpoint_every,
            latent=read_latent, predict_change=args.predict_change,
        )
        log.info(
            "geometry: source_z_mode=%s; H=%.6f m; "
            "first reference z=%.9f m; "
            "model z offset=%.6f m; model z scale=%.6f m",
            cfg.source_z_mode, cfg.terrain_top_height, cfg.fe_surface_reference_z,
            scaling.z.offset, scaling.z.scale,
        )
        log.info(
            "device=%s; local cases=%d/%d; latent dim=%d; "
            "batch/GPU=%d; global batch=%d; q_v conversion=%g; mode=%s; "
            "lr=%g->%s; boundary weight=%g; pde weight=%g; tke weight=%g; latent=%s; params=%d",
            device, len(paths), len(all_paths), z_dim,
            batch_cases, batch_cases * world,
            args.q_v_to_kg_per_kg,
            "lazy" if args.lazy else "resident",
            args.lr, args.lr_min, args.boundary_weight, args.pde_weight, args.tke_weight,
            args.latent, sum(p.numel() for p in model.parameters()),
        )

        if args.lazy:
            hist = train_conditional(
                model, config=cfg, case_paths=paths,
            )
        else:
            t0 = time.time()
            cases = [read_conditional_case(path, read_latent) for path in paths]

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