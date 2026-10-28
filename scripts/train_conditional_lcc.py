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

import torch
import torch.distributed as dist

from wrf_pinn.config.physics import DEFAULT_PHYSICS
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.data.conditional_case import read_conditional_case
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
from wrf_pinn.training.train_conditional import (
    ConditionalTrainConfig, train_conditional,
)

log = logging.getLogger("train_conditional_lcc")
# Physical-altitude normalization for the network, in metres; not a ceiling.
MODEL_Z_SCALE_METRES = 3000.0

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
        recipe = json.loads(
            (args.cases_dir / "metadata.json").read_text()
        )["normalization"]

        source_scaling = _scaling_from_recipe(recipe)
        scaling = replace(source_scaling,
                            z=VariableScale(offset=0.0, scale=MODEL_Z_SCALE_METRES))
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
        z_dim = read_conditional_case(all_paths[0]).z.shape[0]

        # Each rank loads only its own portion of the dataset.
        paths = all_paths[rank::world]
        batch_cases = min(args.batch_cases, len(all_paths) // world)

        model_config = ConditionalModelConfig(
            tke_offset=scaling.e_sgs.offset, tke_scale=scaling.e_sgs.scale)

        model = ConditionalModel(z_dim, model_config)
        cfg = ConditionalTrainConfig(
            epochs=args.epochs, batch_cases=batch_cases,
            cases_per_pass=args.cases_per_pass, readers=args.readers,
            loader_workers=args.loader_workers, device=device,
            log_every=args.log_every,
            source_scaling=source_scaling,
            scaling=scaling, physics=physics,
            flux_offset=tuple(recipe["flux_offset"]),
            flux_scale=tuple(s or 1.0 for s in recipe["flux_scale"]),
            profile=args.profile, bf16_data=args.bf16_data,
            compile=args.compile_model, events_path=args.events,
            checkpoint_path=args.checkpoint, checkpoint_every=args.checkpoint_every,
        )

        log.info(
            "geometry: FE terrain-following; H=%.6f m;  first reference z=%.9f m; "
            "model z offset=%.6f m; model z scale=%.6f m",
            cfg.terrain_top_height, cfg.fe_surface_reference_z,
            scaling.z.offset, scaling.z.scale,
        )

        log.info(
            "device=%s; local cases=%d/%d; latent dim=%d; "
            "batch/GPU=%d; global batch=%d; q_v conversion=%g; mode=%s",
            device,len(paths), len(all_paths),
            z_dim, batch_cases, batch_cases * world, args.q_v_to_kg_per_kg,
            "lazy" if args.lazy else "resident",
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
            cases = [read_conditional_case(path) for path in paths]

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