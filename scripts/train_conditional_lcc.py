"""Run the conditional PINN over preprocessed cases_v2.

Loads cases + the normalization recipe, builds residual/flux scaling from it, and
trains the shared ConditionalModel on the five losses. Use --n-cases for a small
proof-of-concept run; omit it to train on all cases in the split.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import torch

from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.data.conditional_case import read_conditional_case
from wrf_pinn.training.train_conditional import (
    ConditionalTrainConfig, train_conditional,
)
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig

log = logging.getLogger("train_conditional_lcc")


def _scaling_from_recipe(recipe: dict) -> ResidualScalingConfig:
    """Build the affine residual scaling (physical = offset + scale*norm) from the recipe."""
    coord = dict(zip(recipe["coord_names"], zip(recipe["coord_offset"], recipe["coord_scale"])))
    state = dict(zip(recipe["state_vars"], zip(recipe["state_offset"], recipe["state_scale"])))
    both = {**coord, **state}
    return ResidualScalingConfig(**{
        name: VariableScale(offset=off, scale=(scale or 1.0))
        for name, (off, scale) in both.items()
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases_dir", type=Path, help="cases_v2 dir with train/ test/ + metadata.json")
    ap.add_argument("--n-cases", type=int, default=None, help="limit to N train cases (PoC)")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-cases", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    recipe = json.loads((args.cases_dir / "metadata.json").read_text())["normalization"]
    scaling = _scaling_from_recipe(recipe)
    f_off = tuple(recipe["flux_offset"])
    f_scale = tuple(s or 1.0 for s in recipe["flux_scale"])

    paths = sorted(glob.glob(str(args.cases_dir / "train" / "*.npz")))
    if args.n_cases:
        paths = paths[:args.n_cases]
    log.info("loading %d cases from %s", len(paths), args.cases_dir / "train")
    t0 = time.time()
    cases = [read_conditional_case(p) for p in paths]
    log.info("loaded %d cases in %.0fs; latent z dim=%d",
             len(cases), time.time() - t0, cases[0].z.shape[0])

    model = ConditionalModel(cases[0].z.shape[0], ConditionalModelConfig())
    cfg = ConditionalTrainConfig(
        epochs=args.epochs, batch_cases=args.batch_cases, device=args.device,
        log_every=args.log_every, scaling=scaling,
        flux_offset=f_off, flux_scale=f_scale,
    )
    log.info("training: epochs=%d batch_cases=%d device=%s", cfg.epochs, cfg.batch_cases, cfg.device)
    hist = train_conditional(model, cases, cfg)
    log.info("done. total loss: first=%.4e last=%.4e", hist.total[0], hist.total[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
