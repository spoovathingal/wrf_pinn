"""Run reduced FastEddy training and local smoke checks."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
import os
import numpy as np
import torch

from wrf_pinn.config.conditions import ConditionSpec, ConditionsConfig
from wrf_pinn.config.domain import make_cartesian_wrf_domain
from wrf_pinn.config.model import ModelConfig
from wrf_pinn.config.physics import PhysicsConfig
from wrf_pinn.config.boundary_data import (
    BoundaryConfig,
    NoSlipWallConfig,
    WallSurfaceConfig,
)
from wrf_pinn.config.sampling import (
    BoundarySamplingConfig,
    CollocationSamplingConfig,
    SamplingConfig,
)
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.config.training import OptimizerConfig, TrainingConfig
from wrf_pinn.data.case import CaseMetadata, read_case
from wrf_pinn.models.mlp import MLP
from wrf_pinn.training.checkpoint import save_checkpoint
from wrf_pinn.training.losses import PDEResidualScales
from wrf_pinn.training.train_pinn import TrainingSetup, train_pinn


EXPECTED_COLUMNS = (
    "x", "y", "z", "t",
    "u", "v", "w", "theta", "p_prime", "source",
)

SUPERVISED_TARGETS = ("u", "v", "w", "theta", "p_prime")

FASTEDDY_PDE_SCALES = PDEResidualScales(
    mass=3.13536843744e-3,
    x_momentum=3.35707053782e-2,
    y_momentum=1.99362535127e-2,
    z_momentum=5.58677880249e-1,
    potential_temperature=9.61239013761e-1,
)

print("torch_threads:", torch.get_num_threads())
print("interop_threads:", torch.get_num_interop_threads())
print("available_affinity:", len(os.sched_getaffinity(0)))
print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--collocation-points", type=int, default=2_048)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--boundary-kind",
        choices=("none", "no-penetration", "no-slip"),
        default="none",
    )
    parser.add_argument(
        "--boundary-surface",
        default=None,
        help="Normalized x,y,z wall-surface CSV.",
    )
    parser.add_argument(
        "--boundary-points",
        type=int,
        default=512,
        help="Number of spatial wall points sampled each epoch.",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
    )
    return parser.parse_args()

def residual_scaling_from_metadata(metadata: CaseMetadata) -> ResidualScalingConfig:
    """Read the affine scaling used by the model and physics."""
    normalization = metadata.normalization

    if normalization.get("method") != "minmax_01":
        raise ValueError("Expected minmax_01 normalization.")

    columns = normalization["columns"]
    offsets = normalization["offset"]
    scales = normalization["scale"]

    if not len(columns) == len(offsets) == len(scales):
        raise ValueError("Normalization columns, offsets, and scales must match.")

    scaling_by_name = {
        name: VariableScale(offset=float(offset), scale=float(scale))
        for name, offset, scale in zip(columns, offsets, scales)
    }

    return ResidualScalingConfig(**{
        name: scaling_by_name[name]
        for name in EXPECTED_COLUMNS[:-1]
    })

def main() -> None:
    args = parse_args()

    if args.steps < 1 or args.collocation_points < 1:
        raise ValueError("Steps and collocation points must be positive.")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        # reset peak memory stats to measure peak memory for this run
        torch.cuda.reset_peak_memory_stats()
        # wait for previous line to run
        torch.cuda.synchronize()

    if args.device == "cpu":
        cpu_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    data_path = Path(args.data)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # read metadata and check if it contains the expected column names
    metadata = CaseMetadata.load(metadata_path)
    if metadata.columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected columns {EXPECTED_COLUMNS}; got {metadata.columns}."
        )

    case = read_case(
        data_path,
        metadata,
        targets=SUPERVISED_TARGETS,
    )

    normalization = metadata.normalization

    if tuple(normalization.get("columns", ())) != EXPECTED_COLUMNS:
        raise ValueError("Normalization column order must match the schema.")

    if normalization.get("method") != "minmax_01":
        raise ValueError("Expected minmax_01 normalization.")
    
    for name, values in (
        ("coordinates", case.coordinates),
        ("targets", case.targets),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contain NaN or infinity.")

        if (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{name} are outside [0, 1].")

    scaling = residual_scaling_from_metadata(metadata)

    boundary_active = args.boundary_kind != "none"

    if boundary_active and not args.boundary_surface:
        raise ValueError(
            "--boundary-surface is required when boundary loss is enabled."
        )

    boundaries = BoundaryConfig(
        no_slip_wall=NoSlipWallConfig(
            surface=WallSurfaceConfig(
                path=args.boundary_surface or "",
                coordinate_columns=("x", "y", "z"),
            ),
            condition=(
                "no_penetration_z"
                if args.boundary_kind == "no-penetration"
                else "no_slip"
            ),
        ),
    )

    physics = PhysicsConfig()
    model_config = ModelConfig(output_dim=physics.state_dim)

    expected_supervised_variables = physics.active_variables[:-1]
    if case.target_names != expected_supervised_variables:
        raise ValueError(
            "Case targets must match the first five physics variables; "
            f"expected {expected_supervised_variables}, got {case.target_names}."
        )
    model = MLP(model_config, physics=physics)

    conditions = ConditionsConfig(
        pde=ConditionSpec(
            "pde",
            active=not args.data_only,
            weight=0.0 if args.data_only else 1.0,
        ),
        boundary=ConditionSpec(
            "boundary",
            active=boundary_active,
            weight=args.boundary_weight if boundary_active else 0.0,
        ),
        simulation=ConditionSpec(
            "simulation",
            active=True,
            weight=1.0,
        ),
        sensor=ConditionSpec(
            "sensor",
            active=False,
            weight=0.0,
        ),
    )

    coordinate_min = case.coordinates.min(axis=0)
    coordinate_max = case.coordinates.max(axis=0)
    domain = make_cartesian_wrf_domain(
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
        z_min=0.0, z_max=1.0,
        t_min=0.0, t_max=1.0,
    )

    sampling = SamplingConfig(
        collocation=CollocationSamplingConfig(
            n_points=args.collocation_points,
            method="latin_hypercube",
        ),
        boundary=BoundarySamplingConfig(
            n_points=args.boundary_points,
            method="random",
        ),
        seed=args.seed,
    )

    training = TrainingConfig(
        epochs=args.steps,
        log_every=args.log_every,
        device=args.device,
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=1.0e-3,
        ),
    )

    run_config = {
        "data_path": str(data_path.resolve()),
        "data_rows": case.n_points,
        "metadata_path": str(metadata_path.resolve()),
        "normalization": metadata.normalization,
        "variable_scaling": asdict(scaling),
        "data_only": args.data_only,
        "boundaries": asdict(boundaries),
        "boundary_kind": args.boundary_kind,
        "model": asdict(model_config),
        "physics": asdict(physics),
        "conditions": asdict(conditions),
        "sampling": asdict(sampling),
        "training": asdict(training),
        "pde_residual_scales": asdict(FASTEDDY_PDE_SCALES),
    }

    with (output_dir / "run_config.json").open("w") as file:
        json.dump(run_config, file, indent=2)

    print(f"device={args.device}")
    print(f"data_rows={case.n_points}")

    if args.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    start = time.perf_counter()

    setup = TrainingSetup(
        domain=domain,
        case=case,
        boundaries=boundaries,
        training=training,
        conditions=conditions,
        sampling=sampling,
        physics=physics,
        scaling=scaling,
        pde_residual_scales=FASTEDDY_PDE_SCALES,
    )
    history = train_pinn(model, setup)

    if args.device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    checkpoint_path = save_checkpoint(
        output_dir / "final_checkpoint.pt",
        model=model,
        history=history,
        metadata=run_config,
    )

    # Minimum checkpoint reload check.
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    reloaded_model = MLP(model_config, physics=physics)
    reloaded_model.load_state_dict(payload["model_state_dict"])
    print("checkpoint_reload=ok")

    timing = {
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
    }

    if args.device == "cuda":
        timing["cuda_peak_allocated_gb"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        timing["cuda_peak_reserved_gb"] = (
            torch.cuda.max_memory_reserved() / 1024**3
        )

    with (output_dir / "timing.json").open("w") as file:
        json.dump(timing, file, indent=2)

    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"seconds_per_step={elapsed / args.steps:.6f}")

    if args.device == "cuda":
        print(
            "cuda_peak_allocated_gb="
            f"{timing['cuda_peak_allocated_gb']:.3f}"
        )
        print(
            "cuda_peak_reserved_gb="
            f"{timing['cuda_peak_reserved_gb']:.3f}"
        )


if __name__ == "__main__":
    main()