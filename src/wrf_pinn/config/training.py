"""Training-loop configuration for WRF PINN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptimizerName = Literal["adam", "adamw"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for PINN training."""

    name: OptimizerName = "adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive; got {self.learning_rate}."
            )

        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be nonnegative; got {self.weight_decay}.")


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training configuration."""

    epochs: int = 10000
    log_every: int = 100
    gradient_clip_norm: float | None = None
    device: str = "auto"
    optimizer: OptimizerConfig = OptimizerConfig()

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive; got {self.epochs}.")

        if self.log_every < 1:
            raise ValueError(f"log_every must be positive; got {self.log_every}.")

        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be positive when provided; "
                f"got {self.gradient_clip_norm}."
            )


DEFAULT_TRAINING = TrainingConfig()
