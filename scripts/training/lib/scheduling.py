"""Learning Rate Scheduling

Implements various learning rate schedules for training optimization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class ScheduleConfig(Protocol):
    """Protocol for training configs with LR schedule parameters."""
    max_lr: float
    min_lr_ratio: float
    warmup_steps: int
    max_steps: int


def get_lr(step: int, cfg: ScheduleConfig) -> float:
    """
    Calculate learning rate for given step using warmup + cosine decay schedule.

    Schedule:
    1. Linear warmup: 0 → max_lr over warmup_steps
    2. Cosine decay: max_lr → min_lr over remaining steps

    This is a common schedule that:
    - Prevents unstable training at start (warmup)
    - Gradually reduces LR for fine-grained optimization (decay)

    Args:
        step: Current training step
        cfg: Training configuration with max_lr, min_lr_ratio, warmup_steps, max_steps

    Returns:
        Learning rate for this step
    """
    # Calculate minimum learning rate
    min_lr = cfg.max_lr * cfg.min_lr_ratio  # e.g., 6e-4 * 0.1 = 6e-5

    # Phase 1: Linear warmup (steps 0 to warmup_steps)
    # LR increases linearly from 0 to max_lr
    # This prevents large gradient updates when weights are random
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps

    # After max_steps: stay at minimum LR
    if step > cfg.max_steps:
        return min_lr

    # Phase 2: Cosine decay (steps warmup_steps to max_steps)
    # LR decreases smoothly from max_lr to min_lr following cosine curve
    # Calculate progress through decay phase (0.0 to 1.0)
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)  # Clamp to [0, 1]

    # Cosine coefficient: starts at 1.0, ends at 0.0
    # cos(0) = 1, cos(π) = -1, so (1 + cos(π * progress)) / 2 goes from 1 to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # Interpolate between min_lr and max_lr using cosine coefficient
    return min_lr + coeff * (cfg.max_lr - min_lr)
