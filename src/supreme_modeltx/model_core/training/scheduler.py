"""
model_core/training/scheduler.py — LR scheduler builders.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from supreme_modeltx.model_core.config.schema import SchedulerConfig


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a learning-rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Scheduler configuration.
        total_steps: Total training steps (used for decay endpoint).

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR` scheduler.
    """
    warmup = cfg.warmup_steps
    min_ratio = cfg.min_lr_ratio

    if cfg.name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, cosine)

    elif cfg.name == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            decay = 1.0 - (step - warmup) / max(1, total_steps - warmup)
            return max(min_ratio, decay)

    else:  # constant
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
