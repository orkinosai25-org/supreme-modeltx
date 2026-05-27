"""
model_core/training/checkpoint.py — Checkpoint save, load, and resume utilities.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CKPT_STEM = "checkpoint_step"


def save_checkpoint(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    save_dir: str | Path,
    keep_last_n: int = 3,
) -> Path:
    """Save a training checkpoint.

    Checkpoints are named ``checkpoint_step_{step:08d}.pt`` inside *save_dir*.

    Args:
        step: Current training step.
        model: The model to checkpoint.
        optimizer: The current optimizer state.
        scheduler: The current LR scheduler state.
        save_dir: Directory to save into (created if absent).
        keep_last_n: Prune older checkpoints, retaining only the latest N.

    Returns:
        Path to the saved checkpoint file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / f"{_CKPT_STEM}_{step:08d}.pt"
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, ckpt_path)
    logger.info("Checkpoint saved: %s", ckpt_path)

    _prune_checkpoints(save_dir, keep_last_n)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device | None = None,
) -> int:
    """Load a checkpoint into model (and optionally optimizer/scheduler).

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: Model to load weights into.
        optimizer: If given, resume optimizer state.
        scheduler: If given, resume scheduler state.
        map_location: Passed to ``torch.load``.

    Returns:
        The training step stored in the checkpoint.
    """
    path = Path(path)
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    step = state.get("step", 0)
    logger.info("Checkpoint loaded: %s (step=%d)", path, step)
    return step


def find_latest_checkpoint(save_dir: str | Path) -> Path | None:
    """Return the path of the most recent checkpoint in *save_dir*, or None."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    candidates = sorted(
        save_dir.glob(f"{_CKPT_STEM}_*.pt"),
        key=_step_from_path,
    )
    return candidates[-1] if candidates else None


def _step_from_path(p: Path) -> int:
    m = re.search(r"_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else 0


def _prune_checkpoints(save_dir: Path, keep_last_n: int) -> None:
    """Remove old checkpoints, keeping only the last *keep_last_n*."""
    candidates = sorted(
        save_dir.glob(f"{_CKPT_STEM}_*.pt"),
        key=_step_from_path,
    )
    for old in candidates[:-keep_last_n]:
        old.unlink()
        logger.debug("Pruned old checkpoint: %s", old)
