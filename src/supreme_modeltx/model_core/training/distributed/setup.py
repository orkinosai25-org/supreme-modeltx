"""
model_core/training/distributed/setup.py — Distributed process group utilities.

Wraps torch.distributed init for NCCL (GPU) and Gloo (CPU) backends,
with automatic backend selection and graceful single-process no-op.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Return True if we are running inside a distributed job."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """True only on rank 0."""
    return global_rank() == 0


def init_distributed(backend: str = "auto") -> bool:
    """Initialise the default process group if running distributed.

    Args:
        backend: ``"nccl"``, ``"gloo"``, or ``"auto"`` (chooses NCCL when
                 CUDA is available, Gloo otherwise).

    Returns:
        True if distributed was initialised, False for single-process runs.
    """
    if not is_distributed():
        return False

    if dist.is_initialized():
        return True

    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)
    logger.info(
        "Distributed init: backend=%s rank=%d/%d",
        backend, global_rank(), world_size(),
    )
    return True


def cleanup_distributed() -> None:
    """Destroy the process group if initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()
