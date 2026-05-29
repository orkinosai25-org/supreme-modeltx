"""
model_core/training/precision.py — Mixed-precision / BF16 training hooks.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch


def get_autocast_context(dtype_str: str, device_type: str = "cuda"):
    """Return a torch.autocast context manager for the given dtype.

    Args:
        dtype_str: One of ``"bfloat16"``, ``"float16"``, or ``"float32"``.
        device_type: ``"cuda"`` or ``"cpu"``.

    Returns:
        A context manager (real autocast or a null context for float32).
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    if dtype == torch.float32:
        return _null_context()
    return torch.autocast(device_type=device_type, dtype=dtype)


@contextmanager
def _null_context() -> Generator[None, None, None]:
    yield


def get_grad_scaler(dtype_str: str, device_type: str = "cuda") -> torch.cuda.amp.GradScaler | None:
    """Return a GradScaler for float16, or None for bfloat16 / CPU training.

    BF16 does not need gradient scaling (sufficient dynamic range).
    CPU does not support GradScaler.
    """
    if dtype_str == "float16" and device_type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None
