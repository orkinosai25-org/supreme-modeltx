"""
utils/device.py — Device selection helpers.

Provides a simple, consistent way to obtain a torch.device with CUDA
preference when available, falling back gracefully to CPU or MPS.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch.device.

    Priority order (unless *prefer* overrides):
      1. CUDA — if torch.cuda.is_available()
      2. MPS  — if torch.backends.mps.is_available() (Apple Silicon)
      3. CPU  — fallback

    Args:
        prefer: Optional override — one of "cuda", "mps", "cpu".
                ``None`` triggers automatic selection.

    Returns:
        A :class:`torch.device` ready for use.
    """
    prefer = (
        prefer
        or os.environ.get("SUPREME_MODELTX_DEVICE", os.environ.get("SMTX_DEVICE", ""))
    ).lower().strip()

    if prefer == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    if prefer == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available — falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")

    if prefer == "cpu":
        return torch.device("cpu")

    # Automatic selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device: MPS (Apple Silicon)")
        return torch.device("mps")

    logger.info("Device: CPU")
    return torch.device("cpu")


def device_info(device: torch.device) -> dict:
    """Return a dictionary of human-readable device information."""
    info: dict = {"type": device.type}
    if device.type == "cuda":
        idx = device.index or 0
        info["name"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        info["total_memory_gb"] = round(props.total_memory / 1e9, 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    return info
