"""Device selection utilities."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def select_device(prefer: str | None = None) -> torch.device:
    """Select the best available compute device.

    Args:
        prefer: Optional explicit device string, e.g. ``"cuda"``,
                ``"mps"``, or ``"cpu"``. When ``None`` the best
                available device is chosen automatically.

    Returns:
        :class:`torch.device` to use for model operations.
    """
    if prefer is not None:
        device = torch.device(prefer)
        logger.info("Using explicitly requested device: %s", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "CUDA device selected: %s (%.1f GiB VRAM)",
            props.name,
            props.total_memory / 2**30,
        )
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple MPS device selected.")
        return device

    device = torch.device("cpu")
    logger.info("No GPU found; using CPU.")
    return device


def device_info() -> dict[str, object]:
    """Return a summary of available compute devices."""
    info: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_properties(i).name,
                "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info
