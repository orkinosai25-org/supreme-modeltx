"""
utils/logging.py — Standardised logging setup for supreme_modeltx.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger for supreme_modeltx with a clean format.

    Idempotent — calling multiple times has no extra effect.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger("supreme_modeltx")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(numeric)
