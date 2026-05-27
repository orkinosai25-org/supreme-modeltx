"""Logging configuration helpers."""

from __future__ import annotations

import logging
import sys


def configure_logging(
    level: int | str = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt: str = "%Y-%m-%dT%H:%M:%S",
) -> None:
    """Configure root logger with a sensible default format.

    Calling this once at application startup is sufficient; all
    child loggers will inherit the configuration.
    """
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
    )
    logging.getLogger("supreme_modeltx").setLevel(level)
