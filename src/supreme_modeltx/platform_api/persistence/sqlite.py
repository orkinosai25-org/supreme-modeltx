"""platform_api/persistence/sqlite.py — Shared SQLite helpers."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

_DEFAULT_DB_PATH = os.path.join(tempfile.gettempdir(), "supreme_modeltx_platform.sqlite3")


def resolve_db_path(db_path: str | None = None) -> str:
    """Resolve the SQLite database path for platform-api persistence."""
    path = db_path or os.environ.get("SUPREME_MODELTX_PLATFORM_DB_PATH", _DEFAULT_DB_PATH)
    if path != ":memory:":
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    return path


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Create a SQLite connection for the given database path."""
    return sqlite3.connect(resolve_db_path(db_path), check_same_thread=False)
