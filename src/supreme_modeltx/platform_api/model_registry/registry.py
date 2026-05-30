"""
platform_api/model_registry/registry.py — Model registry.

Tracks available models, their metadata, and deployment status.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from supreme_modeltx.platform_api.persistence.sqlite import connect, resolve_db_path


class ModelEntry(BaseModel):
    id: str
    name: str
    family: str = "t-series"
    variant: str
    stage: Literal["development", "staging", "production", "deprecated"] = "development"
    description: str = ""
    parameter_count: Optional[int] = None
    context_length: int = 2048
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    model_config_path: Optional[str] = None
    inference_dtype: Optional[str] = None
    is_available: bool = True


# Default registry seeded with the canonical dev model
_DEFAULT_ENTRIES: list[ModelEntry] = [
    ModelEntry(
        id="t-dev-6l",
        name="T-Dev-6L",
        family="t-series",
        variant="t-dev-6l",
        stage="development",
        description="6-layer, 512-hidden smoke/dev model (~25M params). CPU-runnable.",
        parameter_count=25_000_000,
        context_length=512,
    ),
    ModelEntry(
        id="t-101",
        name="T-101",
        family="t-series",
        variant="t101",
        stage="staging",
        description="7B dense transformer base model (training in progress).",
        parameter_count=7_000_000_000,
        context_length=4096,
        is_available=False,
    ),
]


class ModelRegistry:
    """SQLite-backed model registry."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = resolve_db_path(db_path)
        self._initialize()
        self._seed_defaults()

    def list_models(self) -> list[ModelEntry]:
        with connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    id, name, family, variant, stage, description,
                    parameter_count, context_length, created_at, checkpoint_path,
                    tokenizer_path, model_config_path, inference_dtype, is_available
                FROM model_registry
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        with connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    id, name, family, variant, stage, description,
                    parameter_count, context_length, created_at, checkpoint_path,
                    tokenizer_path, model_config_path, inference_dtype, is_available
                FROM model_registry
                WHERE id = ?
                """,
                (model_id,),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def register(self, entry: ModelEntry) -> ModelEntry:
        with connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO model_registry (
                    id, name, family, variant, stage, description,
                    parameter_count, context_length, created_at, checkpoint_path,
                    tokenizer_path, model_config_path, inference_dtype, is_available
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    family = excluded.family,
                    variant = excluded.variant,
                    stage = excluded.stage,
                    description = excluded.description,
                    parameter_count = excluded.parameter_count,
                    context_length = excluded.context_length,
                    created_at = excluded.created_at,
                    checkpoint_path = excluded.checkpoint_path,
                    tokenizer_path = excluded.tokenizer_path,
                    model_config_path = excluded.model_config_path,
                    inference_dtype = excluded.inference_dtype,
                    is_available = excluded.is_available
                """,
                (
                    entry.id,
                    entry.name,
                    entry.family,
                    entry.variant,
                    entry.stage,
                    entry.description,
                    entry.parameter_count,
                    entry.context_length,
                    entry.created_at.isoformat(),
                    entry.checkpoint_path,
                    entry.tokenizer_path,
                    entry.model_config_path,
                    entry.inference_dtype,
                    int(entry.is_available),
                ),
            )
            conn.commit()
        return entry

    def deregister(self, model_id: str) -> bool:
        with connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM model_registry WHERE id = ?", (model_id,))
            conn.commit()
        return cursor.rowcount > 0

    def _initialize(self) -> None:
        with connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    family TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    parameter_count INTEGER,
                    context_length INTEGER NOT NULL DEFAULT 2048,
                    created_at TEXT NOT NULL,
                    checkpoint_path TEXT,
                    tokenizer_path TEXT,
                    model_config_path TEXT,
                    inference_dtype TEXT,
                    is_available INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()

    def _seed_defaults(self) -> None:
        with connect(self._db_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO model_registry (
                    id, name, family, variant, stage, description,
                    parameter_count, context_length, created_at, checkpoint_path,
                    tokenizer_path, model_config_path, inference_dtype, is_available
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        entry.id,
                        entry.name,
                        entry.family,
                        entry.variant,
                        entry.stage,
                        entry.description,
                        entry.parameter_count,
                        entry.context_length,
                        entry.created_at.isoformat(),
                        entry.checkpoint_path,
                        entry.tokenizer_path,
                        entry.model_config_path,
                        entry.inference_dtype,
                        int(entry.is_available),
                    )
                    for entry in _DEFAULT_ENTRIES
                ],
            )
            conn.commit()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row | tuple) -> ModelEntry:
        return ModelEntry(
            id=row[0],
            name=row[1],
            family=row[2],
            variant=row[3],
            stage=row[4],
            description=row[5] or "",
            parameter_count=row[6],
            context_length=row[7],
            created_at=row[8],
            checkpoint_path=row[9],
            tokenizer_path=row[10],
            model_config_path=row[11],
            inference_dtype=row[12],
            is_available=bool(row[13]),
        )
