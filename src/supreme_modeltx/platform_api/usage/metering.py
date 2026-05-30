"""platform_api/usage/metering.py — SQLite-backed token usage metering."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from supreme_modeltx.platform_api.persistence.sqlite import connect, resolve_db_path


class UsageEvent(BaseModel):
    project_id: str
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UsageSummary(BaseModel):
    project_id: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0


class UsageLedger:
    """SQLite-backed usage accumulator."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = resolve_db_path(db_path)
        self._initialize()

    def record(self, event: UsageEvent) -> None:
        with connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO usage_events (project_id, model_id, prompt_tokens, completion_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.project_id,
                    event.model_id,
                    event.prompt_tokens,
                    event.completion_tokens,
                    event.timestamp.isoformat(),
                ),
            )
            conn.commit()

    def summarise(self, project_id: str) -> UsageSummary:
        with connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(prompt_tokens), 0),
                    COALESCE(SUM(completion_tokens), 0),
                    COUNT(*)
                FROM usage_events
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()
        return UsageSummary(
            project_id=project_id,
            total_prompt_tokens=int(row[0]),
            total_completion_tokens=int(row[1]),
            total_requests=int(row[2]),
        )

    def _initialize(self) -> None:
        with connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()
