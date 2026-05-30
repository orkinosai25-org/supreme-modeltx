"""platform_api/audit/log.py — SQLite-backed audit event log."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from supreme_modeltx.platform_api.persistence.sqlite import connect, resolve_db_path


class AuditEvent(BaseModel):
    """A single immutable audit event."""
    id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    project_id: str
    event_type: str
    model: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLog:
    """SQLite-backed append-only audit log."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = resolve_db_path(db_path)
        self._initialize()

    def record(self, event: AuditEvent) -> None:
        """Append an event to the log."""
        with connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_events (id, project_id, event_type, model, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.project_id,
                    event.event_type,
                    event.model,
                    event.timestamp.isoformat(),
                    json.dumps(event.metadata),
                ),
            )
            conn.commit()

    def query(
        self,
        *,
        project_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return events matching the given filters, newest first."""
        query = """
            SELECT id, project_id, event_type, model, timestamp, metadata
            FROM audit_events
            WHERE 1=1
        """
        params: list[Any] = []
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with connect(self._db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _initialize(self) -> None:
        with connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    model TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _row_to_event(row: tuple[str, str, str, str | None, str, str]) -> AuditEvent:
        return AuditEvent(
            id=row[0],
            project_id=row[1],
            event_type=row[2],
            model=row[3],
            timestamp=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )
