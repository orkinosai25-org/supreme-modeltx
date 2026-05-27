"""
platform_api/audit/log.py — In-memory audit event log.

Production: replace with an append-only database table (PostgreSQL, ClickHouse, etc.)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    """A single immutable audit event."""
    id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    project_id: str
    event_type: str
    model: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLog:
    """In-memory append-only audit log."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        """Append an event to the log."""
        self._events.append(event)

    def query(
        self,
        *,
        project_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return events matching the given filters, newest first."""
        results = self._events
        if project_id is not None:
            results = [e for e in results if e.project_id == project_id]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return list(reversed(results))[:limit]
