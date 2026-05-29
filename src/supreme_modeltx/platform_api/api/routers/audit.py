"""audit — Audit event log router (/v1/audit)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.audit.log import AuditEvent, AuditLog

router = APIRouter()

# Module-level audit log singleton (replace with DB-backed store in production)
_audit_log = AuditLog()


def get_audit_log() -> AuditLog:
    """Return the module-level audit log singleton."""
    return _audit_log


class AuditEventsListResponse(BaseModel):
    events: list[AuditEvent]
    total: int


@router.get("/events", response_model=AuditEventsListResponse)
async def list_audit_events(
    project_id: Optional[str] = Query(None, description="Filter by project ID."),
    event_type: Optional[str] = Query(None, description="Filter by event type."),
    since: Optional[datetime] = Query(None, description="Return events after this ISO 8601 timestamp."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return."),
    _: str = Depends(require_api_key),
) -> AuditEventsListResponse:
    """Query the immutable audit event log.

    Supports filtering by project, event type, and time range.
    Returns events in reverse-chronological order (newest first).
    """
    events = _audit_log.query(
        project_id=project_id,
        event_type=event_type,
        since=since,
        limit=limit,
    )
    return AuditEventsListResponse(events=events, total=len(events))

