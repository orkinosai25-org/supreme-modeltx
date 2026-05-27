"""Usage metering for API calls — tokens consumed, request counts, etc."""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field


class UsageRecord(BaseModel):
    """A single usage event."""

    tenant_id: str
    project_id: Optional[str] = None
    model_id: str
    operation: str = Field(description="e.g. 'chat', 'embeddings', 'fine_tune'")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: float = Field(default_factory=time.time)
    request_id: Optional[str] = None

    @property
    def billable_tokens(self) -> int:
        return self.total_tokens


class UsageMeter:
    """Accumulates usage records and provides aggregation helpers.

    In production, replace the in-memory store with a time-series
    database or streaming pipeline.
    """

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def record(
        self,
        tenant_id: str,
        model_id: str,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        project_id: str | None = None,
        request_id: str | None = None,
    ) -> UsageRecord:
        rec = UsageRecord(
            tenant_id=tenant_id,
            project_id=project_id,
            model_id=model_id,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            request_id=request_id,
        )
        self._records.append(rec)
        return rec

    def get_usage(
        self,
        tenant_id: str,
        since: float | None = None,
        until: float | None = None,
    ) -> list[UsageRecord]:
        records = [r for r in self._records if r.tenant_id == tenant_id]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]
        return records

    def summarise(self, tenant_id: str, since: float | None = None) -> dict[str, int]:
        records = self.get_usage(tenant_id, since=since)
        return {
            "request_count": len(records),
            "total_tokens": sum(r.total_tokens for r in records),
            "prompt_tokens": sum(r.prompt_tokens for r in records),
            "completion_tokens": sum(r.completion_tokens for r in records),
        }
