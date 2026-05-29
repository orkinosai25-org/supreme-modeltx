"""
platform_api/usage/metering.py — Token usage metering and rate-limit scaffolding.

In production, this should write to a time-series database (e.g. ClickHouse,
TimescaleDB, or a simple append-only log) and read aggregates via SQL or
pre-computed summaries.

This scaffolding uses an in-memory ledger.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


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
    """In-memory usage accumulator (replace with persistent store in production)."""

    def __init__(self) -> None:
        self._events: list[UsageEvent] = []

    def record(self, event: UsageEvent) -> None:
        self._events.append(event)

    def summarise(self, project_id: str) -> UsageSummary:
        summary = UsageSummary(project_id=project_id)
        for evt in self._events:
            if evt.project_id == project_id:
                summary.total_prompt_tokens += evt.prompt_tokens
                summary.total_completion_tokens += evt.completion_tokens
                summary.total_requests += 1
        return summary
