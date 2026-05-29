"""
platform_api/model_registry/registry.py — Model registry.

Tracks available models, their metadata, and deployment status.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    """In-memory model registry."""

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {e.id: e for e in _DEFAULT_ENTRIES}

    def list_models(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        return self._entries.get(model_id)

    def register(self, entry: ModelEntry) -> ModelEntry:
        self._entries[entry.id] = entry
        return entry

    def deregister(self, model_id: str) -> bool:
        if model_id in self._entries:
            del self._entries[model_id]
            return True
        return False
