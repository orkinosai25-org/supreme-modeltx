"""Model registry — tracks available models, versions, and routing."""

from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelRecord(BaseModel):
    """Metadata for a registered model version."""

    model_id: str = Field(description="Unique model identifier, e.g. 'supmtx-t301-v1'.")
    display_name: str
    version: str
    base_model: str = Field(description="T-Series family member, e.g. 't301'.")
    status: Literal["available", "deprecated", "training", "pending"] = "pending"
    checkpoint_path: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    description: Optional[str] = None
    context_length: int = 2048
    is_default: bool = False


class ModelRegistry:
    """Central registry for sovereign model versions.

    Provides lookup, registration, and routing capabilities used by
    the API serving layer to select the right model backend.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelRecord] = {}

    def register(
        self,
        model_id: str,
        display_name: str,
        version: str,
        base_model: str,
        checkpoint_path: str | None = None,
        description: str | None = None,
        context_length: int = 2048,
        is_default: bool = False,
    ) -> ModelRecord:
        if is_default:
            for m in self._models.values():
                m.is_default = False
        record = ModelRecord(
            model_id=model_id,
            display_name=display_name,
            version=version,
            base_model=base_model,
            checkpoint_path=checkpoint_path,
            description=description,
            context_length=context_length,
            is_default=is_default,
            status="available" if checkpoint_path else "pending",
        )
        self._models[model_id] = record
        return record

    def get(self, model_id: str) -> ModelRecord | None:
        return self._models.get(model_id)

    def get_default(self) -> ModelRecord | None:
        for m in self._models.values():
            if m.is_default and m.status == "available":
                return m
        available = [m for m in self._models.values() if m.status == "available"]
        return available[0] if available else None

    def list_available(self) -> list[ModelRecord]:
        return [m for m in self._models.values() if m.status == "available"]

    def deprecate(self, model_id: str) -> bool:
        if model_id in self._models:
            self._models[model_id].status = "deprecated"
            return True
        return False
