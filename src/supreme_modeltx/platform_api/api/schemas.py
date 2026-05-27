"""
platform_api/api/schemas.py — Request/response Pydantic schemas for the platform API.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completions request."""
    model: str = Field("t-dev-6l", description="Model identifier from the registry.")
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]
