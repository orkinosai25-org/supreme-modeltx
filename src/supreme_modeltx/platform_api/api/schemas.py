"""
platform_api/api/schemas.py — Request/response Pydantic schemas for the platform API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat completions  (/v1/chat/completions)
# ---------------------------------------------------------------------------

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
    object: str = "chat.completion"
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


# ---------------------------------------------------------------------------
# Responses API  (/v1/responses)
# ---------------------------------------------------------------------------

class ResponsesRequest(BaseModel):
    """Structured responses API request (OpenAI Responses API-compatible)."""
    model: str = Field("t-dev-6l", description="Model identifier from the registry.")
    input: Union[str, list[ChatMessage]] = Field(
        ..., description="Plain text or list of messages to respond to."
    )
    instructions: str = Field(
        "", description="Optional system-level instructions prepended to the conversation."
    )
    max_output_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)


class ResponseOutputItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    model: str
    output: list[ResponseOutputItem]
    usage: dict[str, int]


# ---------------------------------------------------------------------------
# Embeddings  (/v1/embeddings)
# ---------------------------------------------------------------------------

class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible embeddings request."""
    model: str = Field("t-dev-6l", description="Model identifier.")
    input: Union[str, list[str]] = Field(..., description="Text or list of texts to embed.")
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingObject]
    usage: dict[str, int]


# ---------------------------------------------------------------------------
# API key management  (/v1/keys)
# ---------------------------------------------------------------------------

class KeyIssueRequest(BaseModel):
    """Request to issue a new API key for a project."""
    project_id: str = Field(..., description="Project to issue the key for.")
    label: str = Field("", description="Human-readable label for this key.")


class KeyIssueResponse(BaseModel):
    """Issued key response — key value is shown once only."""
    key_id: str
    project_id: str
    label: str
    key: str = Field(..., description="Plain-text API key. Store securely; not shown again.")
    created_at: datetime


class KeyMetadata(BaseModel):
    """Key metadata (never includes the plain-text key value)."""
    key_id: str
    project_id: str
    label: str
    key_prefix: str = Field(..., description="First 8 characters of the key for identification.")
    created_at: datetime
