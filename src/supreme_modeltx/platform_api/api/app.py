"""FastAPI application for the supreme-modeltx platform API.

Exposes endpoints compatible with the OpenAI API surface so that
existing tooling can integrate with minimal changes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from supreme_modeltx.platform_api.auth.tokens import TokenStore
from supreme_modeltx.platform_api.model_registry.registry import ModelRegistry
from supreme_modeltx.platform_api.usage.meter import UsageMeter

security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(description="'system', 'user', or 'assistant'")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "supreme-modeltx"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class KeyCreateRequest(BaseModel):
    tenant_id: str
    project_id: Optional[str] = None


class KeyCreateResponse(BaseModel):
    key_id: str
    secret: str
    tenant_id: str


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    token_store: TokenStore | None = None,
    model_registry: ModelRegistry | None = None,
    usage_meter: UsageMeter | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    All stores are injectable for testing; production deployments
    should provide persistent-backed implementations.
    """
    _token_store = token_store or TokenStore()
    _registry = model_registry or ModelRegistry()
    _meter = usage_meter or UsageMeter()

    app = FastAPI(
        title="Supreme ModelTX Platform API",
        description=(
            "British sovereign AI platform — PyTorch-native LLM engine "
            "with an API-first business surface."
        ),
        version="0.1.0",
    )

    def _authenticate(
        creds: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> str:
        if creds is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token.",
            )
        parts = creds.credentials.split(":", 1)
        if len(parts) != 2:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format. Use 'key_id:secret'.",
            )
        key_id, raw_secret = parts
        api_key = _token_store.authenticate(key_id, raw_secret)
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key.",
            )
        return api_key.tenant_id

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "platform": "supreme-modeltx"}

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    @app.get("/v1/models", response_model=ModelsResponse, tags=["models"])
    def list_models(tenant_id: str = Depends(_authenticate)) -> ModelsResponse:
        records = _registry.list_available()
        return ModelsResponse(
            data=[
                ModelInfo(id=r.model_id, created=int(r.created_at))
                for r in records
            ]
        )

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["chat"])
    def chat_completions(
        request: ChatCompletionRequest,
        tenant_id: str = Depends(_authenticate),
    ) -> ChatCompletionResponse:
        record = _registry.get(request.model)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model}' not found.",
            )

        # Placeholder: real implementation routes to InferenceEngine
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion = "This is a placeholder response from the supreme-modeltx engine."
        completion_tokens = len(completion.split())

        _meter.record(
            tenant_id=tenant_id,
            model_id=request.model,
            operation="chat",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            request_id=str(uuid.uuid4()),
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    @app.get("/v1/usage", tags=["usage"])
    def get_usage(tenant_id: str = Depends(_authenticate)) -> dict[str, Any]:
        return _meter.summarise(tenant_id)

    # ------------------------------------------------------------------
    # API key management
    # ------------------------------------------------------------------

    @app.post("/v1/keys", response_model=KeyCreateResponse, tags=["keys"])
    def create_key(request: KeyCreateRequest) -> KeyCreateResponse:
        api_key, raw_secret = _token_store.issue(
            tenant_id=request.tenant_id, project_id=request.project_id
        )
        return KeyCreateResponse(
            key_id=api_key.key_id,
            secret=raw_secret,
            tenant_id=api_key.tenant_id,
        )

    return app
