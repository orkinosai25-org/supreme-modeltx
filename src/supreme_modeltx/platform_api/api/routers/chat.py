"""chat — Chat/completions router (OpenAI-compatible schema)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.api.schemas import ChatRequest, ChatResponse, ChatMessage

router = APIRouter()


@router.post("/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    project_id: str = Depends(require_api_key),
) -> ChatResponse:
    """Chat completions endpoint (stub — model backend wired in Phase 2)."""
    # Stub response for scaffolding; real model dispatch happens via deployment layer
    reply = ChatMessage(role="assistant", content="[Model response stub — inference not yet wired]")
    return ChatResponse(
        id="stub-0001",
        model=request.model,
        choices=[{"index": 0, "message": reply, "finish_reason": "stop"}],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
