"""chat — Chat/completions router (OpenAI-compatible schema)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from supreme_modeltx.platform_api.api import engine as engine_module
from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.api.schemas import ChatMessage, ChatRequest, ChatResponse

router = APIRouter()


@router.post("/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    project_id: str = Depends(require_api_key),
) -> ChatResponse:
    """Chat completions endpoint backed by the local checkpoint InferenceEngine.

    Returns HTTP 503 when no checkpoint has been loaded (i.e. neither
    ``SMTX_CHECKPOINT_PATH`` nor ``SMTX_TOKENIZER_PATH`` are set or the files
    do not exist).  Returns HTTP 500 if inference raises an unexpected error.
    """
    backend = engine_module.get_engine()
    if backend is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model backend not configured. "
                "Set SMTX_CHECKPOINT_PATH and SMTX_TOKENIZER_PATH to enable local inference."
            ),
        )

    try:
        text, prompt_tokens, completion_tokens = backend.generate_from_messages(
            request.messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    reply = ChatMessage(role="assistant", content=text)
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=request.model,
        choices=[{"index": 0, "message": reply.model_dump(), "finish_reason": "stop"}],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
