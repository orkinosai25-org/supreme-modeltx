"""
routers/chat.py — Basic inference / chat endpoint

POST /v1/chat/completions

Forwards the request to the T-X Orchestrator and wraps the response in
an OpenAI-compatible chat-completion envelope.  Streaming is reserved for
a future release.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_token
from api.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)

logger = logging.getLogger("smtx.api.chat")

router = APIRouter(prefix="/chat", tags=["Chat"])

_ORCHESTRATOR_URL = os.environ.get(
    "ORCHESTRATOR_URL",
    f"http://localhost:{os.environ.get('ORCHESTRATOR_PORT', '8080')}",
)
_HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "120"))


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    summary="Chat completion",
    description=(
        "Sends a list of messages to the SUMOTX T-X pipeline and returns a "
        "single chat completion.  The pipeline runs T-101 (inference), T-201 "
        "(reasoning), T-301 (retrieval), and T-501 (verification) before "
        "returning the final response.  Streaming responses are planned for a "
        "future release (`stream: true` is accepted but treated as `false`)."
    ),
)
async def chat_completions(
    body: ChatCompletionRequest,
    _token: Annotated[str, Depends(require_token)],
) -> ChatCompletionResponse:
    # Build a single prompt from the conversation history
    prompt = "\n".join(f"{m.role}: {m.content}" for m in body.messages)

    payload = {
        "prompt": prompt,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "top_p": body.top_p,
    }

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        try:
            r = await client.post(f"{_ORCHESTRATOR_URL}/orchestrate", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Orchestrator error: {exc}",
            ) from exc

    final_text: str = data.get("final_response", "")
    prompt_tokens: int = sum(len(m.content.split()) for m in body.messages)
    completion_tokens: int = len(final_text.split())

    return ChatCompletionResponse(
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final_text),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
