"""responses — Structured responses router (OpenAI Responses API-compatible)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.api.schemas import (
    ResponsesRequest,
    ResponsesResponse,
    ResponseOutputItem,
)

router = APIRouter()


@router.post("/", response_model=ResponsesResponse)
async def create_response(
    request: ResponsesRequest,
    project_id: str = Depends(require_api_key),
) -> ResponsesResponse:
    """Structured responses endpoint (stub — model backend wired in Phase 2)."""
    output_text = "[Model response stub — inference not yet wired]"
    return ResponsesResponse(
        id=f"resp-{uuid.uuid4().hex[:12]}",
        model=request.model,
        output=[ResponseOutputItem(text=output_text)],
        usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
