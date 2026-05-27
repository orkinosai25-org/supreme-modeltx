"""embeddings — Text embeddings router (OpenAI-compatible)."""
from __future__ import annotations

from typing import Union

from fastapi import APIRouter, Depends

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.api.schemas import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingObject,
)

router = APIRouter()


@router.post("/", response_model=EmbeddingsResponse)
async def create_embeddings(
    request: EmbeddingsRequest,
    project_id: str = Depends(require_api_key),
) -> EmbeddingsResponse:
    """Text embeddings endpoint (stub — real embedding model wired in Phase 2).

    Returns zero-vectors as a placeholder. The embedding dimension will match
    the selected model's hidden size once the inference backend is connected.
    """
    inputs: list[str] = (
        [request.input] if isinstance(request.input, str) else request.input
    )
    # Stub: return a zero vector of dimension 1 per input
    data = [
        EmbeddingObject(index=i, embedding=[0.0])
        for i, _ in enumerate(inputs)
    ]
    total_tokens = sum(len(text.split()) for text in inputs)
    return EmbeddingsResponse(
        model=request.model,
        data=data,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )
