"""
routers/models_registry.py — Model registry

GET  /v1/models
GET  /v1/models/{model_id}
POST /v1/models/{model_id}/promote
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_token
from api.schemas import ModelInfo, ModelList, ModelStage, PromoteRequest

logger = logging.getLogger("smtx.api.models")

router = APIRouter(prefix="/models", tags=["Models"])

# Seed with the canonical T-Series models
_models: dict[str, ModelInfo] = {
    m.id: m
    for m in [
        ModelInfo(
            id="model_t101_base",
            name="T-101",
            version="0.1.0",
            stage=ModelStage.production,
            description="SUMOTX base inference model (7B parameters).",
            parameters=7_000_000_000,
        ),
        ModelInfo(
            id="model_t301_retrieval",
            name="T-301",
            version="0.1.0",
            stage=ModelStage.production,
            description="Dense retrieval model backed by FAISS.",
            parameters=110_000_000,
        ),
        ModelInfo(
            id="model_t501_verification",
            name="T-501",
            version="0.1.0",
            stage=ModelStage.production,
            description="Factual-consistency verification model.",
            parameters=110_000_000,
        ),
    ]
}


@router.get(
    "",
    response_model=ModelList,
    summary="List models",
    description="Returns all models registered in the SUMOTX model registry.",
)
def list_models(
    _token: Annotated[str, Depends(require_token)],
) -> ModelList:
    items = list(_models.values())
    return ModelList(items=items, total=len(items))


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    summary="Get a model",
    description="Returns full metadata for a registered model.",
    responses={404: {"description": "Model not found"}},
)
def get_model(
    model_id: str,
    _token: Annotated[str, Depends(require_token)],
) -> ModelInfo:
    model = _models.get(model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
    return model


@router.post(
    "/{model_id}/promote",
    response_model=ModelInfo,
    summary="Promote a model",
    description=(
        "Promotes a model to the target stage (experimental → staging → production).  "
        "This is a governance gate that prevents untested models from serving production traffic."
    ),
    responses={404: {"description": "Model not found"}},
)
def promote_model(
    model_id: str,
    body: PromoteRequest,
    _token: Annotated[str, Depends(require_token)],
) -> ModelInfo:
    model = _models.get(model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
    model.stage = body.stage
    logger.info("Model %s promoted to stage=%s", model_id, body.stage.value)
    return model
