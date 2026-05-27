"""models — Model registry listing router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.model_registry.registry import ModelRegistry, ModelEntry

router = APIRouter()
_registry = ModelRegistry()


@router.get("/", response_model=list[ModelEntry])
async def list_models(project_id: str = Depends(require_api_key)) -> list[ModelEntry]:
    """List all available models in the registry."""
    return _registry.list_models()


@router.get("/{model_id}", response_model=ModelEntry)
async def get_model(model_id: str, project_id: str = Depends(require_api_key)) -> ModelEntry:
    """Get a specific model by ID."""
    entry = _registry.get_model(model_id)
    if entry is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")
    return entry
