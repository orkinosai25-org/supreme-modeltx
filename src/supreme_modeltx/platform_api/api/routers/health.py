"""health — Liveness and readiness probes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from supreme_modeltx.platform_api.api import engine as engine_module

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Ready only after the selected model has loaded and registered."""
    backend = engine_module.get_engine()
    if backend is None or not backend.is_ready:
        raise HTTPException(status_code=503, detail="Model backend is not ready.")
    return {"status": "ready", "backend": str(backend.metadata["backend"])}
