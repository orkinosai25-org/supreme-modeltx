"""auth — API key verification router."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from supreme_modeltx.platform_api.auth.keys import verify_api_key

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> str:
    """Dependency: validate Bearer token, return project_id."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key.")
    project_id = verify_api_key(credentials.credentials)
    if project_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return project_id


@router.get("/validate")
async def validate_key(project_id: str = Security(require_api_key)) -> dict:
    """Validate the supplied API key and return the associated project."""
    return {"valid": True, "project_id": project_id}
