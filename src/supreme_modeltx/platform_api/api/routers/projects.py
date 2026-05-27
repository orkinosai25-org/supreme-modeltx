"""projects — Project / tenant management router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.tenants.models import Project, ProjectCreate
from supreme_modeltx.platform_api.tenants.store import ProjectStore

router = APIRouter()
_store = ProjectStore()


@router.get("/", response_model=list[Project])
async def list_projects(project_id: str = Depends(require_api_key)) -> list[Project]:
    """List all projects (admin view)."""
    return _store.list_projects()


@router.post("/", response_model=Project, status_code=201)
async def create_project(
    body: ProjectCreate,
    _: str = Depends(require_api_key),
) -> Project:
    """Create a new project."""
    return _store.create_project(body)


@router.get("/{pid}", response_model=Project)
async def get_project(pid: str, _: str = Depends(require_api_key)) -> Project:
    """Get a project by ID."""
    project = _store.get_project(pid)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {pid!r} not found.")
    return project
