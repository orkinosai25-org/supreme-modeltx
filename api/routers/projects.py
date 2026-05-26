"""
routers/projects.py — CRUD for Projects / workspaces

GET    /v1/projects
POST   /v1/projects
GET    /v1/projects/{project_id}
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_token
from api.schemas import Project, ProjectCreate, ProjectList

logger = logging.getLogger("smtx.api.projects")

router = APIRouter(prefix="/projects", tags=["Projects"])

# In-memory store (replace with a real database in production)
_projects: dict[str, Project] = {}


@router.get(
    "",
    response_model=ProjectList,
    summary="List projects",
    description="Returns all projects accessible to the authenticated principal.",
)
def list_projects(
    _token: Annotated[str, Depends(require_token)],
) -> ProjectList:
    items = list(_projects.values())
    return ProjectList(items=items, total=len(items))


@router.post(
    "",
    response_model=Project,
    status_code=status.HTTP_201_CREATED,
    summary="Create a project",
    description=(
        "Creates a new project workspace.  Projects are the top-level isolation "
        "boundary for deployments, training runs, and model artefacts."
    ),
)
def create_project(
    body: ProjectCreate,
    _token: Annotated[str, Depends(require_token)],
) -> Project:
    project = Project(name=body.name, description=body.description, tags=body.tags)
    _projects[project.id] = project
    logger.info("Created project id=%s name=%s", project.id, project.name)
    return project


@router.get(
    "/{project_id}",
    response_model=Project,
    summary="Get a project",
    description="Returns metadata for a single project by its unique identifier.",
    responses={404: {"description": "Project not found"}},
)
def get_project(
    project_id: str,
    _token: Annotated[str, Depends(require_token)],
) -> Project:
    project = _projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    return project
