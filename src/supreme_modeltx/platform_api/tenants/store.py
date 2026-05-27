"""
platform_api/tenants/store.py — In-memory project store (scaffolding).

Replace with a database-backed implementation for production.
"""
from __future__ import annotations

from typing import Optional

from supreme_modeltx.platform_api.tenants.models import Project, ProjectCreate


class ProjectStore:
    """Simple in-memory project registry."""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        # Seed a dev project
        dev = Project(id="dev-project", name="Dev Project", description="Default development project")
        self._projects[dev.id] = dev

    def list_projects(self) -> list[Project]:
        return list(self._projects.values())

    def get_project(self, project_id: str) -> Optional[Project]:
        return self._projects.get(project_id)

    def create_project(self, body: ProjectCreate) -> Project:
        project = Project(name=body.name, description=body.description, owner_email=body.owner_email)
        self._projects[project.id] = project
        return project
