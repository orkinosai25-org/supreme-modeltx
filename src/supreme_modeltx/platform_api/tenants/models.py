"""Tenant and project models for the platform API."""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field


class Project(BaseModel):
    """A project within a tenant."""

    project_id: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class Tenant(BaseModel):
    """A top-level tenant (organisation) on the platform."""

    tenant_id: str
    name: str
    plan: str = Field(default="standard", description="Billing plan identifier.")
    created_at: float = Field(default_factory=time.time)
    active: bool = True
    projects: list[Project] = Field(default_factory=list)

    def add_project(self, project_id: str, name: str, description: str | None = None) -> Project:
        project = Project(
            project_id=project_id,
            tenant_id=self.tenant_id,
            name=name,
            description=description,
        )
        self.projects.append(project)
        return project

    def get_project(self, project_id: str) -> Project | None:
        for p in self.projects:
            if p.project_id == project_id:
                return p
        return None


class TenantStore:
    """In-memory tenant registry (replace with persistent backend in production)."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}

    def create(self, tenant_id: str, name: str, plan: str = "standard") -> Tenant:
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' already exists.")
        tenant = Tenant(tenant_id=tenant_id, name=name, plan=plan)
        self._tenants[tenant_id] = tenant
        return tenant

    def get(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def list_all(self) -> list[Tenant]:
        return list(self._tenants.values())
