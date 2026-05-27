"""
platform_api/deployment/service.py — Deployment service boundary.

Manages the lifecycle of model deployments:
  - Creating a deployment from a registry entry + compute spec
  - Tracking deployment status
  - Routing inference requests to the correct backend

This is a scaffolding layer.  Production implementation would integrate
with Kubernetes, container orchestration, or vLLM serving infrastructure.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ComputeSpec(BaseModel):
    """Compute requirements for a deployment."""
    replicas: int = Field(1, ge=1)
    gpu_count: int = Field(0, ge=0)
    gpu_type: Optional[str] = None
    cpu_cores: int = Field(4, ge=1)
    memory_gb: int = Field(16, ge=1)


class Deployment(BaseModel):
    id: str
    model_id: str
    project_id: str
    status: Literal["pending", "running", "stopped", "failed"] = "pending"
    compute: ComputeSpec = Field(default_factory=ComputeSpec)
    endpoint_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeploymentService:
    """Manages deployment lifecycle (in-memory scaffold)."""

    def __init__(self) -> None:
        self._deployments: dict[str, Deployment] = {}

    def create(self, model_id: str, project_id: str, compute: ComputeSpec | None = None) -> Deployment:
        import uuid
        dep = Deployment(
            id=str(uuid.uuid4()),
            model_id=model_id,
            project_id=project_id,
            compute=compute or ComputeSpec(),
        )
        self._deployments[dep.id] = dep
        return dep

    def get(self, deployment_id: str) -> Optional[Deployment]:
        return self._deployments.get(deployment_id)

    def list_for_project(self, project_id: str) -> list[Deployment]:
        return [d for d in self._deployments.values() if d.project_id == project_id]

    def stop(self, deployment_id: str) -> bool:
        dep = self._deployments.get(deployment_id)
        if dep:
            dep.status = "stopped"
            dep.updated_at = datetime.now(timezone.utc)
            return True
        return False
