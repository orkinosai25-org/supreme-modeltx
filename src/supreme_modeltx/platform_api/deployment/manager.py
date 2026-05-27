"""Deployment manager — tracks model deployment targets and health."""

from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DeploymentRecord(BaseModel):
    """Represents a deployed model endpoint."""

    deployment_id: str
    model_id: str
    tenant_id: str
    endpoint_url: Optional[str] = None
    status: Literal["pending", "running", "stopped", "failed"] = "pending"
    created_at: float = Field(default_factory=time.time)
    replicas: int = Field(default=1, ge=1)
    hardware: str = Field(default="cpu", description="e.g. 'cpu', 'a100', 'h100'")


class DeploymentManager:
    """Manages model deployments on behalf of tenants.

    In production this connects to an orchestration layer (Kubernetes,
    Azure Batch, etc.). This scaffold tracks deployment state in-memory.
    """

    def __init__(self) -> None:
        self._deployments: dict[str, DeploymentRecord] = {}

    def create_deployment(
        self,
        deployment_id: str,
        model_id: str,
        tenant_id: str,
        replicas: int = 1,
        hardware: str = "cpu",
    ) -> DeploymentRecord:
        record = DeploymentRecord(
            deployment_id=deployment_id,
            model_id=model_id,
            tenant_id=tenant_id,
            replicas=replicas,
            hardware=hardware,
        )
        self._deployments[deployment_id] = record
        return record

    def get(self, deployment_id: str) -> DeploymentRecord | None:
        return self._deployments.get(deployment_id)

    def update_status(
        self, deployment_id: str, status: Literal["pending", "running", "stopped", "failed"]
    ) -> bool:
        if deployment_id not in self._deployments:
            return False
        self._deployments[deployment_id].status = status
        return True

    def list_by_tenant(self, tenant_id: str) -> list[DeploymentRecord]:
        return [d for d in self._deployments.values() if d.tenant_id == tenant_id]

    def stop(self, deployment_id: str) -> bool:
        return self.update_status(deployment_id, "stopped")
