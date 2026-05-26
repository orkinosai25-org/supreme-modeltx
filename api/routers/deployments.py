"""
routers/deployments.py — One-click Azure provisioning

POST /v1/deployments
GET  /v1/deployments/{deployment_id}/status
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_token
from api.schemas import (
    Deployment,
    DeploymentCreate,
    DeploymentStatus,
    DeploymentStatusResponse,
)

logger = logging.getLogger("smtx.api.deployments")

router = APIRouter(prefix="/deployments", tags=["Deployments"])

# In-memory store (replace with a real database in production)
_deployments: dict[str, Deployment] = {}


@router.post(
    "",
    response_model=Deployment,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger one-click Azure deployment",
    description=(
        "Provisions the full Azure infrastructure stack for a model deployment: "
        "GPU VM, Batch pools, Storage accounts, networking, and identity wiring.  "
        "Returns immediately with `status: pending`; poll "
        "`GET /v1/deployments/{id}/status` to track progress."
    ),
)
def create_deployment(
    body: DeploymentCreate,
    _token: Annotated[str, Depends(require_token)],
) -> Deployment:
    deployment = Deployment(
        project_id=body.project_id,
        model_id=body.model_id,
        azure_region=body.azure_region,
        vm_sku=body.vm_sku,
        min_replicas=body.min_replicas,
        max_replicas=body.max_replicas,
        environment=body.environment,
        status=DeploymentStatus.pending,
    )
    _deployments[deployment.id] = deployment
    logger.info(
        "Deployment queued id=%s model=%s region=%s env=%s",
        deployment.id,
        deployment.model_id,
        deployment.azure_region,
        deployment.environment,
    )
    # TODO: enqueue an Azure Batch / Bicep provisioning job here
    return deployment


@router.get(
    "/{deployment_id}/status",
    response_model=DeploymentStatusResponse,
    summary="Get deployment status",
    description=(
        "Returns the current provisioning status of a deployment.  "
        "Status transitions: `pending` → `provisioning` → `running` (or `failed`)."
    ),
    responses={404: {"description": "Deployment not found"}},
)
def get_deployment_status(
    deployment_id: str,
    _token: Annotated[str, Depends(require_token)],
) -> DeploymentStatusResponse:
    deployment = _deployments.get(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found.")
    return DeploymentStatusResponse(
        id=deployment.id,
        status=deployment.status,
        endpoint_url=deployment.endpoint_url,
        message=f"Deployment is {deployment.status.value}.",
        updated_at=deployment.updated_at,
    )
