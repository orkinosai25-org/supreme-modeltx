"""
routers/governed_models.py — Enterprise governed model assignment

Implements the SUMOTX governance lifecycle for admin-controlled model selection:

- GET  /v1/governed-models               List approved models (filtered by policy)
- GET  /v1/model-assignments/{tenant}/{workspace}
                                         Get active model assignment for a workspace
- POST /v1/model-assignments             Assign an approved model (policy check + audit)
- GET  /v1/model-assignments/{tenant}/{workspace}/audit
                                         Full audit trail for a workspace

Governance invariants enforced:
  1. Only admin-approved models appear in the approved list.
  2. Policy filters (data sensitivity, use case) are applied before the list is returned.
  3. Model assignment is rejected when the model is not approved, does not support the
     requested use case, or its data-sensitivity ceiling is below the requested level.
  4. Every assignment change is recorded in an immutable audit trail.
  5. Benchmark scores surface as informational metadata and do not override governance.
  6. Assignment policy links model + data source + indexing profile + RAG configuration.

Implementation note:
  - The data-source policy catalogue in this module is bootstrap scaffolding for local/demo
    flows. The long-term policy source of truth is the governed control-plane policy store.
"""

from __future__ import annotations

import logging
from typing import Annotated, List, Optional, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.auth import require_token
from api.schemas import (
    DataSensitivity,
    GovernedModelInfo,
    GovernedModelList,
    IndexingProfile,
    ModelAssignmentAuditEntry,
    ModelAssignmentAuditList,
    ModelAssignmentRequest,
    ModelAssignmentResponse,
    ModelStage,
    ModelUseCase,
)

logger = logging.getLogger("smtx.api.governed_models")

router = APIRouter(tags=["Governed Models"])

# ── In-memory stores (replaced by persistent store in production) ──────────────

# Governed model catalogue — extends the base model registry with approval metadata
_governed_models: dict[str, GovernedModelInfo] = {
    m.id: m
    for m in [
        GovernedModelInfo(
            id="model_t101_base",
            name="T-101",
            version="0.1.0",
            stage=ModelStage.production,
            description="SUMOTX base inference model (7B parameters).",
            parameters=7_000_000_000,
            is_approved=True,
            benchmark_score=72.5,
            max_data_sensitivity=DataSensitivity.high,
            supported_use_cases=[ModelUseCase.general_purpose],
        ),
        GovernedModelInfo(
            id="model_t301_retrieval",
            name="T-301",
            version="0.1.0",
            stage=ModelStage.production,
            description="Dense retrieval model backed by FAISS.",
            parameters=110_000_000,
            is_approved=True,
            benchmark_score=81.0,
            max_data_sensitivity=DataSensitivity.medium,
            supported_use_cases=[ModelUseCase.retrieval],
        ),
        GovernedModelInfo(
            id="model_t501_verification",
            name="T-501",
            version="0.1.0",
            stage=ModelStage.production,
            description="Factual-consistency verification model.",
            parameters=110_000_000,
            is_approved=True,
            benchmark_score=78.3,
            max_data_sensitivity=DataSensitivity.high,
            supported_use_cases=[ModelUseCase.verification],
        ),
    ]
}

# Active assignments keyed by (tenant_id, workspace_id, use_case)
_assignments: dict[tuple[str, str, str], ModelAssignmentResponse] = {}

# Audit entries keyed by (tenant_id, workspace_id)
_audit_log: dict[tuple[str, str], list[ModelAssignmentAuditEntry]] = {}


class DataSourcePolicy(TypedDict):
    name: str
    max_data_sensitivity: DataSensitivity
    supported_use_cases: set[ModelUseCase]
    indexing_profiles: set[IndexingProfile]
    rag_enabled: bool


# Enterprise data-source policy catalogue.
# NOTE: This in-memory catalogue is temporary bootstrap scaffolding for local/demo use.
# Policy source of truth should be persisted and governed in the control plane.
_data_sources: dict[str, DataSourcePolicy] = {
    "ds_finance_docs": {
        "name": "Finance Documents",
        "max_data_sensitivity": DataSensitivity.high,
        "supported_use_cases": {ModelUseCase.general_purpose, ModelUseCase.retrieval},
        "indexing_profiles": {
            IndexingProfile.keyword_basic,
            IndexingProfile.hybrid_enterprise,
        },
        "rag_enabled": True,
    },
    "ds_hr_kb": {
        "name": "HR Knowledge Base",
        "max_data_sensitivity": DataSensitivity.restricted,
        "supported_use_cases": {ModelUseCase.general_purpose, ModelUseCase.retrieval},
        "indexing_profiles": {
            IndexingProfile.vector_balanced,
            IndexingProfile.hybrid_enterprise,
        },
        "rag_enabled": True,
    },
    "ds_compliance_archive": {
        "name": "Compliance Archive",
        "max_data_sensitivity": DataSensitivity.restricted,
        "supported_use_cases": {ModelUseCase.verification, ModelUseCase.orchestration},
        "indexing_profiles": {IndexingProfile.keyword_basic},
        "rag_enabled": False,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

_SENSITIVITY_ORDER = {
    DataSensitivity.low: 1,
    DataSensitivity.medium: 2,
    DataSensitivity.high: 3,
    DataSensitivity.restricted: 4,
}


def _sensitivity_gte(a: DataSensitivity, b: DataSensitivity) -> bool:
    """Return True when sensitivity level *a* is >= sensitivity level *b*."""
    return _SENSITIVITY_ORDER[a] >= _SENSITIVITY_ORDER[b]


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.get(
    "/governed-models",
    response_model=GovernedModelList,
    summary="List approved models",
    description=(
        "Returns only admin-approved models, optionally filtered by enterprise policy rules "
        "(use case and maximum data-sensitivity level). Benchmark scores are surfaced as "
        "informational metadata and do not determine approval status."
    ),
)
def list_governed_models(
    _token: Annotated[str, Depends(require_token)],
    use_case: Optional[ModelUseCase] = Query(None, description="Filter by supported use case."),
    max_data_sensitivity: Optional[DataSensitivity] = Query(
        None,
        description=(
            "Filter to models cleared for at least this data-sensitivity level. "
            "E.g. 'high' returns models with max_data_sensitivity in {high, restricted}."
        ),
    ),
) -> GovernedModelList:
    items = [m for m in _governed_models.values() if m.is_approved]

    if use_case is not None:
        items = [m for m in items if use_case in m.supported_use_cases]

    if max_data_sensitivity is not None:
        items = [
            m
            for m in items
            if _sensitivity_gte(m.max_data_sensitivity, max_data_sensitivity)
        ]

    # Surface highest-benchmarked models first (informational ordering only)
    items.sort(key=lambda m: (m.benchmark_score or 0.0), reverse=True)

    return GovernedModelList(items=items, total=len(items))


@router.get(
    "/model-assignments/{tenant_id}/{workspace_id}",
    response_model=ModelAssignmentResponse,
    summary="Get active model assignment",
    description="Returns the active model assignment for a given tenant / workspace / use-case scope.",
    responses={404: {"description": "No assignment found for this scope."}},
)
def get_assignment(
    tenant_id: str,
    workspace_id: str,
    _token: Annotated[str, Depends(require_token)],
    use_case: ModelUseCase = Query(..., description="Use case scope for this assignment."),
) -> ModelAssignmentResponse:
    key = (tenant_id, workspace_id, use_case.value)
    assignment = _assignments.get(key)
    if assignment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model assignment found for this tenant/workspace/use-case scope.",
        )
    return assignment


@router.post(
    "/model-assignments",
    response_model=ModelAssignmentResponse,
    summary="Assign an approved model to a workspace",
    description=(
        "Assigns an admin-approved model to a tenant/workspace/use-case scope. "
        "Policy is enforced: the model must be approved, must support the requested use case, "
        "its data-sensitivity ceiling must meet or exceed the requested level, and the "
        "linked data-source/indexing/RAG configuration must satisfy enterprise policy. "
        "An immutable audit record is written for every assignment change."
    ),
    responses={
        400: {"description": "Governance policy violation or validation error."},
        404: {"description": "Model not found in the governed catalogue."},
    },
)
def assign_model(
    body: ModelAssignmentRequest,
    _token: Annotated[str, Depends(require_token)],
) -> ModelAssignmentResponse:
    model = _governed_models.get(body.model_id)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{body.model_id}' not found in the governed model catalogue.",
        )

    # ── Governance gate 1: model must be admin-approved ────────────────────────
    if not model.is_approved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{model.name}' is not approved for enterprise use and cannot be assigned. "
                "An admin must explicitly approve the model before it may be selected."
            ),
        )

    # ── Governance gate 2: use-case support ────────────────────────────────────
    if body.use_case not in model.supported_use_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{model.name}' does not support the '{body.use_case.value}' use case. "
                f"Supported use cases: {[uc.value for uc in model.supported_use_cases]}."
            ),
        )

    # ── Governance gate 3: data-sensitivity ceiling ────────────────────────────
    if not _sensitivity_gte(model.max_data_sensitivity, body.data_sensitivity):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{model.name}' is cleared for '{model.max_data_sensitivity.value}' data "
                f"but the requested sensitivity is '{body.data_sensitivity.value}'. "
                "Select a model that is cleared for the required sensitivity level."
            ),
        )

    # ── Governance gate 4: data-source, indexing, and RAG linkage ─────────────
    source = _data_sources.get(body.data_source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source '{body.data_source_id}' is not registered for enterprise assignment.",
        )

    source_name = str(source["name"])
    source_sensitivity = source["max_data_sensitivity"]
    source_use_cases = source["supported_use_cases"]
    source_profiles = source["indexing_profiles"]
    source_rag_enabled = bool(source["rag_enabled"])

    if body.use_case not in source_use_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Data source '{source_name}' is not approved for the '{body.use_case.value}' use case. "
                f"Allowed use cases: {[u.value for u in source_use_cases]}."
            ),
        )

    if not _sensitivity_gte(source_sensitivity, body.data_sensitivity):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Data source '{source_name}' is cleared for '{source_sensitivity.value}' data but "
                f"the assignment requests '{body.data_sensitivity.value}'."
            ),
        )

    if body.indexing_profile not in source_profiles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Data source '{source_name}' does not support indexing profile "
                f"'{body.indexing_profile.value}'."
            ),
        )

    if body.rag.enabled and not source_rag_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data source '{source_name}' is not approved for RAG-enabled assignments.",
        )

    if body.use_case == ModelUseCase.retrieval and not body.rag.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Retrieval use-case assignments require rag.enabled=true.",
        )

    if body.rag.enabled and ModelUseCase.retrieval not in model.supported_use_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{model.name}' does not support retrieval but rag.enabled=true requires "
                "retrieval capability."
            ),
        )

    # ── Apply assignment and write audit entry ─────────────────────────────────
    key = (body.tenant_id, body.workspace_id, body.use_case.value)
    existing = _assignments.get(key)
    previous_model_id = existing.model_id if existing is not None else None

    assignment = ModelAssignmentResponse(
        workspace_id=body.workspace_id,
        tenant_id=body.tenant_id,
        model_id=body.model_id,
        model_name=model.name,
        use_case=body.use_case,
        data_sensitivity=body.data_sensitivity,
        data_source_id=body.data_source_id,
        indexing_profile=body.indexing_profile,
        rag=body.rag,
        assigned_by=body.assigned_by,
    )
    _assignments[key] = assignment

    audit_key = (body.tenant_id, body.workspace_id)
    _audit_log.setdefault(audit_key, []).append(
        ModelAssignmentAuditEntry(
            workspace_id=body.workspace_id,
            tenant_id=body.tenant_id,
            use_case=body.use_case,
            previous_model_id=previous_model_id,
            new_model_id=body.model_id,
            data_source_id=body.data_source_id,
            indexing_profile=body.indexing_profile,
            rag=body.rag,
            changed_by=body.assigned_by,
            reason=body.reason,
        )
    )

    logger.info(
        (
            "Model assignment: tenant=%s workspace=%s use_case=%s model=%s (%s) "
            "data_source=%s indexing=%s rag_enabled=%s assigned_by=%s"
        ),
        body.tenant_id,
        body.workspace_id,
        body.use_case.value,
        body.model_id,
        model.name,
        body.data_source_id,
        body.indexing_profile.value,
        body.rag.enabled,
        body.assigned_by,
    )

    return assignment


@router.get(
    "/model-assignments/{tenant_id}/{workspace_id}/audit",
    response_model=ModelAssignmentAuditList,
    summary="Get model assignment audit trail",
    description=(
        "Returns the immutable audit trail for a tenant/workspace scope, most recent first. "
        "Each entry records who changed the model, from which version, to which, and why. "
        "Supply `use_case` to narrow results to a single use-case assignment history."
    ),
)
def get_audit_trail(
    tenant_id: str,
    workspace_id: str,
    _token: Annotated[str, Depends(require_token)],
    use_case: Optional[ModelUseCase] = Query(
        None,
        description=(
            "Optional filter. When provided, only audit entries for this use case are returned. "
            "Omit to retrieve the full workspace-wide trail across all use cases."
        ),
    ),
) -> ModelAssignmentAuditList:
    key = (tenant_id, workspace_id)
    all_entries: List[ModelAssignmentAuditEntry] = _audit_log.get(key, [])

    if use_case is not None:
        all_entries = [e for e in all_entries if e.use_case == use_case]

    entries = list(reversed(all_entries))
    return ModelAssignmentAuditList(
        workspace_id=workspace_id,
        tenant_id=tenant_id,
        entries=entries,
        total=len(entries),
    )
