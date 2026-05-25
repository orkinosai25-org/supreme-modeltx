"""
tests/test_governed_models.py — Governance gate and audit trail tests for the
governed_models router.

Covers:
  - Only approved models are returned from the approved-models list
  - Use-case and data-sensitivity policy filters work correctly
  - Gate 1: unapproved model is rejected on assignment
  - Gate 2: data-sensitivity ceiling is enforced
  - Gate 3: unsupported use-case is rejected
  - Audit entry is written on every assignment change
  - Audit trail can be filtered by use case
  - Benchmark ordering does not bypass approval
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
import api.routers.governed_models as _gm_module
from api.schemas import DataSensitivity, GovernedModelInfo, ModelStage, ModelUseCase

# ── Fixtures ───────────────────────────────────────────────────────────────────

_AUTH_HEADERS = {"Authorization": "Bearer dev-secret"}

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_stores():
    """Restore module-level in-memory stores to a clean state before each test."""
    orig_models = dict(_gm_module._governed_models)
    orig_assignments = dict(_gm_module._assignments)
    orig_audit = {k: list(v) for k, v in _gm_module._audit_log.items()}

    yield

    _gm_module._governed_models.clear()
    _gm_module._governed_models.update(orig_models)
    _gm_module._assignments.clear()
    _gm_module._assignments.update(orig_assignments)
    _gm_module._audit_log.clear()
    _gm_module._audit_log.update({k: list(v) for k, v in orig_audit.items()})


def _add_unapproved_model() -> GovernedModelInfo:
    m = GovernedModelInfo(
        id="model_unapproved",
        name="Draft Model",
        version="0.0.1",
        stage=ModelStage.experimental,
        is_approved=False,
        max_data_sensitivity=DataSensitivity.high,
        supported_use_cases=[ModelUseCase.general_purpose],
    )
    _gm_module._governed_models[m.id] = m
    return m


def _assignment_body(
    *,
    model_id: str,
    use_case: str,
    data_sensitivity: str,
    data_source_id: str = "ds_finance_docs",
    indexing_profile: str = "hybrid_enterprise",
    rag: dict | None = None,
    workspace_id: str = "ws-test",
    tenant_id: str = "tenant-test",
    assigned_by: str = "admin@corp",
    reason: str = "Governance test assignment reason",
) -> dict:
    if rag is None:
        rag = {"enabled": use_case == "retrieval", "top_k": 5}

    return {
        "workspace_id": workspace_id,
        "tenant_id": tenant_id,
        "model_id": model_id,
        "use_case": use_case,
        "data_sensitivity": data_sensitivity,
        "data_source_id": data_source_id,
        "indexing_profile": indexing_profile,
        "rag": rag,
        "assigned_by": assigned_by,
        "reason": reason,
    }


# ── Sensitivity helper ─────────────────────────────────────────────────────────


def test_sensitivity_ordering_is_correct():
    order = _gm_module._SENSITIVITY_ORDER
    assert order[DataSensitivity.low] < order[DataSensitivity.medium]
    assert order[DataSensitivity.medium] < order[DataSensitivity.high]
    assert order[DataSensitivity.high] < order[DataSensitivity.restricted]


def test_sensitivity_gte():
    gte = _gm_module._sensitivity_gte
    assert gte(DataSensitivity.high, DataSensitivity.medium)
    assert gte(DataSensitivity.high, DataSensitivity.high)
    assert not gte(DataSensitivity.medium, DataSensitivity.high)


# ── Approved model listing ─────────────────────────────────────────────────────


def test_list_governed_models_returns_only_approved():
    _add_unapproved_model()
    resp = client.get("/v1/governed-models", headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert all(m["is_approved"] for m in items)
    assert not any(m["name"] == "Draft Model" for m in items)


def test_list_governed_models_benchmark_ordering_does_not_bypass_approval():
    _add_unapproved_model()
    resp = client.get("/v1/governed-models", headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    names = [m["name"] for m in resp.json()["items"]]
    assert "Draft Model" not in names


def test_list_governed_models_filter_by_use_case():
    resp = client.get(
        "/v1/governed-models", params={"use_case": "retrieval"}, headers=_AUTH_HEADERS
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) >= 1
    assert all("retrieval" in m["supported_use_cases"] for m in items)


def test_list_governed_models_filter_by_sensitivity_excludes_lower():
    resp = client.get(
        "/v1/governed-models",
        params={"max_data_sensitivity": "high"},
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200
    order = _gm_module._SENSITIVITY_ORDER
    for m in resp.json()["items"]:
        clearance = DataSensitivity(m["max_data_sensitivity"])
        assert order[clearance] >= order[DataSensitivity.high], (
            f"Model '{m['name']}' has clearance '{clearance}' below 'high'"
        )


# ── Gate 1: model must be admin-approved ───────────────────────────────────────


def test_assign_model_gate1_unapproved_raises_400():
    _add_unapproved_model()
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_unapproved",
            use_case="general_purpose",
            data_sensitivity="low",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 400
    assert "not approved" in resp.json()["detail"].lower()


# ── Gate 2: data-sensitivity ceiling ──────────────────────────────────────────


def test_assign_model_gate2_exceeds_ceiling_raises_400():
    # T-301 is cleared for 'medium'; requesting 'high' must fail
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="high",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "medium" in detail
    assert "high" in detail


def test_assign_model_gate2_equal_sensitivity_succeeds():
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="medium",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "model_t301_retrieval"


def test_assign_model_gate2_lower_sensitivity_succeeds():
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="low",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200


# ── Gate 3: use-case support ───────────────────────────────────────────────────


def test_assign_model_gate3_unsupported_use_case_raises_400():
    # T-101 supports general_purpose only; assigning as retrieval must fail
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t101_base",
            use_case="retrieval",
            data_sensitivity="low",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 400
    assert "retrieval" in resp.json()["detail"].lower()


def test_assign_model_gate3_supported_use_case_succeeds():
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t101_base",
            use_case="general_purpose",
            data_sensitivity="low",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "model_t101_base"


# ── Gate 4: data-source + indexing + RAG linkage ───────────────────────────────


def test_assign_model_gate4_unknown_data_source_raises_404():
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="low",
            reason="Unknown data source must fail",
            data_source_id="ds-does-not-exist",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 404
    assert "data source" in resp.json()["detail"].lower()


def test_assign_model_gate4_unsupported_indexing_profile_raises_400():
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="low",
            reason="Unsupported indexing profile must fail",
            data_source_id="ds_hr_kb",
            indexing_profile="keyword_basic",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 400
    assert "indexing profile" in resp.json()["detail"].lower()


def test_assign_model_gate4_retrieval_requires_rag_enabled():
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="low",
            reason="Retrieval assignment should require rag",
            rag={"enabled": False, "top_k": 5},
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 400
    assert "rag.enabled=true" in resp.json()["detail"].lower()


def test_assign_model_response_and_audit_include_data_linkage_fields():
    resp = client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t501_verification",
            use_case="verification",
            data_sensitivity="high",
            data_source_id="ds_compliance_archive",
            indexing_profile="keyword_basic",
            rag={"enabled": False, "top_k": 5},
            workspace_id="ws-linkage",
            tenant_id="tenant-linkage",
            reason="Verification assignment with explicit data linkage",
        ),
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assignment = resp.json()
    assert assignment["data_source_id"] == "ds_compliance_archive"
    assert assignment["indexing_profile"] == "keyword_basic"
    assert assignment["rag"] == {"enabled": False, "top_k": 5}

    audit = client.get(
        "/v1/model-assignments/tenant-linkage/ws-linkage/audit",
        headers=_AUTH_HEADERS,
    )
    assert audit.status_code == 200
    entry = audit.json()["entries"][0]
    assert entry["data_source_id"] == "ds_compliance_archive"
    assert entry["indexing_profile"] == "keyword_basic"
    assert entry["rag"] == {"enabled": False, "top_k": 5}


# ── Audit trail ────────────────────────────────────────────────────────────────


def test_assign_model_writes_audit_entry():
    client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t101_base",
            use_case="general_purpose",
            data_sensitivity="low",
            workspace_id="ws-audit",
            tenant_id="tenant-audit",
            assigned_by="admin@corp",
            reason="Initial assignment for audit test",
        ),
        headers=_AUTH_HEADERS,
    )

    resp = client.get(
        "/v1/model-assignments/tenant-audit/ws-audit/audit", headers=_AUTH_HEADERS
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    entry = data["entries"][0]
    assert entry["new_model_id"] == "model_t101_base"
    assert entry["previous_model_id"] is None
    assert entry["changed_by"] == "admin@corp"
    assert "audit test" in entry["reason"]


def test_assign_model_second_assignment_records_previous_model():
    common = {
        "use_case": "general_purpose",
        "data_sensitivity": "low",
        "workspace_id": "ws-chain",
        "tenant_id": "tenant-chain",
    }
    client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t101_base",
            reason="First assignment in chain",
            **common,
        ),
        headers=_AUTH_HEADERS,
    )

    # Register a second approved general-purpose model temporarily
    second = GovernedModelInfo(
        id="model_t101_v2",
        name="T-101 v2",
        version="0.2.0",
        stage=ModelStage.production,
        is_approved=True,
        max_data_sensitivity=DataSensitivity.high,
        supported_use_cases=[ModelUseCase.general_purpose],
    )
    _gm_module._governed_models[second.id] = second

    client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t101_v2",
            reason="Upgrade to v2 in chain",
            **common,
        ),
        headers=_AUTH_HEADERS,
    )

    resp = client.get(
        "/v1/model-assignments/tenant-chain/ws-chain/audit", headers=_AUTH_HEADERS
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    # Most-recent first: v2 upgrade is index 0
    assert data["entries"][0]["new_model_id"] == "model_t101_v2"
    assert data["entries"][0]["previous_model_id"] == "model_t101_base"


def test_audit_trail_filtered_by_use_case_returns_only_matching():
    client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t101_base",
            use_case="general_purpose",
            data_sensitivity="low",
            workspace_id="ws-multiuse",
            tenant_id="tenant-filter",
            reason="General purpose assignment for filter",
        ),
        headers=_AUTH_HEADERS,
    )
    client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="low",
            workspace_id="ws-multiuse",
            tenant_id="tenant-filter",
            reason="Retrieval assignment for filter test",
        ),
        headers=_AUTH_HEADERS,
    )

    resp = client.get(
        "/v1/model-assignments/tenant-filter/ws-multiuse/audit",
        params={"use_case": "retrieval"},
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["entries"][0]["use_case"] == "retrieval"


def test_audit_trail_no_filter_returns_all_use_cases():
    client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t101_base",
            use_case="general_purpose",
            data_sensitivity="low",
            workspace_id="ws-all",
            tenant_id="tenant-all",
            reason="General purpose assignment test",
        ),
        headers=_AUTH_HEADERS,
    )
    client.post(
        "/v1/model-assignments",
        json=_assignment_body(
            model_id="model_t301_retrieval",
            use_case="retrieval",
            data_sensitivity="low",
            workspace_id="ws-all",
            tenant_id="tenant-all",
            reason="Retrieval assignment all test",
        ),
        headers=_AUTH_HEADERS,
    )

    resp = client.get(
        "/v1/model-assignments/tenant-all/ws-all/audit", headers=_AUTH_HEADERS
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 2
