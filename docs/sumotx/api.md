# API

SUMOTX exposes an API-first control plane. The API is the governance surface for how AI is registered, approved, deployed, grounded, and audited.

---

## API Design Principles

- **Control-plane-first** — lifecycle governance is the primary API concern
- **Versioned** — endpoints live under explicit version prefixes
- **OpenAPI-defined** — contract-first integrations
- **Auditable** — operations are designed to produce traceable lifecycle records

---

## Governance Domains

### 1) Model and Provider Registration

Register model/provider metadata and execution profiles under customer policy constraints.

### 2) Deployment and Approval Workflow

Request deployment, evaluate policy, and apply reviewer approvals before activation.

### 3) Inference and Grounding Orchestration

Run inference through governed runtime paths with retrieval/verification hooks.

### 4) Audit and Traceability

Expose operational and governance records for compliance and incident review.

---

## Implemented / Current Repository Surface

Current implementation includes control-plane, orchestrator, and runtime microservice endpoints.

### A) Control-plane API (implemented in C#)

Source: `control-plane/src/SMTX.ControlPlane.Api/Controllers/*`

| Endpoint | Method | Status |
|---|---|---|
| `/api/health` | `GET` | Implemented |
| `/api/train/sharepoint` | `POST` | Implemented |
| `/api/train/{jobId}` | `GET` | Implemented |
| `/api/train` | `GET` | Implemented |
| `/api/models` | `GET` | Implemented |
| `/api/models/{id}/activate` | `POST` | Implemented |
| `/api/inference/reload` | `POST` | Implemented |

### B) Runtime services (implemented in Python)

| Service | Endpoint(s) | Notes |
|---|---|---|
| `inference/inference_service.py` | `GET /health`, `GET /ready`, `POST /generate`, `POST /reload` | Stateless reloadable inference scaffold used for control-plane reload integration/testing; default runtime serving is `inference/cpu_inference_server.py` |
| `inference/cpu_inference_server.py` | `GET /health`, `POST /generate` | Default CPU inference backend |
| `inference/vllm_server.py` | `GET /health`, `POST /generate` | Optional GPU backend |
| `inference/retrieval_service.py` | `GET /health`, `POST /retrieve` | Retrieval grounding service |
| `inference/verification_service.py` | `GET /health`, `POST /verify` | Verification/evidence scoring service |
| `tmodels/tx/orchestrator.py` | `GET /health`, `POST /orchestrate` | Multi-stage T-Series orchestration |

These endpoints represent baseline control/routing behavior. Expanded policy/approval/audit API coverage is part of the control-plane roadmap.

---

## Current Target Architecture (API Pattern)

These target lifecycle APIs are architectural direction, not a claim that full approval/audit endpoint coverage is already implemented.

Representative lifecycle/API flow (target pattern):

1. Register model/provider
2. Submit deployment request
3. Run policy evaluation
4. Capture reviewer approval or rejection
5. Activate deployment
6. Execute inference with grounding metadata
7. Query audit records

---

## Optional Execution Modes (API-Relevant)

- Runtime provider selection (CPU default, GPU optional) is handled by orchestrator/backend configuration, not separate governance APIs
- Retrieval and verification services remain independently callable (`/retrieve`, `/verify`) and can be policy-gated by control-plane workflows
- Training execution may run on VM or Batch profiles while lifecycle actions continue to flow through control-plane APIs

---

## Future-State Programs

- Expanded approval, policy decision, and audit-query endpoint coverage
- Richer deployment request lifecycle APIs (submission, review, promotion, rollback)
- Stronger provider abstraction endpoints while preserving current customer-boundary controls

---

## Error Model

All APIs should use a consistent error envelope:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

---

## OpenAPI Specification

The static OpenAPI export is available in repository root documentation assets:

- [`docs/openapi.yaml`](../openapi.yaml)
- [`docs/openapi.json`](../openapi.json)
