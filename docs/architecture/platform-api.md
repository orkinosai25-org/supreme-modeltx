# Platform API Architecture

## Overview

`platform_api` is the API-first business access layer for Supreme ModelTX. It provides governed, metered, auditable access to the model core for business users.

It is built with **FastAPI** and exposes an **OpenAI-compatible** chat completions and responses interface alongside platform-specific endpoints for project management, usage reporting, model registry access, API key management, and audit logging.

---

## Module breakdown

### `api/`
- `app.py` — FastAPI application factory (`create_app()`), router mounting, CORS
- `schemas.py` — all request/response Pydantic models:
  - `ChatMessage`, `ChatRequest`, `ChatResponse`
  - `ResponsesRequest`, `ResponsesResponse`, `ResponseOutputItem`
  - `EmbeddingsRequest`, `EmbeddingsResponse`, `EmbeddingObject`
  - `KeyIssueRequest`, `KeyIssueResponse`, `KeyMetadata`
- `routers/` — individual route modules:
  - `health.py` — `/health` liveness probe
  - `auth.py` — `/v1/auth/validate` key verification; `require_api_key` dependency
  - `models.py` — `/v1/models` registry listing
  - `chat.py` — `/v1/chat/completions` (stub; model dispatch wired in Phase 4)
  - `responses.py` — `/v1/responses` structured responses (stub)
  - `embeddings.py` — `/v1/embeddings` text embeddings (stub)
  - `projects.py` — `/v1/projects` CRUD
  - `usage.py` — `/v1/usage/{project_id}` aggregated token usage
  - `keys.py` — `/v1/keys` API key issuance, listing, and revocation
  - `audit.py` — `/v1/audit/events` immutable audit event query

### `auth/`
- `keys.py` — `issue_key(project_id)`, `verify_api_key(key)`, `revoke_key(key)`
  - Keys are stored as scrypt hashes (never plain text)
  - Constant-time comparison (`hmac.compare_digest`) to resist timing attacks
  - Dev key seeded from `SMTX_API_KEY` environment variable
- `key_store.py` — `KeyMetadataStore` — tracks key_id, label, prefix, project for the keys router
  - Stores metadata only (no plain key, no hash)

### `audit/`
- `log.py` — `AuditEvent`, `AuditLog` — append-only in-memory audit log
  - Records event type, project, model, timestamp, and metadata
  - Queryable by project_id, event_type, time range, and limit
  - Production: replace with PostgreSQL or ClickHouse

### `tenants/`
- `models.py` — `Project`, `ProjectCreate` Pydantic models
- `store.py` — `ProjectStore` (in-memory; replace with DB for production)

### `usage/`
- `metering.py` — `UsageEvent`, `UsageSummary`, `UsageLedger`
  - Records prompt tokens, completion tokens, and request counts per project
  - In-memory for scaffolding; production backend should be ClickHouse or TimescaleDB

### `model_registry/`
- `registry.py` — `ModelEntry`, `ModelRegistry`
  - Tracks model id, family, variant, stage, checkpoint path, context length
  - Seeded with T-Dev-6L (development) and T-101 (staging, not yet available)

### `deployment/`
- `service.py` — `Deployment`, `ComputeSpec`, `DeploymentService`
  - Manages deployment lifecycle: pending → running → stopped
  - Production: integrate with Kubernetes, container registry, vLLM serving

---

## API surface

```
GET  /health                          → liveness probe

GET  /v1/auth/validate                → validate API key

GET  /v1/models                       → list models
GET  /v1/models/{model_id}            → get model details

POST /v1/chat/completions             → chat completions (OpenAI-compatible stub)
POST /v1/responses                    → structured responses (OpenAI Responses API-compatible stub)
POST /v1/embeddings                   → text embeddings (OpenAI-compatible stub)

GET  /v1/projects                     → list projects
POST /v1/projects                     → create project
GET  /v1/projects/{id}                → get project

GET  /v1/usage/{project_id}           → usage summary

POST   /v1/keys                       → issue new API key (returns plain key once)
GET    /v1/keys                       → list key metadata (prefix, label, project)
DELETE /v1/keys/{key_id}              → revoke API key

GET  /v1/audit/events                 → query audit event log
```

Swagger UI: `http://localhost:9000/docs`
ReDoc: `http://localhost:9000/redoc`

---

## Authentication

All endpoints except `/health` require a Bearer token:

```http
Authorization: Bearer <api-key>
```

Keys are issued per-project via `POST /v1/keys`. The dev key defaults to the value of `SMTX_API_KEY` (default: `"dev-secret"` — change in production).

---

## Roadmap for platform_api

- [ ] PostgreSQL / SQLite backend for project and usage persistence
- [ ] API key rotation and expiry
- [ ] Per-project rate limiting (token bucket or sliding window)
- [ ] Real chat completions dispatch to InferenceEngine
- [ ] Real embeddings dispatch to embedding model
- [ ] Streaming responses (SSE)
- [ ] Full audit log persistence (ClickHouse or PostgreSQL)
- [ ] RBAC (admin / viewer / billing roles per project)
- [ ] Deployment CRUD with Kubernetes/container backend
- [ ] Fine-tuning job submission (`POST /v1/fine_tunes`)
- [ ] Batch inference (`POST /v1/batches`)

