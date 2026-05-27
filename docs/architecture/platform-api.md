# Platform API Architecture

## Overview

`platform_api` is the API-first business access layer for Supreme ModelTX. It provides governed, metered, auditable access to the model core for business users.

It is built with **FastAPI** and exposes an **OpenAI-compatible** chat completions interface alongside platform-specific endpoints for project management, usage reporting, and model registry access.

---

## Module breakdown

### `api/`
- `app.py` — FastAPI application factory (`create_app()`), router mounting, CORS
- `schemas.py` — `ChatMessage`, `ChatRequest`, `ChatResponse`
- `routers/` — individual route modules:
  - `health.py` — `/health` liveness probe
  - `auth.py` — `/v1/auth/validate` key verification; `require_api_key` dependency
  - `models.py` — `/v1/models` registry listing
  - `chat.py` — `/v1/chat/completions` (stub; model dispatch wired in Phase 4)
  - `projects.py` — `/v1/projects` CRUD
  - `usage.py` — `/v1/usage/{project_id}` aggregated token usage

### `auth/`
- `keys.py` — `issue_key(project_id)`, `verify_api_key(key)`, `revoke_key(key)`
  - Keys are stored as SHA-256 hashes (never plain text)
  - Constant-time comparison (`hmac.compare_digest`) to resist timing attacks
  - Dev key seeded from `SMTX_API_KEY` environment variable

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
POST /v1/chat/completions             → chat completions (stub)
GET  /v1/projects                     → list projects
POST /v1/projects                     → create project
GET  /v1/projects/{id}                → get project
GET  /v1/usage/{project_id}           → usage summary
```

Swagger UI: `http://localhost:9000/docs`  
ReDoc: `http://localhost:9000/redoc`

---

## Authentication

All endpoints except `/health` require a Bearer token:

```http
Authorization: Bearer <api-key>
```

Keys are issued per-project. The dev key defaults to the value of `SMTX_API_KEY` (default: `"dev-secret"` — change in production).

---

## Roadmap for platform_api

- [ ] PostgreSQL / SQLite backend for project and usage persistence
- [ ] API key rotation and expiry
- [ ] Per-project rate limiting (token bucket or sliding window)
- [ ] Real chat completions dispatch to InferenceEngine
- [ ] Streaming responses (SSE)
- [ ] Audit log (immutable event trail for compliance)
- [ ] RBAC (admin / viewer / billing roles per project)
- [ ] Deployment CRUD with Kubernetes/container backend
