# Platform API Architecture

The `platform_api` package provides an API-first business surface for `supreme-modeltx`.

## Design Goals

1. **OpenAI-compatible surface** — existing client tooling integrates with minimal changes
2. **Tenant isolation** — strict per-tenant data and usage boundaries
3. **Metered usage** — every token is counted for billing and governance
4. **Stable API** — businesses integrate once; model upgrades are transparent
5. **Auditable** — all key actions are logged

## Endpoint Surface

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat generation |
| `GET`  | `/v1/usage` | Usage summary for tenant |
| `POST` | `/v1/keys` | Issue API key |

Planned additions:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/embeddings` | Generate embeddings |
| `POST` | `/v1/fine_tunes` | Submit fine-tuning job |
| `POST` | `/v1/projects` | Create project |
| `GET`  | `/v1/audit/events` | Audit log |

## Authentication

All endpoints (except `/health` and `/v1/keys`) require a bearer token in the form:

```
Authorization: Bearer <key_id>:<raw_secret>
```

API keys are issued via `POST /v1/keys` and scoped to a tenant (and optionally a project). Keys are stored as SHA-256 hashes — the raw secret is shown once at issuance and cannot be retrieved afterwards.

## Package Structure

```text
platform_api/
├── api/
│   └── app.py          # FastAPI application factory (create_app)
├── auth/
│   └── tokens.py       # APIKey model and TokenStore
├── tenants/
│   └── models.py       # Tenant, Project, TenantStore
├── usage/
│   └── meter.py        # UsageRecord, UsageMeter
├── model_registry/
│   └── registry.py     # ModelRecord, ModelRegistry
└── deployment/
    └── manager.py      # DeploymentRecord, DeploymentManager
```

## Running the API

```bash
pip install -e ".[dev]"
uvicorn supreme_modeltx.platform_api.api.app:app --factory --reload
```

Or with a custom factory:

```python
from supreme_modeltx.platform_api.api.app import create_app

app = create_app(
    token_store=my_persistent_token_store,
    model_registry=my_registry,
    usage_meter=my_meter,
)
```

## Production Considerations

The in-memory stores (`TokenStore`, `ModelRegistry`, `UsageMeter`, `TenantStore`, `DeploymentManager`) are scaffolded for development. Before production deployment:

- Replace `TokenStore` with a database-backed implementation
- Replace `UsageMeter` with a time-series store or streaming pipeline
- Connect `DeploymentManager` to Kubernetes / Azure Batch orchestration
- Add rate limiting middleware
- Add structured audit logging
- Add OpenTelemetry tracing

## OpenAPI Spec

The FastAPI app auto-generates an OpenAPI spec at `/openapi.json` and a Swagger UI at `/docs`.
