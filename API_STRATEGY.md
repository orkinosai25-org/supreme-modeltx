# API Strategy

> **Supreme ModelTX** — API-first sovereign LLM platform.  
> This document describes the platform API surface, design principles, and product roadmap for business access.

---

## Vision

The Supreme ModelTX Platform API is the stable commercial interface between business customers and the underlying sovereign model engine. The API is designed so that:

- **Customers integrate once** and can adopt better models as they are released, without changing their integration code.
- **The API remains stable** even as the model backend evolves.
- **All access is metered, governed, and auditable** — essential for enterprise and public sector deployment.
- **No vendor lock-in** — the interface is OpenAI-compatible at the chat completions layer, so customers can migrate from hosted providers with minimal friction.

---

## API Surface (Current + Planned)

### Implemented (Phase 0 — Scaffold)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/auth/validate` | Validate API key |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{id}` | Get model details |
| `POST` | `/v1/chat/completions` | Chat completions (OpenAI-compatible stub) |
| `POST` | `/v1/responses` | Structured responses API (stub) |
| `POST` | `/v1/embeddings` | Text embeddings (stub) |
| `GET` | `/v1/projects` | List projects |
| `POST` | `/v1/projects` | Create project |
| `GET` | `/v1/projects/{id}` | Get project |
| `GET` | `/v1/usage/{project_id}` | Usage summary |
| `POST` | `/v1/keys` | Issue a new API key |
| `DELETE` | `/v1/keys/{key_id}` | Revoke an API key |
| `GET` | `/v1/keys` | List key metadata |
| `GET` | `/v1/audit/events` | Query audit event log |

### Planned (Phase 1–5)

| Method | Path | Description | Phase |
|---|---|---|---|
| `POST` | `/v1/fine_tunes` | Submit a fine-tuning job | 3 |
| `GET` | `/v1/fine_tunes/{id}` | Get fine-tune job status | 3 |
| `POST` | `/v1/batches` | Batch inference request | 4 |
| `GET` | `/v1/batches/{id}` | Get batch status | 4 |
| `GET` | `/v1/audit/events` | Full audit trail with filtering | 5 |
| `POST` | `/v1/rate_limits` | Configure per-project rate limits | 5 |
| `GET` | `/v1/billing/invoices` | Billing invoice listing | 5 |

---

## Authentication

All endpoints (except `/health`) require a **Bearer token**:

```http
Authorization: Bearer smtx_<project_id>_<random_hex>
```

Keys are issued per-project via `POST /v1/keys`. The key is shown **once** at issuance; store it securely. Keys are stored as **scrypt hashes** (never plain text) with constant-time comparison to resist timing attacks.

### Key lifecycle

```
POST /v1/keys   →  issue key (returns plain-text key once)
GET  /v1/keys   →  list key metadata (prefix, project, created_at — never full key)
DELETE /v1/keys/{key_id}  →  revoke key
```

---

## Project / Tenant Model

Every API key is scoped to a **project**. A project is the unit of:
- **Usage metering** — token consumption tracked per project
- **Rate limiting** — quota enforced at project level
- **Billing** — invoiced per project
- **Model access** — model routing can be configured per project
- **Audit** — all requests logged with project identity

```
Organisation
  └── Project A  (api_key_1, api_key_2)
       ├── Usage ledger
       ├── Audit log
       └── Deployment config
  └── Project B  (api_key_3)
```

---

## OpenAI Compatibility

The following endpoints follow the OpenAI API schema to allow zero-friction migration from OpenAI or Azure OpenAI:

| SMTX endpoint | OpenAI equivalent |
|---|---|
| `POST /v1/chat/completions` | `POST /v1/chat/completions` |
| `POST /v1/responses` | `POST /v1/responses` (Responses API) |
| `POST /v1/embeddings` | `POST /v1/embeddings` |
| `GET /v1/models` | `GET /v1/models` |

Where SMTX schemas differ from OpenAI, the differences are documented in schema annotations.

---

## Starter API Schemas

### `POST /v1/chat/completions`

```json
{
  "model": "t-dev-6l",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is sovereign AI?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": false
}
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "t-dev-6l",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Sovereign AI refers to..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 22, "completion_tokens": 64, "total_tokens": 86}
}
```

---

### `POST /v1/responses`

```json
{
  "model": "t-dev-6l",
  "input": "Summarise the key principles of sovereign AI.",
  "instructions": "Be concise and use bullet points.",
  "max_output_tokens": 256,
  "temperature": 0.7
}
```

Response:
```json
{
  "id": "resp-abc123",
  "object": "response",
  "model": "t-dev-6l",
  "output": [{"type": "text", "text": "• British data sovereignty...\n• Provider independence..."}],
  "usage": {"input_tokens": 14, "output_tokens": 48, "total_tokens": 62}
}
```

---

### `POST /v1/embeddings`

```json
{
  "model": "t-dev-6l",
  "input": ["Sovereign AI", "British enterprise LLM"],
  "encoding_format": "float"
}
```

Response:
```json
{
  "object": "list",
  "model": "t-dev-6l",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.021, -0.014, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.009, 0.031, ...]}
  ],
  "usage": {"prompt_tokens": 6, "total_tokens": 6}
}
```

---

### `GET /v1/models`

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "t-dev-6l",
      "object": "model",
      "family": "t-series",
      "variant": "dev-6l",
      "stage": "development",
      "context_length": 2048
    }
  ]
}
```

---

### `GET /v1/usage/{project_id}`

Response:
```json
{
  "project_id": "proj-abc",
  "total_requests": 142,
  "total_prompt_tokens": 28400,
  "total_completion_tokens": 14200,
  "total_tokens": 42600
}
```

---

### `POST /v1/projects`

```json
{
  "name": "Acme Legal Assistant",
  "description": "Internal legal research assistant",
  "owner_email": "admin@acme.co.uk"
}
```

Response:
```json
{
  "id": "proj-abc123",
  "name": "Acme Legal Assistant",
  "description": "Internal legal research assistant",
  "owner_email": "admin@acme.co.uk",
  "created_at": "2025-05-27T12:00:00Z"
}
```

---

### `POST /v1/keys`

Request:
```json
{
  "project_id": "proj-abc123",
  "label": "production-key-1"
}
```

Response (key shown **once only**):
```json
{
  "key_id": "key-xyz789",
  "project_id": "proj-abc123",
  "label": "production-key-1",
  "key": "smtx_abc123...hex64chars",
  "created_at": "2025-05-27T12:00:00Z"
}
```

---

### `GET /v1/audit/events`

Query params: `project_id`, `event_type`, `since`, `limit`

Response:
```json
{
  "events": [
    {
      "id": "evt-001",
      "project_id": "proj-abc123",
      "event_type": "chat.completion",
      "model": "t-dev-6l",
      "timestamp": "2025-05-27T12:01:00Z",
      "metadata": {"prompt_tokens": 22, "completion_tokens": 64}
    }
  ],
  "total": 1
}
```

---

## Design Principles

1. **Stable first**: API paths and response schemas are versioned (`/v1/`). Breaking changes require a new version prefix.
2. **Auth everywhere**: no endpoint (except `/health`) is accessible without a valid API key.
3. **Meter everything**: every inference call records project, model, token counts, and timestamp.
4. **Audit by default**: all write operations and inference calls append to an immutable audit log.
5. **OpenAI-compatible where possible**: reduces migration friction for customers coming from hosted providers.
6. **Fail safe**: invalid or expired keys return `401`; malformed input returns `422` with a schema error; unexpected errors return `500` with a request ID for support correlation.

---

## SDK Generation Path

The FastAPI app auto-generates an OpenAPI 3.1 schema at `/openapi.json`. This can be used directly with:

- **openapi-generator-cli** for Java, Go, TypeScript, Python SDK generation
- **Speakeasy** for production-grade SDKs
- **Postman** collection import

```bash
# Export OpenAPI schema
python scripts/export_openapi.py > docs/openapi.json
```

---

## Rate Limiting (Planned — Phase 5)

Rate limits enforced at project level:

| Tier | Requests/min | Tokens/min |
|---|---|---|
| Free (dev) | 60 | 100 000 |
| Professional | 600 | 1 000 000 |
| Enterprise | Custom | Custom |

---

## Observability (Planned — Phase 5)

- **Request logs**: all inference requests logged with latency, token counts, model version
- **Prometheus metrics**: `smtx_requests_total`, `smtx_tokens_total`, `smtx_latency_seconds`
- **Audit log**: immutable, append-only, queryable via `/v1/audit/events`
- **Alerting**: latency SLO breach, error rate spike, quota exhaustion

---

## What Was Reused from Existing SMTX vs. Newly Introduced

| Component | Decision | Notes |
|---|---|---|
| `api/` (legacy top-level FastAPI) | **Retained** | Original governance API; preserved for continuity |
| `api/routers/` (legacy) | **Retained** | Legacy router definitions; not extended |
| `src/supreme_modeltx/platform_api/` | **All new** | Canonical business API going forward |
| `api/schemas.py` (legacy) | **Retained** | Legacy schema definitions; new schemas live in `platform_api/api/schemas.py` |
| `control-plane/` (C# ASP.NET Core) | **Retained as-is** | Governance control plane |
| `infra/` (Bicep) | **Retained as-is** | Infrastructure; new API deploys alongside |
