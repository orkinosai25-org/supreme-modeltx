# SUMOTX Platform API

The `api/` module exposes the SUMOTX Platform as a versioned OpenAPI service
backed by FastAPI.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server (default port 9000)
uvicorn api.main:app --host 0.0.0.0 --port 9000 --reload
```

Open **Swagger UI** at [http://localhost:9000/docs](http://localhost:9000/docs)  
Open **ReDoc** at [http://localhost:9000/redoc](http://localhost:9000/redoc)  
Raw OpenAPI JSON: [http://localhost:9000/openapi.json](http://localhost:9000/openapi.json)

---

## Authentication

All endpoints except `GET /v1/health` and `POST /v1/auth/token` require a
**Bearer token**.

### 1. Exchange your API key for a token

```bash
curl -X POST http://localhost:9000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "dev-secret"}'
```

Response:

```json
{
  "access_token": "dev-secret",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 2. Use the token in subsequent requests

```bash
curl http://localhost:9000/v1/projects \
  -H "Authorization: Bearer dev-secret"
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SUMOTX_API_KEY` | `dev-secret` | Shared secret accepted by all protected endpoints. **Change in production.** |
| `GPU_VM_IP` | `10.0.1.4` | Private IP of the GPU VM (vLLM inference) |
| `CPU_VM_IP` | `10.0.2.4` | Private IP of the CPU VM (retrieval + verification) |
| `INFERENCE_PORT` | `8000` | vLLM inference port |
| `RETRIEVAL_PORT` | `8001` | T-301 retrieval port |
| `VERIFICATION_PORT` | `8002` | T-501 verification port |
| `ORCHESTRATOR_URL` | `http://localhost:8080` | T-X orchestrator URL |
| `HTTP_TIMEOUT` | `120` | Downstream HTTP timeout (seconds) |

---

## API surface (v0.1)

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/health` | ❌ | Platform health & readiness |
| `POST` | `/v1/auth/token` | ❌ | Exchange API key for Bearer token |
| `GET` | `/v1/projects` | ✅ | List projects |
| `POST` | `/v1/projects` | ✅ | Create a project |
| `GET` | `/v1/projects/{id}` | ✅ | Get a project |
| `POST` | `/v1/deployments` | ✅ | Trigger one-click Azure deployment |
| `GET` | `/v1/deployments/{id}/status` | ✅ | Get deployment status |
| `POST` | `/v1/training-runs` | ✅ | Submit a training run |
| `GET` | `/v1/training-runs` | ✅ | List training runs |
| `GET` | `/v1/training-runs/{id}` | ✅ | Get a training run |
| `GET` | `/v1/training-runs/{id}/logs` | ✅ | Get training run logs |
| `GET` | `/v1/models` | ✅ | List models |
| `GET` | `/v1/models/{id}` | ✅ | Get a model |
| `POST` | `/v1/models/{id}/promote` | ✅ | Promote a model to a new stage |
| `POST` | `/v1/chat/completions` | ✅ | Chat completion via T-X pipeline |

---

## Static OpenAPI spec

The spec is exported to [`docs/openapi.yaml`](../docs/openapi.yaml) and
[`docs/openapi.json`](../docs/openapi.json) for tooling that needs an offline
copy.

To regenerate:

```bash
python scripts/export_openapi.py
```

---

## Versioning strategy

All endpoints live under `/v1`.  Breaking changes are published under a new
prefix (`/v2`, etc.).  Non-breaking additions (new optional fields, new
endpoints) may be added within the same version.
