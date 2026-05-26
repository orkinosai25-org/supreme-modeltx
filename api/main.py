"""
main.py — SUMOTX Platform API (v0.1)

Entry point for the SUMOTX platform API.  Swagger UI is available at /docs,
ReDoc at /redoc, and the raw OpenAPI JSON spec at /openapi.json.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 9000 --reload

Environment variables:
    SUMOTX_API_KEY      — Bearer token accepted by all protected endpoints
                          (default: "dev-secret" — change in production!)
    GPU_VM_IP           — Private IP of GPU VM  (default: 10.0.1.4)
    CPU_VM_IP           — Private IP of CPU VM  (default: 10.0.2.4)
    INFERENCE_PORT      — vLLM inference port    (default: 8000)
    RETRIEVAL_PORT      — T-301 retrieval port   (default: 8001)
    VERIFICATION_PORT   — T-501 verification port (default: 8002)
    ORCHESTRATOR_URL    — T-X orchestrator URL   (default: http://localhost:8080)
    HTTP_TIMEOUT        — Downstream HTTP timeout in seconds (default: 120)
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.routers import auth, chat, deployments, governed_models, health, models_registry, projects, training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.api")

# ── OpenAPI metadata ──────────────────────────────────────────────────────────

_DESCRIPTION = """
## SUMOTX Platform API — v0.1

**SUMOTX** exposes a versioned OpenAPI interface for provisioning, training,
and managing AI workloads on Azure.

### Authentication

All endpoints (except `GET /v1/health` and `POST /v1/auth/token`) require a
**Bearer token**.  Exchange your API key for a token via `POST /v1/auth/token`,
then include it as:

```
Authorization: Bearer <token>
```

### Versioning

All endpoints are prefixed with `/v1`.  Breaking changes will be published
under a new version prefix.

### API domains

| Domain | Description |
|---|---|
| `/v1/auth` | Token issuance |
| `/v1/projects` | Project / workspace management |
| `/v1/deployments` | One-click Azure infrastructure provisioning |
| `/v1/training-runs` | Fine-tuning job lifecycle |
| `/v1/models` | Model registry |
| `/v1/governed-models` | Approved model listing (policy-filtered) |
| `/v1/model-assignments` | Governed model + data-source/indexing/RAG assignment and audit trail |
| `/v1/chat` | Inference / chat completions |
| `/v1/health` | Platform health & readiness |
"""

app = FastAPI(
    title="SUMOTX Platform API",
    version="0.1.0",
    description=_DESCRIPTION,
    contact={
        "name": "SUMOTX Engineering",
        "url": "https://github.com/orkinosai25-org/SMTX",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {"name": "Health", "description": "Platform health and readiness probes."},
        {"name": "Authentication", "description": "API key → Bearer token exchange."},
        {"name": "Projects", "description": "Project / workspace management."},
        {
            "name": "Deployments",
            "description": "One-click Azure infrastructure provisioning and lifecycle.",
        },
        {
            "name": "Training",
            "description": "Fine-tuning job submission and monitoring.",
        },
        {
            "name": "Models",
            "description": "Model registry — list, inspect, and promote models.",
        },
        {
            "name": "Governed Models",
            "description": (
                "Enterprise governed model assignment — list approved models, "
                "assign models with linked data-source/indexing/RAG policy, and review the audit trail."
            ),
        },
        {
            "name": "Chat",
            "description": "Inference and chat-completion endpoints backed by the T-X pipeline.",
        },
    ],
    # Security scheme definition (Bearer / API key)
    swagger_ui_parameters={"persistAuthorization": True},
)

# ── Security scheme ───────────────────────────────────────────────────────────

# Injected into OpenAPI spec so Swagger UI shows the padlock on protected ops
from fastapi.openapi.utils import get_openapi  # noqa: E402


def _custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        contact=app.contact,
        license_info=app.license_info,
        tags=app.openapi_tags,
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "API key",
        "description": (
            "Obtain a token from `POST /v1/auth/token` and supply it here.  "
            "In production, rotate the underlying `SUMOTX_API_KEY` environment variable."
        ),
    }
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi  # type: ignore[method-assign]

# ── Routers ───────────────────────────────────────────────────────────────────

_V1 = "/v1"

app.include_router(health.router, prefix=_V1)
app.include_router(auth.router, prefix=_V1)
app.include_router(projects.router, prefix=_V1)
app.include_router(deployments.router, prefix=_V1)
app.include_router(training.router, prefix=_V1)
app.include_router(models_registry.router, prefix=_V1)
app.include_router(governed_models.router, prefix=_V1)
app.include_router(chat.router, prefix=_V1)

# ── Root redirect ─────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    """Redirect browsers to Swagger UI."""
    return JSONResponse(
        content={"message": "SUMOTX Platform API", "docs": "/docs", "version": "0.1.0"}
    )
