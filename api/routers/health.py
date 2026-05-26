"""
routers/health.py — GET /v1/health
"""

from __future__ import annotations

import logging
import os
import time
from typing import List

import httpx
from fastapi import APIRouter

from api.schemas import HealthResponse, ServiceHealth

logger = logging.getLogger("smtx.api.health")

router = APIRouter(tags=["Health"])

_API_VERSION = "0.1.0"

_DOWNSTREAM: List[tuple[str, str]] = [
    ("inference", f"http://{os.environ.get('GPU_VM_IP', '10.0.1.4')}:{os.environ.get('INFERENCE_PORT', '8000')}"),
    ("retrieval", f"http://{os.environ.get('CPU_VM_IP', '10.0.2.4')}:{os.environ.get('RETRIEVAL_PORT', '8001')}"),
    ("verification", f"http://{os.environ.get('CPU_VM_IP', '10.0.2.4')}:{os.environ.get('VERIFICATION_PORT', '8002')}"),
]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Platform health check",
    description=(
        "Returns the overall platform status and the health of each downstream "
        "service (inference, retrieval, verification).  Suitable for Azure health "
        "probes, Kubernetes readiness checks, and uptime monitoring."
    ),
)
async def health() -> HealthResponse:
    services: List[ServiceHealth] = [ServiceHealth(name="api", status="ok", latency_ms=0.0)]

    async with httpx.AsyncClient(timeout=5) as client:
        for name, base_url in _DOWNSTREAM:
            t0 = time.monotonic()
            try:
                r = await client.get(f"{base_url}/health")
                latency_ms = (time.monotonic() - t0) * 1000
                svc_status = (
                    r.json().get("status", "unknown") if r.status_code == 200 else "degraded"
                )
            except Exception:
                latency_ms = None
                svc_status = "unreachable"
            services.append(ServiceHealth(name=name, status=svc_status, latency_ms=latency_ms))

    overall = "ok" if all(s.status == "ok" for s in services) else "degraded"
    return HealthResponse(status=overall, version=_API_VERSION, services=services)
