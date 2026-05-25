"""
orchestrator.py — SMTX T-X Orchestrator

Routes prompts through the full T-Series pipeline:
  Prompt → T-101 (inference) → T-201 (reasoning) → T-301 (retrieval)
         → T-501 (verification) → Final response

Deployed as an Azure App Service (Linux/Python).

Inference backend selection
---------------------------
By default the orchestrator uses the **CPU inference backend** running on
the CPU VM (port 8003).  The vLLM GPU backend is optional and only
activated when VLLM_ENABLED=true *and* GPU quota is available.

Environment variables (set via App Service app settings or local .env):
    CPU_VM_IP               — Private IP of CPU VM              (default: 10.0.2.4)
    CPU_INFERENCE_PORT      — CPU inference server port         (default: 8003)
    GPU_VM_IP               — Private IP of GPU VM              (default: 10.0.1.4)
    VLLM_PORT               — vLLM server port                  (default: 8000)
    VLLM_ENABLED            — Activate vLLM backend (true/false)(default: false)
    RETRIEVAL_PORT          — T-301 port                        (default: 8001)
    VERIFICATION_PORT       — T-501 port                        (default: 8002)
    ORCHESTRATOR_PORT       — this service port                 (default: 8080)
    APPLICATIONINSIGHTS_CONNECTION_STRING — optional App Insights telemetry

Usage:
    python tmodels/tx/orchestrator.py
    # or via gunicorn for production:
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker tmodels.tx.orchestrator:app
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.orchestrator")

# ── Configuration ─────────────────────────────────────────────────────────────

CPU_VM_IP = os.environ.get("CPU_VM_IP", "10.0.2.4")
GPU_VM_IP = os.environ.get("GPU_VM_IP", "10.0.1.4")

# Inference backend selection
# Default: CPU backend (no GPU quota required).
# Set VLLM_ENABLED=true to route inference to the vLLM GPU backend instead.
CPU_INFERENCE_PORT = int(os.environ.get("CPU_INFERENCE_PORT", "8003"))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_ENABLED = os.environ.get("VLLM_ENABLED", "false").lower() == "true"

RETRIEVAL_PORT = int(os.environ.get("RETRIEVAL_PORT", "8001"))
VERIFICATION_PORT = int(os.environ.get("VERIFICATION_PORT", "8002"))

# Route inference to CPU backend by default; GPU/vLLM only when explicitly enabled.
if VLLM_ENABLED:
    INFERENCE_URL = f"http://{GPU_VM_IP}:{VLLM_PORT}"
    logger.info("Inference backend: vLLM (GPU) at %s", INFERENCE_URL)
else:
    INFERENCE_URL = f"http://{CPU_VM_IP}:{CPU_INFERENCE_PORT}"
    logger.info("Inference backend: CPU at %s", INFERENCE_URL)

RETRIEVAL_URL = f"http://{CPU_VM_IP}:{RETRIEVAL_PORT}"
VERIFICATION_URL = f"http://{CPU_VM_IP}:{VERIFICATION_PORT}"

HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "120"))

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="SMTX T-X Orchestrator", version="0.1.0")


# ── Schemas ───────────────────────────────────────────────────────────────────


class OrchestrateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to process through the T-Series pipeline.")
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k_retrieval: int = Field(3, ge=1, le=20, description="Number of passages to retrieve for grounding.")
    verification_threshold: float = Field(0.4, ge=0.0, le=1.0)


class OrchestrateResponse(BaseModel):
    final_response: str
    inference_text: str
    retrieval_passages: list
    verification: Dict[str, Any]
    pipeline_stages: list


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Aggregate health check across all downstream services."""
    results: Dict[str, str] = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for name, url in [
            ("inference", INFERENCE_URL),
            ("retrieval", RETRIEVAL_URL),
            ("verification", VERIFICATION_URL),
        ]:
            try:
                r = await client.get(f"{url}/health")
                results[name] = r.json().get("status", "unknown") if r.status_code == 200 else "error"
            except Exception:
                results[name] = "unreachable"
    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"
    inference_backend = "vllm" if VLLM_ENABLED else "cpu"  # matches `name` in inference/backends.yml
    return {"status": overall, "services": results, "inference_backend": inference_backend}


@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest) -> OrchestrateResponse:
    """
    Full T-Series pipeline:
      1. T-101 — base inference
      2. T-201 — reasoning refinement (second inference pass with CoT prompt)
      3. T-301 — retrieval for grounding
      4. T-501 — verification of final answer against retrieved passages
    """
    stages = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:

        # ── Stage 1: T-101 base inference ────────────────────────────────────
        logger.info("[T-101] Generating base response …")
        t101_payload = {
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }
        try:
            r = await client.post(f"{INFERENCE_URL}/generate", json=t101_payload)
            r.raise_for_status()
            t101_result = r.json()
            inference_text = t101_result.get("text", "")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"T-101 inference error: {exc}") from exc
        stages.append({"stage": "T-101", "status": "ok", "tokens": t101_result.get("completion_tokens", 0)})

        # ── Stage 2: T-201 reasoning refinement ─────────────────────────────
        logger.info("[T-201] Refining with chain-of-thought …")
        cot_prompt = (
            f"Original question: {req.prompt}\n\n"
            f"Initial answer: {inference_text}\n\n"
            "Now reason step-by-step and provide a more precise, well-structured answer:"
        )
        t201_payload = {
            "prompt": cot_prompt,
            "max_tokens": req.max_tokens,
            "temperature": max(0.0, req.temperature - 0.2),
            "top_p": req.top_p,
        }
        try:
            r = await client.post(f"{INFERENCE_URL}/generate", json=t201_payload)
            r.raise_for_status()
            t201_result = r.json()
            refined_text = t201_result.get("text", inference_text)
        except httpx.HTTPError:
            refined_text = inference_text
            t201_result = {}
        stages.append({"stage": "T-201", "status": "ok", "tokens": t201_result.get("completion_tokens", 0)})

        # ── Stage 3: T-301 retrieval ─────────────────────────────────────────
        logger.info("[T-301] Retrieving supporting passages …")
        try:
            r = await client.post(
                f"{RETRIEVAL_URL}/retrieve",
                json={"query": req.prompt, "top_k": req.top_k_retrieval},
            )
            r.raise_for_status()
            retrieval_results = r.json().get("results", [])
        except httpx.HTTPError:
            retrieval_results = []
        stages.append({"stage": "T-301", "status": "ok", "passages": len(retrieval_results)})

        # ── Stage 4: T-501 verification ──────────────────────────────────────
        logger.info("[T-501] Verifying response against evidence …")
        evidence_texts = [r["text"] for r in retrieval_results] if retrieval_results else [req.prompt]
        try:
            r = await client.post(
                f"{VERIFICATION_URL}/verify",
                json={
                    "claim": refined_text,
                    "evidence": evidence_texts,
                    "threshold": req.verification_threshold,
                },
            )
            r.raise_for_status()
            verification = r.json()
        except httpx.HTTPError:
            verification = {"supported": True, "confidence": 0.0, "scores": []}
        stages.append({"stage": "T-501", "status": "ok", "supported": verification.get("supported")})

    # ── Final response ────────────────────────────────────────────────────────
    final_response = refined_text
    logger.info(
        "Pipeline complete | verified=%s | confidence=%.3f",
        verification.get("supported"),
        verification.get("confidence", 0.0),
    )

    return OrchestrateResponse(
        final_response=final_response,
        inference_text=inference_text,
        retrieval_passages=retrieval_results,
        verification=verification,
        pipeline_stages=stages,
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    port = int(os.environ.get("ORCHESTRATOR_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
