"""
cpu_inference_server.py — SMTX CPU Inference Server

Default inference backend using PyTorch + HuggingFace Transformers.
No GPU required. Provides the same /health and /generate REST API
as vllm_server.py so the orchestrator can route to either backend
without changing its call-site.

This is the authoritative default inference endpoint for SUMOTX.
vLLM is an optional accelerator; this server runs on CPU.

Usage:
    python inference/cpu_inference_server.py --model tmodels/t101 --port 8003

Requirements:
    pip install torch transformers fastapi uvicorn pydantic
"""

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.cpu_inference")

# ── Configuration constants ───────────────────────────────────────────────────

# Maximum number of new tokens the /generate endpoint will produce.
# Capped here to protect CPU resources; the per-request max_tokens is
# further bounded by GenerateRequest.max_tokens (max=4096).
MAX_NEW_TOKENS_LIMIT = 4096

# ── Model path (resolved from env or CLI in main()) ───────────────────────────
# Module-level so the lifespan hook can access it before main() runs.
# Updated by main() from the --model CLI flag when provided.
_model_path: str = os.environ.get("MODEL_PATH", "tmodels/t101")

# Module-level state — populated during the FastAPI lifespan startup hook,
# which runs before the first request is processed.
#
# ⚠️  Single-worker constraint: this server is intended to run as a single
# uvicorn worker process (the default, without --workers N > 1).  Launching
# multiple workers would create separate model copies per process, which is
# memory-inefficient for large models.  If horizontal scaling is required,
# deploy additional single-worker instances behind a load balancer instead.
_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None


# ── Lifespan: load model before serving requests ──────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model and tokenizer at startup; nothing to clean up on shutdown."""
    global _model, _tokenizer

    logger.info("Loading tokenizer from: %s", _model_path)
    _tokenizer = AutoTokenizer.from_pretrained(_model_path, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    logger.info("Loading model from: %s (CPU)", _model_path)
    _model = AutoModelForCausalLM.from_pretrained(
        _model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    _model.eval()
    logger.info(
        "Model loaded — parameters: %s",
        f"{sum(p.numel() for p in _model.parameters()):,}",
    )

    yield


app = FastAPI(title="SMTX CPU Inference Server", version="0.1.0", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt.")
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    ready = _model is not None and _tokenizer is not None
    return {"status": "ok" if ready else "loading", "backend": "cpu"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    inputs = _tokenizer(
        req.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    prompt_token_count = inputs["input_ids"].shape[-1]

    # Greedy / sampling generation on CPU
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=min(req.max_tokens, MAX_NEW_TOKENS_LIMIT),
            do_sample=req.temperature > 0.0,
            temperature=req.temperature if req.temperature > 0.0 else 1.0,
            top_p=req.top_p,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_ids = output_ids[0][prompt_token_count:]
    completion_text = _tokenizer.decode(new_ids, skip_special_tokens=True)
    completion_token_count = new_ids.shape[-1]

    return GenerateResponse(
        text=completion_text,
        prompt_tokens=prompt_token_count,
        completion_tokens=completion_token_count,
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMTX CPU inference server.")
    parser.add_argument(
        "--model",
        type=str,
        default=_model_path,
        help="Path to the model directory or HuggingFace model ID.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    return parser.parse_args()


def main() -> None:
    global _model_path
    args = parse_args()
    # Allow --model CLI flag to override the env-var default.
    _model_path = args.model

    logger.info("Starting CPU inference server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
