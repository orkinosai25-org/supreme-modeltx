"""
inference_service.py — SMTX Stateless Inference Service Skeleton

Step-3 scaffold for a reloadable, independently deployable inference service.

Responsibilities:
- load active model checkpoint (dummy placeholder in this step)
- serve inference requests
- handle control-plane reload intents

Out of scope:
- training, retrieval/memory, scheduling, metadata persistence
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.inference")


@dataclass(slots=True)
class DummyModel:
    checkpoint_path: str
    loaded_at_utc: datetime


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt.")
    max_tokens: int = Field(128, ge=1, le=4096)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int


class ReloadRequest(BaseModel):
    checkpoint_path: str | None = Field(None, description="Optional override for active checkpoint path.")


class ReloadResponse(BaseModel):
    status: str
    checkpoint_path: str
    reloaded_at_utc: str


_model_lock = Lock()
_model: DummyModel | None = None
_reload_timestamp_utc: datetime | None = None
_checkpoint_path = os.environ.get("MODEL_PATH", "stub://t101")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_model(checkpoint_path: str) -> None:
    global _model
    _model = DummyModel(checkpoint_path=checkpoint_path, loaded_at_utc=_utc_now())
    logger.info("model loaded: checkpoint=%s", checkpoint_path)


def _unload_model() -> None:
    global _model
    if _model is not None:
        logger.info("model unloaded: checkpoint=%s", _model.checkpoint_path)
    _model = None


def _reload_model(checkpoint_path: str | None = None) -> tuple[str, datetime]:
    global _checkpoint_path, _reload_timestamp_utc

    with _model_lock:
        if checkpoint_path:
            _checkpoint_path = checkpoint_path

        _unload_model()
        _load_model(_checkpoint_path)

        _reload_timestamp_utc = _utc_now()
        logger.info("reload timestamp: %s", _reload_timestamp_utc.isoformat())
        return _checkpoint_path, _reload_timestamp_utc


@asynccontextmanager
async def lifespan(app: FastAPI):
    _reload_model(_checkpoint_path)
    yield
    _unload_model()


app = FastAPI(title="SMTX Inference Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    is_ready = _model is not None
    return {
        "status": "ok" if is_ready else "loading",
        "ready": is_ready,
        "checkpoint_path": _model.checkpoint_path if _model else None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    current_model = _model
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Step-3 placeholder output to keep service verifiable without ML runtime dependency.
    # Placeholder implementation: `max_tokens` is treated as an approximate
    # character budget in this scaffold (real model tokenization is out of scope).
    completion = f"[{current_model.checkpoint_path}] generated for: {req.prompt}"[: req.max_tokens]

    return GenerateResponse(
        text=completion,
        # Step-3 approximation only: production inference should use tokenizer-based counts.
        prompt_tokens=len(req.prompt.split()),
        completion_tokens=max(len(completion.split()), 1),
    )


@app.post("/reload", status_code=202, response_model=ReloadResponse)
def reload_endpoint(req: ReloadRequest | None = Body(default=None)) -> ReloadResponse:
    checkpoint_path, reloaded_at = _reload_model(req.checkpoint_path if req else None)

    return ReloadResponse(
        status="reload_accepted",
        checkpoint_path=checkpoint_path,
        reloaded_at_utc=reloaded_at.isoformat(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMTX stateless inference service skeleton.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-path",
        type=str,
        default=_checkpoint_path,
        help="Path or identifier for active checkpoint (stub value allowed).",
    )
    return parser.parse_args()


def main() -> None:
    global _checkpoint_path

    args = parse_args()
    _checkpoint_path = args.model_path

    logger.info("Starting inference service on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
