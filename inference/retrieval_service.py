"""
retrieval_service.py — SMTX T-301 Retrieval Microservice

Serves dense retrieval via FAISS + sentence-transformers on port 8001.

Usage:
    python inference/retrieval_service.py [--index-path /mnt/index] [--port 8001]

Requirements:
    pip install faiss-cpu sentence-transformers fastapi uvicorn pydantic
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.retrieval")

app = FastAPI(title="SMTX T-301 Retrieval Service", version="0.1.0")

# Module-level state (populated in main())
_index = None
_texts: List[str] = []
_model = None


# ── Request / Response schemas ────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural-language query to retrieve against.")
    top_k: int = Field(5, ge=1, le=100, description="Number of top results to return.")


class RetrieveResult(BaseModel):
    text: str
    score: float


class RetrieveResponse(BaseModel):
    results: List[RetrieveResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    ready = _index is not None and _model is not None
    return {"status": "ok" if ready else "loading", "index_size": len(_texts)}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    if _index is None or _model is None:
        raise HTTPException(status_code=503, detail="Index not loaded yet.")

    query_vec = _model.encode([req.query], normalize_embeddings=True).astype("float32")
    distances, indices = _index.search(query_vec, req.top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_texts):
            continue
        results.append(RetrieveResult(text=_texts[idx], score=float(dist)))
    return RetrieveResponse(results=results)


# ── Bootstrap ─────────────────────────────────────────────────────────────────


def load_index(index_path: str, model_name: str) -> None:
    """Load FAISS index and sentence-transformer model into module globals."""
    global _index, _texts, _model

    import faiss
    from sentence_transformers import SentenceTransformer

    index_file = Path(index_path) / "faiss.index"
    texts_file = Path(index_path) / "texts.txt"

    if index_file.exists() and texts_file.exists():
        logger.info("Loading FAISS index from %s", index_file)
        _index = faiss.read_index(str(index_file))
        _texts = texts_file.read_text(encoding="utf-8").splitlines()
        logger.info("Index loaded: %d vectors, %d texts", _index.ntotal, len(_texts))
    else:
        logger.warning(
            "No FAISS index found at %s — creating empty flat index (dim=384). "
            "Populate via /index endpoint or restart with a valid index path.",
            index_path,
        )
        _index = faiss.IndexFlatIP(384)
        _texts = []

    logger.info("Loading sentence-transformer model: %s", model_name)
    _model = SentenceTransformer(model_name)
    logger.info("Model loaded.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMTX T-301 Retrieval Service")
    parser.add_argument("--index-path", type=str, default=os.environ.get("INDEX_PATH", "/mnt/index"))
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_index(args.index_path, args.model_name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
