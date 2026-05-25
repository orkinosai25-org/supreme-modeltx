"""
verification_service.py — SMTX T-501 Verification Microservice

Verifies and scores model responses for factual consistency on port 8002.

Usage:
    python inference/verification_service.py [--port 8002]

Requirements:
    pip install sentence-transformers fastapi uvicorn pydantic
"""

import argparse
import logging
import sys
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.verification")

app = FastAPI(title="SMTX T-501 Verification Service", version="0.1.0")

_model = None


# ── Request / Response schemas ────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    claim: str = Field(..., description="The claim / model response to verify.")
    evidence: List[str] = Field(..., description="Supporting evidence passages.")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score to consider supported.")


class VerifyResponse(BaseModel):
    supported: bool
    confidence: float
    best_evidence: Optional[str] = None
    scores: List[float]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _model is not None else "loading"}


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest) -> VerifyResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    import numpy as np

    claim_vec = _model.encode([req.claim], normalize_embeddings=True)
    evidence_vecs = _model.encode(req.evidence, normalize_embeddings=True)

    # Cosine similarity (vectors are unit-normalised)
    scores = (evidence_vecs @ claim_vec.T).flatten().tolist()

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    supported = best_score >= req.threshold

    return VerifyResponse(
        supported=supported,
        confidence=best_score,
        best_evidence=req.evidence[best_idx] if supported else None,
        scores=scores,
    )


# ── Bootstrap ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMTX T-501 Verification Service")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global _model
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model: %s", args.model_name)
    _model = SentenceTransformer(args.model_name)
    logger.info("Verification model loaded.")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
