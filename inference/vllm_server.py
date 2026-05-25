"""
vllm_server.py — SMTX vLLM Inference Server

Optional GPU-accelerated inference backend using vLLM.

This server is NOT the default inference backend.
The default is inference/cpu_inference_server.py (CPU, no GPU required).

vLLM is an optional accelerator — it is only started when:
  - A CUDA-capable VM (NC / ND series) is available
  - VLLM_ENABLED=true is set in the orchestrator environment

See inference/backends.yml for the full backend capability registry.

Usage:
    python inference/vllm_server.py --model tmodels/t101 --port 8000

Requirements:
    pip install vllm fastapi uvicorn
"""

import argparse
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMTX vLLM inference server.")
    parser.add_argument(
        "--model",
        type=str,
        default="tmodels/t101",
        help="Path to the model directory or Hugging Face model ID.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", None],
        help="Optional quantization method.",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    return parser.parse_args()


app = FastAPI(title="SMTX vLLM Server", version="0.1.0")
llm: LLM | None = None


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop if req.stop else None,
    )
    outputs = llm.generate([req.prompt], sampling_params)
    output = outputs[0]
    completion_text = output.outputs[0].text

    return GenerateResponse(
        text=completion_text,
        prompt_tokens=len(output.prompt_token_ids),
        completion_tokens=len(output.outputs[0].token_ids),
    )


def main() -> None:
    args = parse_args()

    # Guard: vLLM requires a CUDA-capable GPU.  Exiting here instead of
    # crashing inside vLLM gives a clear, actionable error message and
    # prevents the service from entering a crash-restart loop on CPU VMs.
    try:
        import torch
        if not torch.cuda.is_available():
            print(
                "[ERROR] vLLM requires a CUDA-capable GPU but none was detected.\n"
                "        Use the CPU inference server (inference/cpu_inference_server.py) "
                "on CPU-only machines.\n"
                "        Set VLLM_ENABLED=true only when GPU quota is available.",
                file=sys.stderr,
            )
            sys.exit(1)
    except ImportError:
        pass  # torch not installed — let vLLM handle it

    global llm
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
