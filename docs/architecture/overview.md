# Architecture Overview

`supreme-modeltx` is a **British sovereign AI platform** that combines a PyTorch-native trainable LLM engine with an API-first business platform for secure, governable, high-value enterprise AI applications.

## High-Level Structure

```text
supreme-modeltx/
├── src/supreme_modeltx/
│   ├── model_core/          # Sovereign LLM engine
│   │   ├── config.py        # Pydantic config schema (ModelConfig, TrainingConfig)
│   │   ├── models/          # T-Series transformer family
│   │   ├── training/        # Training loop (Trainer)
│   │   ├── data/            # Reproducible data pipeline
│   │   ├── tokenizer/       # BPE tokenizer wrapper
│   │   ├── eval/            # Evaluation harness (perplexity, accuracy)
│   │   └── inference/       # Local inference engine (greedy, sampling)
│   ├── platform_api/        # API-first business platform
│   │   ├── api/             # FastAPI app factory
│   │   ├── auth/            # API key management
│   │   ├── tenants/         # Tenant / project management
│   │   ├── usage/           # Token metering
│   │   ├── model_registry/  # Model version catalogue
│   │   └── deployment/      # Deployment record management
│   └── utils/               # Shared utilities (device selection, logging)
├── tests/                   # Pytest test suite
├── control-plane/           # .NET control plane (Blazor + API)
├── infra/                   # Azure Bicep infrastructure
├── scripts/                 # Dockerfiles, run scripts
├── docs/                    # Documentation
└── pyproject.toml           # Python package metadata
```

## Two First-Class Domains

### A. `model_core` — Sovereign LLM Engine

The model core is a self-contained PyTorch training and inference stack:

- **T-Series model family** — dense decoder-only transformers (T101 → T501 → T-X)
- **RoPE positional embeddings** and **SwiGLU FFN** for modern performance
- **RMSNorm** for training stability
- **Reproducible data pipeline** with JSONL, plain-text, and HuggingFace dataset support
- **Trainer** compatible with Accelerate and DeepSpeed for GPU scaling
- **EvalHarness** for perplexity and accuracy measurement
- **InferenceEngine** with greedy and nucleus (top-p) sampling

### B. `platform_api` — Business API Platform

The platform API exposes an OpenAI-compatible surface for enterprise integration:

- `POST /v1/chat/completions` — chat generation
- `GET /v1/models` — list available model versions
- `GET /v1/usage` — token usage accounting
- `POST /v1/keys` — API key provisioning

All endpoints are authenticated via bearer API keys, with tenant-scoped metering.

## Sovereignty Principles

See [sovereignty/principles.md](../sovereignty/principles.md) for the full statement.

Key commitments:

1. **UK-owned infrastructure** — all training, inference, and data stays within sovereign boundaries
2. **Auditable** — every API call and deployment decision is logged
3. **Provider-independent** — no hard dependency on any single cloud provider's AI services
4. **Portable** — model weights, tokenizers, and checkpoints use open formats
5. **Governable** — policy-gated deployment and approval workflows

## Technology Choices

| Layer | Technology | Rationale |
|---|---|---|
| Model framework | PyTorch | De facto research standard; maximum control |
| Training acceleration | Accelerate / DeepSpeed | GPU-scaling without framework lock-in |
| Tokenizer | HuggingFace `tokenizers` | Fast BPE with custom vocabulary support |
| API framework | FastAPI + Pydantic v2 | Type-safe, OpenAPI-native |
| Infrastructure | Azure Bicep | UK-region-capable, sovereign-aligned |
| Control plane | ASP.NET Core + Blazor | Enterprise-grade .NET governance surface |
