# Supreme ModelTX

**Supreme ModelTX** is a British sovereign LLM platform — a PyTorch-first model development stack paired with an API-first business access layer.

> **Sovereign AI, designed and built in Britain.**  
> Our platform gives UK organisations a path to owning their AI infrastructure end-to-end: from pretraining data and tokenisation through to governed deployment and auditable usage metering.

---

## Two first-class layers

```
supreme-modeltx/
├── src/supreme_modeltx/
│   ├── model_core/       ← PyTorch-native LLM development
│   └── platform_api/     ← API-first business access
```

### 1. `model_core` — Model Development Stack

| Module | Purpose |
|---|---|
| `config/` | Pydantic schema for model, training, data, and tokenizer settings |
| `models/t_series/` | T-series decoder-only transformers (T-Dev-6L baseline today; T-101 roadmap) |
| `models/common/` | RoPE, GQA attention, RMSNorm, SwiGLU building blocks |
| `training/` | Training loop, checkpoint, optimizer, scheduler, mixed precision, distributed |
| `data/` | Manifest-driven JSONL/Parquet/HF Datasets ingestion with sequence packing |
| `tokenizer/` | SentencePiece-oriented tokenizer workflow boundary |
| `eval/` | Perplexity, validation hooks |
| `inference/` | Checkpoint loading, autoregressive generation, nucleus/top-k sampling |

### 2. `platform_api` — Business API Platform

| Module | Purpose |
|---|---|
| `api/` | FastAPI application with OpenAPI docs (`/docs`) |
| `auth/` | API key issuance and verification |
| `tenants/` | Project / tenant management |
| `usage/` | Token usage metering and rate-limit scaffolding |
| `model_registry/` | Model catalogue with stage tracking |
| `deployment/` | Deployment lifecycle management |

---

## Quick start

### Install

```bash
# Core (model_core only):
pip install -e ".[train]"

# API platform:
pip install -e ".[api]"

# Development (everything + tests):
pip install -e ".[train,api,eval,dev]"
```

### Smoke test — model instantiation and tiny forward pass

```bash
python -m pytest tests/smoke/ -v
```

### Smoke test — 2-step CPU training dry run

```bash
python -m supreme_modeltx.model_core.training.trainer --dry-run
```

### Run the platform API

```bash
SMTX_API_KEY=my-key uvicorn supreme_modeltx.platform_api.api.app:app --port 9000 --reload
# Docs at: http://localhost:9000/docs
```

### Run all tests

```bash
python -m pytest tests/ -v
```

---

## Architecture

See [`docs/architecture/overview.md`](docs/architecture/overview.md) for a full description of the two-layer architecture.

See [`docs/sovereignty/principles.md`](docs/sovereignty/principles.md) for the sovereignty design principles.

---

## T-series model family

| Model | Status | Params | Context |
|---|---|---|---|
| **T-Dev-6L** | ✅ Scaffold complete, CPU smoke-testable | ~58M | 512 |
| **T-101** | 🔜 Architecture designed, training pending GPU allocation | 7B | 4096 |
| T-201 (reasoning) | 🗺 Roadmap | — | — |
| T-301 (retrieval) | 🗺 Roadmap | — | — |

---

## Repository structure

```
supreme-modeltx/
├── src/supreme_modeltx/          ← main Python package
│   ├── model_core/               ← LLM development layer
│   └── platform_api/             ← business API layer
├── tests/
│   ├── unit/                     ← config & schema unit tests
│   └── smoke/                    ← model instantiation & forward-pass smoke tests
├── docs/
│   ├── architecture/             ← architecture docs
│   └── sovereignty/              ← sovereignty principles
├── data/raw/                     ← sample pretraining data
├── control-plane/                ← C# ASP.NET Core governance control plane (retained)
├── infra/                        ← Bicep infrastructure definitions
├── pyproject.toml                ← package metadata and dependencies
├── THIRD_PARTY_NOTICES.md        ← open-source provenance
└── .github/workflows/            ← CI workflows
```

---

## Development

This repository is at **scaffold stage**: the architecture and module boundaries are established, core primitives are implemented and tested, and the design is ready for GPU-backed training experiments.

What is **not yet** in this repository:
- Trained model weights (pending GPU allocation)
- Full SentencePiece tokenizer trained on sovereign corpus
- Production-ready distributed training at scale (FSDP/DeepSpeed wiring is started)
- Production database backends for the platform API

See [`docs/architecture/overview.md`](docs/architecture/overview.md) for the full roadmap.

---

## Sovereignty

This platform is designed with sovereignty as a first principle — not just branding. See [`docs/sovereignty/principles.md`](docs/sovereignty/principles.md).

---

## Provenance

This repository builds on ideas and patterns from well-known open-source AI research projects. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full attribution.

The codebase is original. We are not a fork of DeepSeek, LLaMA, or any other project; we draw inspiration from open research in the same way that all serious LLM implementations do.

---

## Licence

See [LICENSE](LICENSE).
