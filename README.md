# supreme-modeltx

> A British sovereign AI platform combining a PyTorch-native trainable LLM engine with an API-first business platform for secure, governable, high-value enterprise AI applications.

---

## What Is This?

`supreme-modeltx` is built around two first-class domains.
It is currently a **foundation scaffold**: core architecture and interfaces are in place, while some production integrations remain intentionally stubbed.

### A. `model_core` — Sovereign LLM Engine
A self-contained PyTorch training and inference stack:

- **T-Series model family** — dense decoder-only transformers (T101 → T501 → T-X)
- Rotary positional embeddings (RoPE) and SwiGLU FFN
- Reproducible data pipeline (JSONL, plain-text, HuggingFace datasets)
- Trainer compatible with Accelerate and DeepSpeed for GPU scaling
- Evaluation harness (perplexity, accuracy)
- Local inference engine with greedy and nucleus (top-p) sampling

### B. `platform_api` — API-First Business Platform
An OpenAI-compatible API surface for enterprise integration:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/usage`
- `POST /v1/keys`
- Per-tenant metering, API key authentication, model registry

`POST /v1/chat/completions` is scaffolded today and currently returns a placeholder completion path rather than a full production inference routing stack.

---

## Repository Structure

```
supreme-modeltx/
├── src/supreme_modeltx/
│   ├── model_core/          # Sovereign LLM engine
│   │   ├── config.py        # Pydantic config schema
│   │   ├── models/          # T-Series transformer
│   │   ├── training/        # Training loop
│   │   ├── data/            # Data pipeline
│   │   ├── tokenizer/       # BPE tokenizer
│   │   ├── eval/            # Evaluation harness
│   │   └── inference/       # Inference engine
│   ├── platform_api/        # Business API platform
│   │   ├── api/             # FastAPI app
│   │   ├── auth/            # API key management
│   │   ├── tenants/         # Tenant / project management
│   │   ├── usage/           # Token metering
│   │   ├── model_registry/  # Model version catalogue
│   │   └── deployment/      # Deployment management
│   └── utils/               # Device selection, logging
├── tests/                   # Pytest test suite
├── control-plane/           # .NET control plane (Blazor + API)
├── infra/                   # Azure Bicep infrastructure
├── scripts/                 # Dockerfiles, run scripts
├── docs/
│   ├── architecture/        # Architecture documentation
│   └── sovereignty/         # Sovereignty principles
├── pyproject.toml
└── THIRD_PARTY_NOTICES.md
```

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
python -m unittest discover -s tests -v
# or
pytest tests/ -v
```

### Syntax / import check

```bash
python -m compileall src
```

### Run the platform API

```bash
uvicorn supreme_modeltx.platform_api.api.app:create_app --factory --reload
```

---

## Reused vs Newly Introduced

- **Reused from existing repository foundation:** control-plane (`control-plane/`), infrastructure (`infra/`), deployment/runtime scripts (`scripts/`), and broad architecture/documentation direction.
- **Newly introduced in this scaffold line:** Python package under `src/supreme_modeltx/` with `model_core` (trainable PyTorch-first LLM scaffolding), `platform_api` (FastAPI-first business API scaffolding), and aligned tests under `tests/`.

---

## T-Series Model Family

| Model | Hidden | Layers | Heads | FFN   | Context |
|-------|--------|--------|-------|-------|---------|
| T101  | 256    | 4      | 4     | 1024  | 512     |
| T201  | 512    | 8      | 8     | 2048  | 1024    |
| T301  | 768    | 12     | 12    | 3072  | 2048    |
| T501  | 1024   | 24     | 16    | 4096  | 4096    |
| T-X   | TBD    | TBD    | TBD   | TBD   | TBD     |

---

## Sovereignty

See [docs/sovereignty/principles.md](docs/sovereignty/principles.md) for our commitments on:

1. British sovereign ownership
2. Trainable and governable pipelines
3. API-first for enterprises
4. Auditability
5. Provider independence
6. Portability
7. UK domain focus

---

## Third-Party Notices

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution of open-source components and architectural inspirations.

---

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Model Core Architecture](docs/architecture/model-core.md)
- [Platform API Architecture](docs/architecture/platform-api.md)
- [Sovereignty Principles](docs/sovereignty/principles.md)

---

## Licence

MIT — see [LICENSE](LICENSE).
