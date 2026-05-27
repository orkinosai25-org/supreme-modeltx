# Architecture Overview

## Supreme ModelTX — Two-Layer Platform Architecture

Supreme ModelTX is organised around two distinct, independently deployable layers.

> This repository is currently a scaffold/foundation release: architecture, interfaces, and validation scaffolding are in place, while production persistence and fully trained model deployments are follow-up phases.

```
┌─────────────────────────────────────────────────────────────────┐
│                     supreme-modeltx                             │
│                                                                 │
│  ┌───────────────────────────┐  ┌───────────────────────────┐  │
│  │       model_core          │  │      platform_api         │  │
│  │                           │  │                           │  │
│  │  config/                  │  │  api/      (FastAPI)      │  │
│  │  models/                  │  │  auth/     (API keys)     │  │
│  │    common/ (RoPE, GQA)    │  │  tenants/  (projects)     │  │
│  │    t_series/ (baseline)   │  │  usage/    (metering)     │  │
│  │  training/                │  │  model_registry/          │  │
│  │    distributed/           │  │  deployment/              │  │
│  │    checkpoint/            │  │                           │  │
│  │    optimizer/             │  │                           │  │
│  │    scheduler/             │  │                           │  │
│  │    precision/             │  │                           │  │
│  │  data/                    │  │                           │  │
│  │  tokenizer/               │  │                           │  │
│  │  eval/                    │  │                           │  │
│  │  inference/               │  │                           │  │
│  └───────────────────────────┘  └───────────────────────────┘  │
│                                                                 │
│  utils/  (device selection, logging)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Principles

1. **PyTorch-native**: model_core depends only on PyTorch and standard Python. No framework lock-in at the model level.
2. **API-first**: the platform_api layer exposes well-defined REST boundaries. The model backend is pluggable behind those boundaries.
3. **Clean module separation**: model_core and platform_api share no circular imports. They communicate only through stable interfaces (checkpoint paths, model IDs).
4. **Sovereignty by design**: no mandatory dependency on a hosted provider at any layer. All components can run on sovereign compute.
5. **Scaffold-honest**: the repository clearly distinguishes what is implemented, what is stubbed, and what is on the roadmap.

---

## Phase roadmap

| Phase | Description | Status |
|---|---|---|
| **0 — Scaffold** | Package structure, model primitives, config schema, API boundaries | ✅ This PR |
| **1 — Training** | Tokenizer training, data pipeline, first training run on GPU | 🔜 Next |
| **2 — Evaluation** | Perplexity benchmark, lm-eval-harness integration | 🔜 |
| **3 — Scale** | FSDP/DeepSpeed backend, multi-node training | 🗺 |
| **4 — Serving** | vLLM integration, production API hardening | 🗺 |
| **5 — Enterprise** | Database-backed platform_api, RBAC, audit logging | 🗺 |

---

## Module documentation

- [Model Core](model-core.md)
- [Platform API](platform-api.md)
