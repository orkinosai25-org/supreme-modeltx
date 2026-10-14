# Architecture Overview

## Supreme ModelTX — Governed Model Foundation

Supreme ModelTX is an early-stage governed pretrained-model evaluation,
adaptation, and deployment foundation supporting Ebru. The two layers below
show the custom-model research path and API boundary, not a complete pretrained
adaptation or optimized serving stack.

> Architecture, interfaces, local persistence, and CPU validation scaffolding
> exist. Production hardening, approved pretrained evaluation, and GPU validation
> remain gated follow-up work.

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

1. **PyTorch-native research**: model_core contains custom-model primitives; CPU workflows remain research/smoke paths, not Turkish product-quality evidence.
2. **API-first**: the platform_api layer exposes well-defined REST boundaries. The model backend is pluggable behind those boundaries.
3. **Clean module separation**: model_core and platform_api share no circular imports. They communicate only through stable interfaces (checkpoint paths, model IDs).
4. **Sovereignty by design**: no mandatory dependency on a hosted provider at any layer. All components can run on sovereign compute.
5. **Scaffold-honest**: the repository clearly distinguishes what is implemented, what is stubbed, and what is on the roadmap.

---

## Phase roadmap

| Gate | Description | Evidence required |
|---|---|---|
| Candidate approval | Licensed local snapshot and Turkish test set | Revision/hashes, license review, frozen thresholds |
| Local evaluation | CPU pretrained baseline | Complete outputs, scores, reproducibility report |
| GPU evaluation | Approved one-GPU comparison | Quality, latency, throughput, VRAM, runtime versions |
| Adaptation/serving | NeMo / TensorRT-LLM / ONNX feasibility | Held-out comparison and parity within approved tolerance |
| Ebru pilot | Governed deployment | Model card, rollback, safety and operator sign-off |

See [the measurable roadmap](../roadmap.md). These are plans, not completed runs.

## Runtime contracts and NVIDIA status

- API default: `SMTX_INFERENCE_BACKEND=pytorch`, with `model_core` checkpoint and tokenizer.
- Optional pretrained inference: `SMTX_INFERENCE_BACKEND=transformers`,
  `SMTX_PRETRAINED_MODEL_PATH` for an approved complete local snapshot, and strict
  `SMTX_DEVICE=cpu|cuda`; unavailable requested CUDA fails instead of falling back.
- `foundation` TinyPilotLM, `model_core` TSeriesBaseline, and pretrained snapshots
  have distinct artifact contracts and are not interchangeable.
- Canonical T-Dev-6L is ~58M / 512 context; T-101 is an unvalidated legacy ~7B
  configuration. Larger targets are research concepts, not commitments.
- `Qwen/Qwen2.5-0.5B-Instruct` (candidate Apache-2.0, not independently verified here)
  is a candidate pending explicit upstream
  license/revision and Turkish evaluation approval, with no results claimed.
- CUDA code exists but is not GPU-validated; no multi-GPU validation is claimed.
- NeMo, Megatron Core, TensorRT-LLM, and ONNX are **planned, not integrated**.
  cuDNN is a runtime library, not a model backend.

Official resource links and compatibility gates are in
[GPU planning](../gpu-readiness-scaling-plan.md).

---

## Module documentation

- [Model Core](model-core.md)
- [Platform API](platform-api.md)
