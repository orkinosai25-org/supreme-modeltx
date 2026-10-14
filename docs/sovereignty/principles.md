# Sovereignty Principles

## What "sovereign AI" means in this platform

Sovereignty is not branding or a requirement to pretrain every model. It is a
set of concrete decisions about who controls artifacts, data, execution, and
approval. Supreme ModelTX is an early-stage governed pretrained-model evaluation,
adaptation, and deployment foundation supporting Ebru; these principles are
design goals, not certification of a production system.

Supreme ModelTX defines sovereignty across five dimensions:

---

## 1. Data sovereignty

- Training data is curated, documented, and stored under the operator's control.
- No mandatory use of third-party hosted datasets; the platform supports any JSONL, Parquet, or local text source via the manifest system.
- Data provenance is tracked in manifests with named sources and weights.
- The `data/raw/` directory in this repository contains example sovereign-curated datasets.

**What is not yet implemented:** automated dataset lineage tracking and cryptographic provenance signatures.

---

## 2. Model sovereignty

- The near-term strategy is evaluating approved pretrained checkpoints under
  operator control; original custom training remains research/smoke work.
- Checkpoints are saved locally and are portable — no dependency on a model hosting provider.
- Model configuration is fully transparent and inspectable as a JSON/YAML file.
- The T-series model architecture is original PyTorch code; the design is inspired by open research (see `THIRD_PARTY_NOTICES.md`) but is not a copy or fork.

**What is not yet implemented:** cryptographically signed model cards, reproducible training manifests with deterministic seeds end-to-end.

Pretrained artifacts require license and notice review, an immutable revision,
recorded hashes, safety review, and Turkish evaluation approval.
`Qwen/Qwen2.5-0.5B-Instruct` (candidate Apache-2.0, not independently verified here)
is a candidate only; explicit upstream approval
and measurements remain pending. Local custody does not remove upstream
license obligations.

`foundation` TinyPilotLM checkpoints, `model_core` TSeriesBaseline checkpoints,
and Transformers local snapshots have separate contracts. Canonical T-Dev-6L
is ~58M / 512 context; T-101 is an unvalidated legacy ~7B configuration. Larger
T-series targets are research concepts, not delivery commitments.

---

## 3. Infrastructure sovereignty

- The model core (`supreme_modeltx.model_core`) has no cloud provider dependency. It runs on any hardware with PyTorch.
- CPU workflows remain research/smoke paths. CUDA code paths exist but are not
  GPU-validated; neither production CUDA nor multi-GPU validation is claimed.
- The deployment layer is designed for Kubernetes and bare-metal, not just managed cloud services.
- Existing Azure Bicep infrastructure in this repository (`infra/`) is a reference implementation, not a requirement.

**What is not yet implemented:** full Kubernetes deployment manifests for on-prem/air-gapped sovereign compute.

The API defaults to `SMTX_INFERENCE_BACKEND=pytorch`. Optional pretrained
inference uses `SMTX_INFERENCE_BACKEND=transformers`,
`SMTX_PRETRAINED_MODEL_PATH` for a complete local snapshot, and strict
`SMTX_DEVICE=cpu|cuda`. Unavailable requested CUDA must fail, not silently change
the approved execution device.

NeMo, Megatron Core, TensorRT-LLM, and ONNX are **planned, not integrated**.
cuDNN is a runtime library where supplied by a compatible CUDA/PyTorch stack,
not an additional model backend. See [GPU planning](../gpu-readiness-scaling-plan.md)
for official resources and compatibility gates.

---

## 4. Access sovereignty

- API keys are issued and managed by the platform operator, not by a third-party identity provider.
- Projects are isolated units with separate API keys, usage tracking, and (in future) RBAC.
- No mandatory external OAuth flow; the auth layer is pluggable.

**What is not yet implemented:** multi-factor admin authentication, key rotation policy enforcement.

---

## 5. Audit sovereignty

- All usage is recorded in the `UsageLedger` with project, model, and token counts.
- The deployment service tracks lifecycle events.
- The platform API returns auditable responses with request IDs.

**What is not yet implemented:** immutable audit log (append-only store), compliance reporting, red-team / safety evaluation pipeline.

---

## Relationship to the UK Sovereign AI Fund

The project may pursue sovereign AI infrastructure support; no eligibility,
funding, compute allocation, or deployment certification is claimed:

- It targets **operator-approved UK-aligned compute** where appropriate.
- It aims for **operator-controlled approved model artifacts**, including licensed pretrained weights.
- It provides an **API-first business access layer** for deploying sovereign models to UK organisations.
- Local-artifact operation supports an **on-prem/air-gapped design direction**;
  complete deployment and security validation remain pending.

The next milestones are:
1. Candidate license/revision and evaluation-data approval
2. Reproducible local Turkish pretrained baseline with predeclared thresholds
3. Approved single-GPU comparison with quality, safety, latency, and VRAM evidence
4. Adaptation/serving feasibility and a governed Ebru pilot after operator sign-off

See [the roadmap](../roadmap.md) for measurable gates. GPU access alone does not
resolve licensing, dataset quality, safety, or deployment readiness.

---

## What we do not claim

- We do not claim to have trained a frontier model.
- We do not claim production-ready infrastructure on day one.
- We do not present this as a complete system or an application-ready model.
- We do not claim pretrained candidate measurements or actual GPU runs in this plan.

Honesty about the current state of the system is itself a sovereignty principle: auditable, transparent, and not pretending to capabilities we do not yet have.
