# Sovereignty Principles

## What "sovereign AI" means in this platform

Sovereignty is not branding. It is a set of concrete architectural decisions that determine who controls the AI system and under what conditions.

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

- Models are trained from scratch (or from open-licensed checkpoints) on operator-owned compute.
- Checkpoints are saved locally and are portable — no dependency on a model hosting provider.
- Model configuration is fully transparent and inspectable as a JSON/YAML file.
- The T-series model architecture is original PyTorch code; the design is inspired by open research (see `THIRD_PARTY_NOTICES.md`) but is not a copy or fork.

**What is not yet implemented:** cryptographically signed model cards, reproducible training manifests with deterministic seeds end-to-end.

---

## 3. Infrastructure sovereignty

- The model core (`supreme_modeltx.model_core`) has no cloud provider dependency. It runs on any hardware with PyTorch.
- Training targets CPU (smoke/dev) and CUDA (production), with MPS support for Apple Silicon development.
- The deployment layer is designed for Kubernetes and bare-metal, not just managed cloud services.
- Existing Azure Bicep infrastructure in this repository (`infra/`) is a reference implementation, not a requirement.

**What is not yet implemented:** full Kubernetes deployment manifests for on-prem/air-gapped sovereign compute.

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

Supreme ModelTX is designed to be a credible candidate for sovereign AI infrastructure support:

- It targets **British-owned compute** (including the national AI compute allocation).
- It produces **UK-controlled model weights** through a transparent, reproducible training process.
- It provides an **API-first business access layer** for deploying sovereign models to UK organisations.
- The architecture supports **air-gapped and on-prem deployment** for regulated sectors.

The next concrete milestones tied to GPU compute allocation are:
1. Tokenizer training on sovereign corpus
2. T-101 first training run (proof of GPU utilisation)
3. Benchmark evaluation and model card publication
4. Production deployment to pilot customers

---

## What we do not claim

- We do not claim to have trained a frontier model.
- We do not claim production-ready infrastructure on day one.
- We do not present this as a complete system — we present it as a credible, sovereign-designed scaffold ready for GPU-backed development.

Honesty about the current state of the system is itself a sovereignty principle: auditable, transparent, and not pretending to capabilities we do not yet have.
