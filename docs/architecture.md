# SMTX Architecture Overview
Version: 0.2.0

---

## 1. Platform Identity

SMTX is an early-stage governed pretrained-model evaluation, adaptation, and
deployment foundation supporting Ebru. Customer-owned Azure is a reference
deployment boundary, not a requirement or evidence of production readiness.

The architecture targets enterprise lifecycle control around approved models:

1. Register models and runtime configurations
2. Enforce policy and approval gates
3. Deploy into customer-boundary execution environments
4. Ground inference with enterprise retrieval sources
5. Audit decisions and operations end-to-end

T-series custom models and CPU workflows remain research/smoke paths. The
near-term Ebru strategy is to assess approved pretrained models, not make
original large-model pretraining a product prerequisite.

---

## 2. Control Plane + Runtime Pattern

```text
Users / Systems
      │
      ▼
SMTX Control Plane (API + admin)
  ├─ Identity / RBAC
  ├─ Policy + approval workflow
  ├─ Model + deployment registry
  ├─ Audit event pipeline
  └─ Orchestration
      │
      ▼
Execution plane (customer-owned Azure)
  ├─ Local inference runtime (PyTorch default; optional Transformers)
  ├─ Retrieval + grounding services
  ├─ Verification / assurance services
  └─ Optional training workers
      │
      ▼
Enterprise data boundary
  ├─ SharePoint / enterprise knowledge sources
  ├─ Blob/Search/metadata stores
  └─ Immutable audit evidence
```

---

## 3. Governance Lifecycle

SMTX target lifecycle:

- **Register** → Model, provider, configuration, intended use
- **Review/Approve** → Policy checks and human approval gates
- **Deploy** → Controlled release into customer environments
- **Ground** → Retrieval and verification against enterprise sources
- **Audit** → Trace who did what, when, and under which policy

Current repository components implement control-plane and runtime scaffolding.
The full approval, production security, and immutable audit lifecycle remains a
target, not an implemented end-to-end guarantee.

---

## 4. Architectural Roles

### 4.1 Control Plane

Responsibilities:

- API-first orchestration
- Policy enforcement points
- Approval state and lifecycle transitions
- Registry of models/deployments
- Auditability and traceability

Hard boundary:

- Control-plane services coordinate and govern workloads
- They do not own heavy training/inference execution

### 4.2 Execution Plane

Responsibilities:

- Run approved inference and optional training workloads
- Execute retrieval and verification pipelines
- Produce telemetry and evidence back to control-plane systems

Execution profiles:

- **Default backend:** `SMTX_INFERENCE_BACKEND=pytorch`, using `model_core` artifacts
- **Optional backend:** `SMTX_INFERENCE_BACKEND=transformers`, using a complete local
  snapshot at `SMTX_PRETRAINED_MODEL_PATH` and strict `SMTX_DEVICE=cpu|cuda`
- **Research/smoke:** CPU custom-model workflows remain supported
- **GPU:** existing CUDA paths require approved hardware and validation; no
  completed single-GPU or multi-GPU validation is claimed

Requesting unavailable CUDA for pretrained inference fails rather than silently
falling back. Snapshot selection does not imply model approval or a Hub download.
`Qwen/Qwen2.5-0.5B-Instruct` (candidate Apache-2.0, not independently verified here)
is only a candidate pending explicit upstream
license/revision and Turkish evaluation approval.

### 4.3 Data + Grounding Plane

Responsibilities:

- Keep enterprise sources (for example SharePoint) as governed grounding inputs
- Maintain customer-owned storage/search services
- Preserve data residency and tenant boundaries

---

## 5. T-Series Positioning

`model_core`'s canonical T-Dev-6L is ~58M parameters / 512-token context.
T-101 is an unvalidated legacy ~7B config; larger T-series names are research
concepts, not validated runtime capabilities or delivery commitments.
`foundation`'s TinyPilotLM checkpoint format is separate from `model_core`'s
TSeriesBaseline format and from pretrained snapshots.

### NVIDIA / portability boundary

| Technology | Status | Intended role |
|---|---|---|
| CUDA / PyTorch | Existing paths; not GPU-validated | Approved local GPU experiments |
| cuDNN | Runtime library where supplied by a compatible CUDA stack | Runtime compatibility, not a model backend |
| NeMo | Planned; not integrated | Governed pretrained adaptation |
| Megatron Core | Planned; not integrated | Distributed research if justified |
| TensorRT-LLM | Planned; not integrated | Supported-model serving optimization |
| ONNX | Planned; not integrated | Export portability and parity checks |

The [roadmap](roadmap.md) requires license, reproducibility, Turkish quality,
safety, and deployment gates; [GPU planning](gpu-readiness-scaling-plan.md)
provides official resource links. No candidate measurements or actual GPU runs
are claimed by this architecture.

---

## 6. Security and Sovereignty

- Customer-owned Azure subscription is the default trust boundary
- Identity and RBAC govern who can register, approve, deploy, and operate AI
- Policy and audit trails provide compliance evidence
- Provider flexibility is preserved without forcing vendor lock-in
