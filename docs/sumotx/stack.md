# Technology Stack

SUMOTX uses Azure-native and open-source technologies to implement a governed AI control plane.

> **Priority order:** governance and lifecycle control first, execution optimization second.

---

## 1. Control Plane

### C# / ASP.NET Core + Blazor

**Role:** API-first control plane and admin surface.

Core capabilities:

- model/deployment registration
- policy and approval orchestration
- lifecycle state handling
- audit event management

### OpenAPI / Swagger

**Role:** first-class contract for integration and governance operations.

---

## 2. Identity, Security, and Policy

### Managed Identity + RBAC

**Role:** service authentication and access boundaries.

### Key Vault + private networking

**Role:** secret custody and boundary-safe deployment.

### Policy and approval controls

**Role:** gate changes before deployment/inference activation and keep decisions auditable.

---

## 3. Grounding and Data Plane

### Azure Storage + Search services

**Role:** enterprise corpus storage, indexing, and retrieval.

### SharePoint connector patterns

**Role:** ingest enterprise knowledge as governed grounding input.

---

## 4. Runtime and Model Execution

### CPU-first execution profiles (default)

- baseline inference and retrieval execution
- suitable for controlled enterprise deployment slices
- default inference backend: `inference/cpu_inference_server.py` (`/health`, `/generate`)
- retrieval/verification services: `inference/retrieval_service.py`, `inference/verification_service.py`

### Optional accelerated profiles

- GPU-enabled paths when customer policy and quota allow
- same control-plane lifecycle and audit model
- optional backend: `inference/vllm_server.py` (activated conditionally by orchestrator settings)

### Model/provider flexibility

- open-source runtimes
- hosted model providers
- T-Series modules as strategic roadmap capability

Current technical pathways in repo:

| Path | Current State | Primary Artifacts |
| --- | --- | --- |
| Runtime routing | Available | `tmodels/tx/orchestrator.py`, `inference/backends.yml` |
| CPU training | Available | `scripts/run_training.sh`, `training/train_t101.py`, `training/config_cpu_*.json` |
| Batch training (CPU) | Available (optional mode; no GPU quota required) | `scripts/azure_batch_cpu.yml`, `.github/workflows/train.yml` |
| Batch training (GPU) | Available (optional mode; requires quota and policy approval) | `scripts/azure_batch.yml`, `training/config_gpu_accelerated.json` |
| T-Series orchestration pattern | Available (baseline implementation) | `tmodels/tx/orchestrator.py` (`T-101/201/301/501` stages) |

---

## Implemented / Current Repo Surface

- Control-plane API/admin stack in ASP.NET Core + Blazor
- Python runtime services for inference, retrieval, verification, and orchestration
- CPU-first training/inference run paths in scripts/workflows
- Optional GPU runtime/training paths retained behind explicit enablement

---

## Current Target Architecture

- Governance-first control plane remains the product center
- CPU-first execution profile is the default technical baseline
- Retrieval + verification are expected runtime companions for grounded operation
- Identity, networking, and data boundaries are enforced in customer-owned Azure

---

## Optional Execution Modes

- GPU-accelerated inference (`inference/vllm_server.py`) when policy + quota permit
- CPU-distributed and Batch-based execution for larger jobs
- Hosted or open-source provider backends behind orchestrated routing controls

---

## Future-State Programs

- T-Series expansion and broader lifecycle automation
- Deeper policy/audit controls across deployment promotion workflows
- Additional provider integrations without changing control-plane governance model

---

## 5. Operations

### Azure Monitor / telemetry

**Role:** operational observability and governance evidence support.

### Infrastructure as Code (Bicep)

**Role:** repeatable deployment in customer-owned Azure environments.

### GitHub Actions

**Role:** CI/CD automation for platform services and infrastructure workflows.

---

## Stack Summary

| Layer | Primary Technology | Purpose |
|---|---|---|
| Control plane | ASP.NET Core + Blazor + OpenAPI | Governance lifecycle orchestration |
| Runtime inference | FastAPI + PyTorch/Transformers (CPU default), vLLM (optional) | Governed model execution |
| Retrieval/verification | FastAPI + FAISS + sentence-transformers | Grounding and evidence checks |
| Data/grounding | Azure Storage + Azure Search + connector pipelines | Enterprise corpus storage, indexing, and retrieval context |
| Training | `train_t101.py`, `run_training.sh`, torchrun/Gloo, optional DeepSpeed | CPU-first model adaptation with optional acceleration |
| Identity/security | Managed Identity, RBAC, Key Vault, private networking | Access and boundary control |
| IaC/operations | Bicep + Azure Monitor + GitHub Actions | Deployability and observability |
