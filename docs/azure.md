# SMTX Azure Infrastructure

This document describes how SMTX is deployed as a **governed AI control plane** in customer-owned Azure environments.

> **Primary target architecture:** CPU-first governed deployment with control-plane orchestration, policy boundaries, retrieval/grounding, and auditability.
>
> **Optional acceleration mode:** GPU-enabled training/inference can be attached when customer quota, policy, and approvals allow it.

---

## Architecture Overview

SMTX on Azure is organized around three boundaries:

1. **Control plane** — API/admin services for lifecycle orchestration, policy, approvals, and audit records
2. **Execution plane** — inference/runtime and optional training workers in customer-owned compute
3. **Data/grounding plane** — enterprise sources and storage/search services used for retrieval and evidence

```text
Users / Systems
      │
      ▼
SMTX Control Plane (customer Azure)
  ├─ Identity + RBAC
  ├─ Policy + approval gates
  ├─ Model/deployment registry
  └─ Audit log pipeline
      │
      ▼
Execution services
  ├─ Inference (CPU default)
  ├─ Retrieval + verification
  └─ Optional training workers
      │
      ▼
Customer data boundary
  ├─ SharePoint + enterprise knowledge sources
  ├─ Blob/Search/metadata services
  └─ Compliance evidence and logs
```

---

## Current Implemented Repo Surface

This repository currently ships a mixed execution surface:

- **Current target path:** VM-layer CPU baseline (`infra/main.bicep`, `deployVms=true`)
- **Optional compatibility path (disabled by default):** GPU VM + App Service path (`deployGpuVm=true`)
- **Batch path:** retained for optional training workflows (`scripts/azure_batch_cpu.yml`, `scripts/azure_batch.yml`)

The governance-first architecture applies across all paths, but execution profiles differ by deployment mode.

---

## Current Target Architecture

### Governed CPU Baseline (current target)

- CPU-first runtime and orchestration
- No GPU quota requirement
- Suitable for enterprise and public-sector pilots and governed production slices

## Optional Execution Modes

### Governed Accelerated Execution (optional)

- Attach GPU capacity for selected workloads
- Requires customer quota + explicit policy/approval gates
- Same control-plane governance model; only execution profile changes

### CPU Distributed Training (optional)

- Uses Batch/VM scale-out for larger CPU-bound training tasks
- Retains control-plane governance and audit controls
- Primary artifacts: `scripts/azure_batch_cpu.yml`, `.github/workflows/train.yml`

## Future-State Programs

### Future-scale model programs

- Larger distributed training programs (for example T-Series expansion)
- Still governed by the same control-plane lifecycle and audit requirements

---

## Resource Inventory (Governed CPU Baseline)

| Resource | Purpose |
|---|---|
| Control-plane API / admin services | Registration, approvals, lifecycle orchestration, audit administration |
| Azure VM / App service compute | Runs control-plane and runtime services |
| Azure Storage (Blob/files) | Datasets, artifacts, evidence records |
| Azure Search / retrieval services | Grounding and enterprise knowledge retrieval |
| Key Vault | Secret and key custody |
| Managed Identity + RBAC | Service-to-service access and role boundaries |
| Virtual Network + private endpoints | Boundary control and private access paths |
| Monitoring/logging services | Operational telemetry and audit support |

---

## Service Boundaries and Runtime Endpoints

| Service | Repo Artifact | Default Port | Role |
|---|---|---:|---|
| Control-plane API | `control-plane/src/SMTX.ControlPlane.Api` | app-configured | Lifecycle orchestration, registry, governance control |
| T-X orchestrator | `tmodels/tx/orchestrator.py` | `8080` | Multi-stage orchestration + backend routing |
| CPU inference backend (default) | `inference/cpu_inference_server.py` | `8003` | Default inference execution path |
| vLLM backend (optional) | `inference/vllm_server.py` | `8000` | GPU-conditional accelerated inference |
| Retrieval service | `inference/retrieval_service.py` | `8001` | Grounding retrieval operations |
| Verification service | `inference/verification_service.py` | `8002` | Claim/evidence verification |

Control-plane and orchestrator components govern execution; runtime services remain independently deployable.

---

## Prerequisites

1. Azure CLI with Bicep support
2. Azure subscription and target resource group
3. Required Azure provider registrations for compute, network, storage, identity, and monitoring
4. RBAC permissions to deploy and manage resources
5. (Optional) GPU quota only if Mode B is explicitly enabled

### One-time prerequisites for workflow-driven Azure setup

Before running GitHub Actions provisioning, these minimum one-time prerequisites must already exist:

1. Azure Web App provisioned for SMTX deployment (the workflow wires settings/secrets to this existing app)
2. GitHub OIDC service principal/federated credential with rights to deploy subscription/resource-group resources
3. GitHub repository secrets configured:
   - `AZUREAPPSERVICE_CLIENTID_E45C3D43E5AC40DB8318380C0FD93246`
   - `AZUREAPPSERVICE_TENANTID_01CDF649EE9042BBAB65955AC8A5A538`
   - `AZUREAPPSERVICE_SUBSCRIPTIONID_A3780EA742DC446B87D82128AC45B190`
   - `SSH_PUBLIC_KEY`
   - `SUMOTX_SQL_ADMIN_PASSWORD`

---

## Deployment

```bash
az group create --name smtx-rg --location uksouth

az deployment group create \
  --resource-group smtx-rg \
  --template-file infra/main.bicep \
  --parameters @infra/parameters.json
```

### Reproducible Azure-backed setup via GitHub Actions

Use `.github/workflows/azure-platform-setup.yml` to provision and wire the remaining SMTX dependencies with IaC:

- Azure SQL Database for control-plane/policy persistence
- Azure Storage account + blob containers for documents/artifacts/checkpoints/policies
- Azure AI Search for indexing/retrieval
- Azure OpenAI account configuration for model access
- Managed identity role assignments for Web App access to Storage/Search/OpenAI
- Web App app settings + SQL connection string wiring

Recommended deployment controls:

- Use `what-if` before production deployments
- Keep network exposure private by default
- Use managed identities and Key Vault references (no plaintext secrets)
- Keep `deployGpuVm=false` unless optional accelerated mode is intentionally enabled

### Operator workflow (concise current runbook behavior)

1. Provision infra with Bicep (`infra/main.bicep`)
2. Deploy/update control-plane and runtime services
3. Run training/inference workflows (VM-first via `.github/workflows/train.yml`, Batch optional)
4. Validate service health (`/health`, `/ready`) and orchestration path
5. Capture and retain operational/audit evidence in customer-controlled monitoring and storage

---

## Identity, Policy, and Audit Controls

- Managed identities authenticate service-to-service calls
- RBAC defines separation of duties (admin/operator/reviewer/auditor)
- Policy checks gate model registration, deployment, and execution paths
- Audit events capture actor, action, timestamp, policy decision, and target resource

> Control-plane services enforce governance state. Runtime services execute approved workloads inside customer boundaries.

---

## Networking Expectations

- VNet CIDR: `10.0.0.0/16` in the current infrastructure template (`infra/modules/vnet.bicep`, deployed from `infra/main.bicep`)
- Subnets include control, API, frontend, lifecycle, data, inference, and training layers, plus a private-endpoint subnet
- NSG model is explicit-allow + deny-all-inbound per layer
- Storage + Key Vault private endpoints are enabled; public access is restricted by default
- Managed identity + RBAC is the expected service authentication baseline

---

## Retrieval and Grounding in Azure

SMTX grounding pattern in Azure:

- Ingest enterprise knowledge sources (including SharePoint-driven flows)
- Store/index in customer-controlled storage and search services
- Route inference through retrieval and verification services
- Record retrieval and policy metadata for auditability

This keeps enterprise knowledge inside customer tenancy while improving answer quality and traceability.

---

## Execution Profiles

| Profile | Default | Notes |
|---|---|---|
| `cpu-single-node` | ✅ | Baseline profile for governed workloads |
| `cpu-distributed` | Optional | Scale-out CPU execution under same governance controls |
| `gpu-accelerated` | Optional | Enabled only when quota and policy approval exist |

GPU is an execution option, not a product identity.

Profile mapping in repo:

- `scripts/run_training.sh` selects and enforces these profiles
- `.github/workflows/train.yml` exposes CPU profiles by default in manual dispatch
- `inference/backends.yml` defines CPU backend as default and vLLM as conditional

Operational expectations:

- Deploy/update actions are initiated through runbooks and CI workflows, then validated with service health/readiness endpoints
- Control-plane state changes should be traceable in customer-owned monitoring/storage
- Optional profiles must preserve the same identity/network/security boundary controls as the CPU baseline

---

## Service Boundaries

| Service Type | Responsibility |
|---|---|
| Control Plane API | Registry, policy, approvals, orchestration, audit administration |
| Inference Runtime | Execute approved inference requests |
| Retrieval / Verification | Grounding, evidence checks, quality guardrails |
| Training Workers (optional) | Model adaptation programs under lifecycle controls |

---

## Runbook — Training

- Training programs are lifecycle-controlled tasks
- CPU paths are baseline; GPU paths require explicit enablement
- Artifacts and checkpoints remain in customer-owned storage
- VM-based workflow is the recommended operational path; Batch is retained as optional/legacy mode

## Runbook — Inference

- Default deployment path is CPU inference
- Optional vLLM/GPU backends are conditional and policy-gated
- Health/readiness checks must be included in operational workflows
- Orchestrator backend routing is controlled by `VLLM_ENABLED` and endpoint settings

## Runbook — Retrieval + Verification (CPU VM)

- Retrieval and verification services are deployed as independent runtime services
- Both services should read enterprise-grounding indices from customer-owned storage/search services
- Runtime actions and quality decisions should emit audit-compatible metadata

## Runbook — T-X Orchestrator (App Service)

- Deploy and monitor control-plane APIs as the authoritative lifecycle system
- Keep deployment and runtime actions auditable and replayable

---

## Cost and Procurement Positioning

- Azure consumption (compute, storage, networking) remains customer-billed
- SMTX value is governance software: lifecycle control, policy enforcement, boundary-safe deployment, and auditability
- Optional acceleration should be justified by policy and workload requirements, not by default architecture assumptions

---

## Summary

SMTX on Azure is designed to help organizations govern how AI is approved, deployed, grounded, and audited inside their own environment.

Training and model acceleration remain important capabilities, but they are supporting execution modes within a governance-first control-plane architecture.
