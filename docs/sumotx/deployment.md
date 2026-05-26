# Deployment Model

SUMOTX deploys into the customer's Azure subscription and operates as a governed AI control plane.

---

## Core Principle: Customer-Owned Boundary

All resources run in customer-owned infrastructure.

- ✅ Customer controls compute, storage, networking, and identity
- ✅ Data and model artifacts remain in customer tenancy
- ✅ Governance actions are auditable within customer boundary
- ✅ SUMOTX does not require external SaaS custody for core operations

---

## What SUMOTX Deploys

| Resource | Purpose |
|---|---|
| Control-plane API/admin services | Orchestration, policy, approvals, lifecycle state |
| Runtime compute (CPU default) | Inference and retrieval/verification services |
| Optional execution pools | Additional CPU/GPU capacity when approved |
| Storage + search services | Artifacts, grounding corpora, metadata |
| Managed identities + RBAC | Access boundaries and least privilege |
| Private networking | Tenant and data boundary protection |

---

## Deployment Lifecycle

```text
1. Customer authorizes deployment in its Azure subscription
2. SUMOTX control plane is provisioned
3. Policy, identity, and audit controls are enabled
4. Models/providers are registered
5. Deployments require policy/approval before activation
6. Inference runs with retrieval grounding and traceability
```

---

## Execution Modes

### Baseline mode (default)

- CPU-first, governed execution
- No GPU quota requirement

### Accelerated mode (optional)

- Customer-provided GPU capacity for selected workloads
- Same governance controls; different runtime profile only

---

## SharePoint and Enterprise Knowledge Integration

SharePoint and other enterprise repositories fit the grounding architecture:

- ingestion through controlled connectors
- indexing in customer-owned services
- retrieval during inference under policy constraints
- evidence captured for audit and review

---

## Billing Model

| What | Who pays |
|---|---|
| Azure compute/storage/networking | Customer to Microsoft |
| SUMOTX platform software | Customer to OrkinosAI Labs |

SUMOTX value is governance, control-plane lifecycle, and auditable operation — not opaque infrastructure resale.
