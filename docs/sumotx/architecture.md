# Sovereign AI Governance Architecture (SUMOTX/SMTX Control Plane)

SUMOTX/SMTX is built as a sovereign governance control plane plus independent execution services inside customer-owned Azure boundaries.

**Core principle:** lifecycle governance first; execution profile second.

**Naming in this document:** **SMTX** = repository/product family, **SUMOTX** = governed platform/control-plane offering.

---

## Architecture Objective

Define a stable control-plane foundation that remains valid even when custom model training is paused:

- approvals
- policy evaluation
- audit evidence
- model/provider registration
- deployment controls
- external/provider-backed model operation

Custom model training is an optional adjacent capability, not a prerequisite for the SMTX governance product.

---

## High-Level Design

```text
Users / Systems
      │
      ▼
SMTX Sovereign Control Plane (API-first)
  ├─ Identity + RBAC
  ├─ Model/provider registry
  ├─ Policy decision service
  ├─ Approval workflow service
  ├─ Deployment control service
  └─ Audit event pipeline
      │
      ▼
Execution services (customer-owned)
  ├─ Inference runtimes
  ├─ Retrieval + verification services
  └─ Optional model lifecycle workers
      │
      ▼
Enterprise data boundary
  ├─ SharePoint / internal repositories
  ├─ Blob/Search/metadata
  └─ Compliance evidence stores
```

---

## Governance Domains in the Control Plane

| Domain | Primary responsibility | Key records produced |
| --- | --- | --- |
| Model/provider registry | Register model metadata and runtime/provider profiles | Model registration records, provider capability records |
| Policy | Evaluate deployment/runtime requests against enterprise policy | Policy decision records with rule versions and outcomes |
| Approval | Enforce human/delegated approval gates before activation | Approval decisions, approver identity, timestamps |
| Deployment control | Activate only approved artifacts and configurations | Deployment intents, release transitions, rollback events |
| Audit | Preserve end-to-end traceability across lifecycle operations | Immutable audit evidence and correlation IDs |

Current bootstrap implementations may embed temporary in-memory policy fixtures for local/demo flows, but enterprise policy source-of-truth is the governed control-plane policy domain/store.

---

## Governance Lifecycle Contract

`register → evaluate-policy → approve → deploy → operate → audit`

### Lifecycle gates

1. **Register**
   - Model and provider-backed runtime definitions are registered.
2. **Evaluate policy**
   - Policy service returns allow/deny/conditional decisions.
3. **Approve**
   - Required approvals are collected before activation.
4. **Deploy**
   - Deployment control activates only approved artifacts and configs.
5. **Operate**
   - Inference/grounding runs under the same policy and identity boundary.
6. **Audit**
   - Every decision and state transition is recorded for compliance review.

No gate can be skipped; control-plane APIs and deployment control checks enforce gate order before activation.
If a gate fails, activation is denied and the request remains in a non-active state until policy or approval requirements are satisfied.

---

## Control Plane Boundaries

### What the control plane owns

- Governance state and lifecycle transitions
- Approval and policy orchestration
- Deployment authorization and release state
- Audit trail generation and retention controls

### What the control plane does not own

- Heavy model training compute
- High-throughput inference compute
- Data indexing compute internals

**Hard boundary:** control-plane services orchestrate and govern; they do not perform model training/fine-tuning compute or high-throughput inference serving compute.

---

## Provider-Backed Model Strategy

The governance model is provider-agnostic and supports:

- hosted/provider APIs
- open-source self-hosted runtimes
- T-Series internal model assets:
  - Naming aligns with currently defined modules in `docs/architecture.md` and `tmodels/` (`t101`, `t201`, `t301`, `t501`, `tx`)
  - `T-101` base model
  - `T-201` reasoning specialization
  - `T-301` retrieval support
  - `T-501` verification support
  - `T-X` orchestration layer

All provider paths must pass through the same registration, policy, approval, deployment, and audit controls.

---

## Sovereignty and Security Posture

- Customer-owned Azure subscription is the primary trust boundary.
- RBAC and managed identity define who can register, approve, and deploy.
- Private networking and controlled egress protect model/data boundaries.
- Audit evidence is retained for compliance and incident reconstruction.

---

## Execution Profiles

| Profile | Default | Use |
| --- | --- | --- |
| `cpu-single-node` | ✅ | Baseline governed execution |
| `cpu-distributed` | Optional | Scale-out without changing governance model |
| `gpu-accelerated` | Optional | Enabled only when policy + quota allow |

These execution profiles change runtime capacity only; governance lifecycle and control-plane gates remain unchanged.
