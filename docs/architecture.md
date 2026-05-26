# SMTX Architecture Overview
Version: 0.2.0

---

## 1. Platform Identity

SMTX is a governed AI control plane for customer-owned Azure deployments.

The architecture is designed around enterprise lifecycle control, not GPU-first model hosting:

1. Register models and runtime configurations
2. Enforce policy and approval gates
3. Deploy into customer-boundary execution environments
4. Ground inference with enterprise retrieval sources
5. Audit decisions and operations end-to-end

T-Series model components remain part of the roadmap and runtime strategy, but are subordinate to the control-plane product identity.

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
  ├─ Inference runtime (CPU default, GPU optional)
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

Current repository components already implement control-plane and runtime scaffolding. Lifecycle governance depth is being expanded in this same architecture direction.

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

- **Default:** CPU-first operation
- **Optional:** GPU acceleration where customer quota/policy allows

### 4.3 Data + Grounding Plane

Responsibilities:

- Keep enterprise sources (for example SharePoint) as governed grounding inputs
- Maintain customer-owned storage/search services
- Preserve data residency and tenant boundaries

---

## 5. T-Series Positioning

T-Series modules stay relevant as supporting runtime capabilities:

- **T-101** base model path
- **T-201** reasoning specialization
- **T-301** retrieval/benchmark support
- **T-501** verification/evidence support
- **T-X** orchestration and decision layer

Near-term product narrative remains control-plane governance. Large-scale frontier-model training is a future-state option, not the immediate platform identity.

---

## 6. Security and Sovereignty

- Customer-owned Azure subscription is the default trust boundary
- Identity and RBAC govern who can register, approve, deploy, and operate AI
- Policy and audit trails provide compliance evidence
- Provider flexibility is preserved without forcing vendor lock-in

