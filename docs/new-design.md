# SMTX New Design Overview

This document captures the current **new design direction** for SMTX:

- Position SMTX as a **governed AI control plane** inside customer-owned Azure boundaries.
- Keep the **T‑Series modular architecture** (`T‑101`, `T‑201`, `T‑301`, `T‑501`, `T‑X`) as supporting runtime and future model strategy.
- Maintain an **Azure‑first deployment model** for governed deployment, grounding, and auditability.
- Use orchestration for lifecycle control: register → approve → deploy → ground → audit.

## SUMOTX Farm Architecture Checklist

Use this checklist to validate documentation completeness before starting framework implementation.

### 1) Architecture overview

- Define control-plane, execution, and data/grounding responsibilities.
- Clarify policy enforcement points and audit boundaries.
- Mark what scales horizontally vs. vertically.
- Mark stateless vs. stateful boundaries.

### 2) SharePoint grounding + training flow (critical)

Expected flow:

`SharePoint Team Site → SPFx trigger → SUMOTX Control Plane → SharePoint Connector → Blob/Search → Retrieval/Grounding → Policy/Approval → Inference`

- SharePoint ingestion is governance-controlled and read-only from runtime services.
- Grounded retrieval is the primary integration path; training workflows remain optional and policy-gated.

### 2.1) SharePoint-style server node scaling across layers

To support SharePoint-driven workloads at farm scale, each layer should scale independently:

- **Layer 1 — SharePoint entry nodes (SPFx + webhook ingress):** scale out horizontally; keep nodes stateless and idempotent so repeated triggers are safe.
- **Layer 2 — SharePoint connector workers (Graph → Blob):** scale out by queue depth and tenant volume; isolate per-tenant connector pools to avoid noisy neighbor contention.
- **Layer 3 — SUMOTX control plane:** scale API replicas horizontally behind a load balancer; persist only control state in shared data services.
- **Layer 4 — Data/memory services (Blob, Search, metadata):** scale with partitioning and throughput tiers; treat this layer as stateful and independently capacity-managed.
- **Layer 5 — Policy + lifecycle services:** scale policy checks, approvals, and audit event handling independently.
- **Layer 6 — Inference nodes:** scale out read replicas for concurrency first, then scale up instance size when latency SLOs require it.

This separation lets SharePoint-facing nodes grow with collaboration traffic without forcing full-stack scaling for every layer.

### 3) Data and governance

- Data should not persist in containers.
- Blob Storage and Azure Cognitive Search provide durable data/memory services.
- Include permissions, approvals, and auditability expectations.
- Include grounding behavior and optional incremental training behavior.

### 4) Deployment phases

- Single-VM deployment with all layers co-located.
- Multi-VM farm deployment.
- Layer split strategy without refactoring core contracts.

### 5) What SUMOTX is / is not

- SUMOTX is **not** a monolithic LLM.
- SUMOTX is **not** GPU-dependent.
- SUMOTX is a governed control plane coordinating policy, lifecycle, and auditable execution.

## Transition: Design → Framework Development

### Step 1 — Freeze architecture

- Treat current architecture as v1.
- Evolve implementation details, not core boundaries.

### Step 2 — Scaffold framework boundaries

Create skeletal modules first:

```text
sumotx/
├── control-plane/
├── inference/
├── training/
├── memory/
├── data/
├── sharepoint/
│   ├── connector/
│   └── spfx-app/
├── deployment/
└── docs/
```

### Step 3 — Build sequence

1. Control Plane API (job registry, orchestration)
2. Policy + approval flow scaffolding
3. SharePoint Connector (Graph → Blob/Search)
4. Grounded inference + reload logic
5. SPFx add-on (Train / Ask actions)

## Control Plane Implementation Lock-In

- Control Plane technology stack: **C# / ASP.NET Core + Blazor Server**
- Control Plane responsibilities: orchestration, policy, registry, auditability
- Control Plane exclusions: training compute, inference compute, retrieval indexing
- Canonical rule: **SUMOTX is a set of independently deployable services coordinated by an explicit Control Plane.**

For full details, see:

- [Architecture Overview](architecture.md)
- [Azure Infrastructure](azure.md)
- [Development Roadmap](roadmap.md)
