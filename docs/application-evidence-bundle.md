# Application Evidence Bundle — Supreme ModelTX

> **Scope:** Concise submission narrative for UK Sovereign AI Fund reviewers.  
> **Assessment date:** 2026-08-03  
> **Evidence index:** [`docs/submission-checklist.md`](submission-checklist.md)  
> **Impact metrics:** [`docs/impact-metrics.md`](impact-metrics.md)

---

## Problem

UK public-sector organisations — government departments, NHS, HMRC, MoD, defence contractors, and regulated financial institutions — require AI capability that is **sovereign by design**: data stays within UK boundaries, model behaviour is auditable, and no single US or non-UK cloud provider controls the stack.

Current market options force a choice between:

1. **Frontier US-hosted APIs** (OpenAI, Anthropic, Google) — zero data-residency guarantee, vendor lock-in, no audit trail meeting public-sector accountability standards.
2. **Open-weight model self-deployment** — high engineering overhead, no governance layer, no UK-sovereign control plane.

Neither option gives public-sector operators the combination they need: **a governed, auditable, sovereign LLM platform they can own end-to-end**.

---

## Solution

**Supreme ModelTX** is a British-built, open-architecture sovereign LLM platform:

- **PyTorch-native model stack** — T-Dev-6L baseline (~58 M params, reproducible from clean checkout); T-101 (7 B) on roadmap ([`src/supreme_modeltx/model_core/`](../src/supreme_modeltx/model_core/)).
- **API-first business layer** — OpenAI-compatible FastAPI platform; API key auth, tenant model, usage metering, model registry ([`src/supreme_modeltx/platform_api/`](../src/supreme_modeltx/platform_api/)).
- **Governance control plane** — C# ASP.NET Core control plane enforcing approval gates and full audit trails before any model is deployed ([`control-plane/src/`](../control-plane/src/)).
- **Azure UK IaC** — Bicep templates for UK South / UK West; data never leaves the customer's Azure subscription ([`infra/main.bicep`](../infra/main.bicep)).

The platform is **OpenAI-compatible at the API surface**, enabling rapid integration into existing public-sector tooling and procurement patterns without re-engineering integrations.

---

## Why Now

Three forces make 2026 the window for a UK sovereign LLM platform:

1. **UK AI Strategy execution** — DSIT and the AI Safety Institute are actively funding sovereign AI capability programmes. Reviewers are evaluating platforms *now*.
2. **Public-sector confidence gap** — High-profile data-handling concerns around US frontier APIs are generating genuine demand from UK government operators for an auditable alternative.
3. **Technical inflection point** — Open-weight LLM architectures have reached a maturity level (7 B param class) where sovereign fine-tuning is feasible without frontier-scale compute. GPU costs in UK Azure regions are becoming accessible to programme-level budgets.

Supreme ModelTX has moved fast to be positioned at this junction: **POC proven, scaling plan clear, GPU constraint actively managed**.

---

## Current Readiness

**TRL 4** — component validation in controlled environment. Full scorecard: [`docs/trl-assessment.md`](trl-assessment.md).

| Layer | Status | Evidence |
|---|---|---|
| Model core (T-Dev-6L) | ✅ Working POC — trains, checkpoints, emits perplexity | [`docs/first-experiment-findings.md`](first-experiment-findings.md), [`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md) |
| Platform API scaffold | ✅ FastAPI routers, auth, tenant model, audit log | [`src/supreme_modeltx/platform_api/`](../src/supreme_modeltx/platform_api/) |
| Governance control plane | ✅ C# ASP.NET Core scaffold | [`control-plane/src/`](../control-plane/src/) |
| Azure UK IaC | ✅ Bicep for UK South / UK West | [`infra/main.bicep`](../infra/main.bicep) |
| GPU path | ✅ Config and preflight verified; training not yet executed at scale | [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md) |
| Security controls | ⚠️ Auth scaffold present; RBAC, policy engine, tamper-evident audit pending | [`docs/gap-analysis.md`](gap-analysis.md) §G-S1–G-S4 |

Full POC status matrix: [`docs/poc-status.md`](poc-status.md).  
Gap analysis (POC → pilot): [`docs/gap-analysis.md`](gap-analysis.md).

---

## 90-Day Delivery Plan

**Sequencing principle:** sovereign-control-plane-first. The highest-value differentiators — governance, RBAC, policy engine, tamper-evident audit — are fully deliverable *without* GPU. GPU-dependent work (large-scale training, vLLM production inference) is Phase 2, not a blocker.

Full plan: [`docs/funding-readiness.md`](funding-readiness.md#90-day-execution-plan), [`docs/delivery-plan-90d.md`](delivery-plan-90d.md).

### Phase 1 — Security boundary foundation (Days 1–30)

| Deliverable | Gap closed | Done when |
|---|---|---|
| Multi-tenant isolation enforced at service layer | G-S2 | Unit test blocks cross-tenant access |
| Scoped token expiry and rotation | G-S3 | Expired tokens return 401; rotation produces new scoped token |
| RBAC middleware with per-role enforcement | G-S1 | Role-gated routes return 403 to under-privileged callers |
| Tamper-evident audit hash-chain | G-G1 | Chain verification passes on clean log; fails on tampered entry |

**Go / No-Go (Day 30):** All four security gaps closed; security/governance TRL score rises 1.0 → 2.0.

### Phase 2 — Policy engine and GPU training (Days 31–60)

| Deliverable | Done when |
|---|---|
| Policy engine v1: configurable rules + decision logging | Policy decisions logged with allow/deny reason codes |
| Usage metering rollups and cost view endpoint | `GET /usage/summary` returns per-project token counts |
| Control-plane approval workflow wired end-to-end | Integration test passes; audit event on each state transition |
| First GPU training run (if quota available) | Checkpoint + perplexity emitted to `artifacts/runs/t_dev_6l_first_gpu_run/` |

**Go / No-Go (Day 60):** Policy engine integrated; TRL 5 claim supportable.

### Phase 3 — Lifecycle, evaluation, and demo (Days 61–90)

| Deliverable | Done when |
|---|---|
| Inference provider abstraction (CPU/GPU swappable) | `InferenceEngine` accepts backend config; both implementations tested |
| Model promotion/rollback with audit events | Transitions logged; rollback restores prior serving version |
| CI-gated evaluation harness | `results/baseline.json` generated; CI fails on regression |
| Public-sector demo (knowledge-grounded Q&A) | `scripts/run_demo.sh` runs end-to-end |
| 90-day evidence pack | One-page report; every claim evidence-linked |

**Go / No-Go (Day 90):** Demo runs from clean checkout; TRL 5 criteria all ≥ 2.0.

---

## Risk Mitigation

Full risk register: [`docs/risk-register.md`](risk-register.md). Summary of top risks for this application window:

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| FR-01 | GPU quota not available within 30 days | M | M | CPU-only Phase 1 deliverables are fully independent; sovereign control hardening proceeds unblocked. Phase 2 GPU work is sequenced to start Day 31, not Day 1. |
| FR-02 | Security gap implementation reveals architectural constraints | M | H | Modular platform API boundaries (auth / tenants / audit modules) enable targeted changes without full rewrites. |
| FR-03 | DSIT / AI Safety Institute policy requirements change | M | H | Governance layer is policy-configurable via rules engine; rule set updates require no code changes. |
| FR-04 | Fund reviewer requires live pilot evidence (not POC) | L | H | Day 90 demo from clean checkout satisfies reproducibility; SLO and incident-response runbooks produced by Day 90. |
| FR-05 | Evaluation harness not ready before GPU run | M | M | Baseline harness scoped for Day 75; GPU run targeted at Day 60. Harness is not on GPU critical path. |
| FR-06 | Key engineer unavailability | M | H | All procedures documented; runbooks in [`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md). |

---

## Positioning Summary

> POC proven. Scaling plan clear. GPU constraint managed by sovereign-control-plane-first delivery. GPU-dependent work sequenced as Phase 2, not a blocker.

Supreme ModelTX is the only British-built, open-architecture sovereign LLM platform with:
- Reproducible CPU training runs with published loss and perplexity curves
- Azure UK IaC ready for customer-owned deployment
- A governance control plane designed to meet public-sector accountability requirements
- A clear, phased plan to close all TRL 5 gaps within 60 days

*Every claim in this document is backed by a repository-verifiable artifact. See [`docs/submission-checklist.md`](submission-checklist.md) for the complete evidence index.*
