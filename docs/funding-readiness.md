# Funding Readiness — Supreme ModelTX

> **Scope:** Application-ready summary of 90-day execution plan, sovereign-first strategy under GPU constraint, milestones, measurable outcomes, and risk register.  
> **Assessment date:** 2026-07-30  
> **Evidence anchors:** [`docs/poc-status.md`](poc-status.md) (PR #59), [`docs/trl-assessment.md`](trl-assessment.md), [`docs/gap-analysis.md`](gap-analysis.md).

---

## Current position

Supreme ModelTX is at **TRL 4** (see [`docs/trl-assessment.md`](trl-assessment.md) §Aggregate result).

| Layer | Status | Evidence |
|---|---|---|
| Model core | ✅ Working POC — T-Dev-6L trains, checkpoints, and emits perplexity | [`docs/first-experiment-findings.md`](first-experiment-findings.md), [`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md) |
| Platform API | ✅ Scaffold — FastAPI routers, auth, tenant model, audit log | [`src/supreme_modeltx/platform_api/`](../src/supreme_modeltx/platform_api/) |
| Control plane | ✅ C# ASP.NET Core governance scaffold | [`control-plane/src/`](../control-plane/src/) |
| Infrastructure | ✅ Azure Bicep IaC for UK South / UK West | [`infra/main.bicep`](../infra/main.bicep), [`infra/rbac.bicep`](../infra/rbac.bicep) |
| Security controls | ⚠️ Partial — auth scaffold present; RBAC, policy engine, tamper-evident audit absent | [`docs/gap-analysis.md`](gap-analysis.md) §G-S1–G-S4 |
| GPU compute | ⚠️ Constrained — GPU path coded and preflight-verified; training not yet executed at scale | [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md) |

---

## Sovereign-control-plane-first strategy

GPU compute is the primary constraint. The platform is architected so that the **highest-value sovereign differentiators — governance, RBAC, policy engine, tamper-evident audit** — can be fully hardened *before* GPU allocation arrives. This is intentional: fund reviewers and public-sector customers care most about sovereignty guarantees, not raw model throughput at this stage.

**What is deliverable without GPU (CPU-only phase):**

| Deliverable | Status | Effort |
|---|---|---|
| Enforce multi-tenant isolation at service layer (G-S2) | Gap | M |
| Scoped token expiry and lifecycle (G-S3) | Gap | S |
| RBAC middleware with per-role enforcement (G-S1) | Gap | M |
| Tamper-evident audit hash-chain (G-G1) | Gap | M |
| Policy engine v1 with configurable rules and decision logging (G-S4, G-G2) | Gap | L |
| Usage metering rollups and cost views (G-O2) | Gap | S |
| Evaluation harness with CI baseline reports (G-M3) | Gap | M |
| Control-plane approval workflow wired end-to-end | In progress | M |
| Model card (T-Dev-6L) per DSIT/CDEI template | Planned | S |
| 90-day summary report for fund reviewers | Planned | S |

**What requires GPU:**

| Deliverable | GPU dependency | Notes |
|---|---|---|
| T-Dev-6L full GPU training run | 1× A100, UK Azure region | Config + preflight ready: [`configs/real_training/t_dev_6l_first_gpu_run.json`](../configs/real_training/t_dev_6l_first_gpu_run.json) |
| Stable benchmark baseline at GPU scale | Follows GPU run | Eval harness ready once G-M3 complete |
| T-101 (7B) architecture validation | Multi-GPU | Roadmap item; not in this 90-day window |
| vLLM production inference hardening | GPU inference node | [`inference/vllm_server.py`](../inference/vllm_server.py) scaffold ready |

---

## 90-day execution plan

### Phase 1 — Security boundary foundation (Days 1–30)

**Goal:** Remove the four highest-priority sovereign-control blockers (G-S2, G-S3, G-S1, G-G1). Achievable without GPU; directly raises TRL 5 eligibility.

| # | Deliverable | Gap | Effort | Done when |
|---|---|---|---|---|
| 1.1 | Enforce cross-project isolation in `tenants/` service layer | G-S2 | M | Unit test blocks cross-tenant access; no code path leaks data across projects |
| 1.2 | Implement token expiry enforcement and rotation in `auth/keys.py` | G-S3 | S | Expired tokens return 401; rotation produces new scoped token |
| 1.3 | Add `require_role` FastAPI dependency; apply to management routes | G-S1 | M | Role-gated routes return 403 to under-privileged callers |
| 1.4 | Add SHA-256 hash-chain to `audit/log.py` + `verify_chain` utility | G-G1 | M | Chain verification passes on clean log; fails on tampered entry |
| 1.5 | GPU environment pre-requested and quota confirmed in UK Azure region | — | S | Quota approval documented; fallback CPU plan active |

**Go / No-Go (Day 30):** G-S2, G-S3, G-S1, G-G1 all merged with passing tests; security/governance TRL score rises from 1.0 → 2.0.

---

### Phase 2 — Policy engine and governance hardening (Days 31–60)

**Goal:** Implement policy engine and governance logging; wire control-plane approval workflow. GPU run triggered if quota available.

| # | Deliverable | Gap | Effort | Done when |
|---|---|---|---|---|
| 2.1 | Policy engine v1: configurable model/data/region/retention rules with decision logging | G-S4, G-G2 | L | Policy decisions are logged with allow/deny reason codes; rules configurable without code changes |
| 2.2 | Usage metering rollups and per-tenant cost view endpoint | G-O2 | S | `GET /usage/summary` returns token counts and cost estimates per project |
| 2.3 | Control-plane approval workflow wired end-to-end (register → review → deploy) | — | M | Integration test passes; audit event emitted on each state transition |
| 2.4 | Model card for T-Dev-6L drafted per DSIT/CDEI template | — | S | `docs/model-card.md` merged with intended use, limitations, and data provenance |
| 2.5 | First GPU training run completed (if quota available) | — | M | Checkpoint + perplexity emitted to `artifacts/runs/t_dev_6l_first_gpu_run/` |

**Go / No-Go (Day 60):** Policy engine integrated; governance workflow passes end-to-end; no critical security gaps unresolved; TRL 5 claim supportable.

---

### Phase 3 — Lifecycle, evaluation, and demo readiness (Days 61–90)

**Goal:** Operationalize model lifecycle, establish CI-gated evaluation baseline, produce public-sector demo, and publish 90-day evidence pack.

| # | Deliverable | Gap | Effort | Done when |
|---|---|---|---|---|
| 3.1 | Inference provider abstraction (CPU/GPU swappable) | G-M1 | M | `InferenceEngine` accepts backend config; CPU and vLLM implementations pass tests |
| 3.2 | Model registry promotion/rollback workflow with audit events | G-M2 | M | `promote`/`rollback` transitions logged; rollback restores prior serving version |
| 3.3 | Evaluation harness with CI-gated baseline reports | G-M3 | M | `eval/harness.py` produces machine-readable `results/baseline.json`; CI fails on regression |
| 3.4 | Production serving SLOs, alerting config, and incident-response runbook | G-O1 | L | `docs/slo-runbook.md` drafted; alert definitions committed to `infra/` |
| 3.5 | Public-sector use case demo (knowledge-grounded Q&A) | — | M | `scripts/run_demo.sh` runs end-to-end with documented example output |
| 3.6 | 90-day summary report for fund reviewers | — | S | One-page markdown report in `docs/`; evidence linked to every claim |

**Go / No-Go (Day 90):** Demo runs from clean checkout; improved metrics exceed CPU baseline by measurable margin; all TRL 5 criteria ≥ 2.0; funding evidence pack complete.

---

## Measurable outcomes

| Outcome | Metric | Target |
|---|---|---|
| Security hardening | All four security gaps (G-S1–G-S4) closed | 100% by Day 60 |
| Governance logging | Policy decisions logged with reason codes | Every governance event by Day 60 |
| Evaluation regression gating | CI baseline report generated and enforced | Every PR from Day 75 |
| TRL progression | TRL score rises from 4 → 5 | By Day 60 |
| GPU training | First full GPU run checkpoint produced | By Day 60 (if quota available) |
| Demo quality | Demo runs end-to-end from clean checkout | Day 90 |

---

## Risk register (funding window)

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| FR-01 | GPU quota not available within 30 days | M | M | CPU-only Phase 1+2 deliverables are independent; sovereign control hardening proceeds unblocked |
| FR-02 | Security gap implementation reveals architectural constraints | M | H | Modular platform API boundaries enable targeted changes without full rewrites |
| FR-03 | DSIT / AI Safety Institute policy requirements change | M | H | Governance layer is policy-configurable; rule set can be updated without code changes |
| FR-04 | Fund reviewer requires live pilot evidence (not POC) | L | H | Day 90 demo from clean checkout satisfies reproducibility requirement; operationalized with G-O1 |
| FR-05 | Evaluation harness not ready before GPU run | M | M | Baseline harness (G-M3) scoped for Day 75; GPU run scheduled for Day 60 target |
| FR-06 | Key engineer unavailability | M | H | All procedures documented in runbooks; [`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md) |

Full risk register: [`docs/risk-register.md`](risk-register.md).

---

## Evidence index

Every funding claim in this document is backed by a repository-verifiable artifact:

| Claim | Evidence |
|---|---|
| Working POC with CPU training | [`docs/first-experiment-findings.md`](first-experiment-findings.md), [`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md) |
| GPU path coded and preflight-verified | [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md), [`configs/real_training/t_dev_6l_first_gpu_run.json`](../configs/real_training/t_dev_6l_first_gpu_run.json) |
| Azure UK IaC ready | [`infra/main.bicep`](../infra/main.bicep), [`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md) |
| Security gap register with effort/priority | [`docs/gap-analysis.md`](gap-analysis.md) |
| TRL 4 assessment with per-criterion scores | [`docs/trl-assessment.md`](trl-assessment.md) |
| Full POC status matrix | [`docs/poc-status.md`](poc-status.md) (PR #59) |
| 90-day delivery plan (v0.1) | [`docs/delivery-plan-90d.md`](delivery-plan-90d.md) |
| Operational risk register | [`docs/risk-register.md`](risk-register.md) |
