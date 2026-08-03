# Impact Metrics — Supreme ModelTX

> **Scope:** Measurable KPIs for technical and commercial progress, with baseline, current, and target values.  
> **Assessment date:** 2026-08-03  
> **Evidence anchors:** [`docs/trl-assessment.md`](trl-assessment.md), [`docs/funding-readiness.md`](funding-readiness.md), [`docs/poc-status.md`](poc-status.md).

---

## Metric definitions

**Baseline** — state at programme start (pre-fund, CPU-only POC).  
**Current** — state as of this document's assessment date.  
**Target** — committed outcome at the end of the 90-day funded window.  
**Evidence** — repository artifact or CI output that proves the value.

---

## KPI 1 — TRL Stage

*Measures overall platform maturity against the UK Technology Readiness Level scale.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | TRL 3 (analytical proof-of-concept) | Pre-repository state |
| Current | TRL 4 (component validation, controlled environment) | [`docs/trl-assessment.md`](trl-assessment.md) §Aggregate result |
| Target (Day 90) | TRL 5 (validation in relevant environment) | All TRL 5 security/governance criteria ≥ 2.0 — see [`docs/trl-assessment.md`](trl-assessment.md) §What is needed to reach TRL 5 |

---

## KPI 2 — Security / Governance Gap Closure Rate

*Measures closure of the four primary sovereign-control blockers (G-S1, G-S2, G-S3, G-G1) identified in the gap analysis.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | 0 / 4 gaps closed | [`docs/gap-analysis.md`](gap-analysis.md) §Security gaps |
| Current | 0 / 4 gaps closed | [`docs/gap-analysis.md`](gap-analysis.md) §Security gaps |
| Target (Day 30) | 4 / 4 gaps closed | All four merged with passing tests; TRL security/governance score 1.0 → 2.0 |
| Target (Day 60) | Policy engine + decision logging live (G-S4, G-G2) | [`docs/funding-readiness.md`](funding-readiness.md) §Phase 2 |

---

## KPI 3 — CI Test Pass Rate

*Measures code quality gate reliability — the percentage of CI runs on the main branch that pass all checks.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | Not tracked (pre-CI) | — |
| Current | Target: 100% on `main` branch | [`.github/workflows/`](../.github/workflows/) |
| Target (Day 90) | ≥ 98% pass rate on `main`; CI-gated evaluation baseline enforced | [`docs/funding-readiness.md`](funding-readiness.md) §3.3 |

---

## KPI 4 — Training Loss (T-Dev-6L, CPU baseline)

*Measures model learning quality on the canonical CPU training run.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | Not measured (pre-POC) | — |
| Current | ~7.83 validation loss at step 10/20 (tiny corpus slice) | [`docs/first-experiment-findings.md`](first-experiment-findings.md) |
| Target (Day 60) | Stable convergence curve on GPU with validation loss ≤ 4.0 at end of first full GPU run | [`docs/t_dev_6l_first_gpu_run.md`](t_dev_6l_first_gpu_run.md); artifact: `artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts/training_summary.json` |

---

## KPI 5 — Perplexity (T-Dev-6L, CPU baseline)

*Measures language model quality as perplexity on validation set.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | Not measured | — |
| Current | ~2 471 perplexity (tiny corpus slice, 20 steps) | [`docs/first-experiment-findings.md`](first-experiment-findings.md) |
| Target (Day 60) | Meaningful perplexity reduction vs CPU baseline on GPU run (target: ≤ 500 on comparable eval set) | [`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md); artifact: `artifacts/runs/t_dev_6l_first_gpu_run/benchmark_outputs/` |

---

## KPI 6 — Platform API Endpoint Coverage

*Measures the fraction of platform API routes protected by RBAC and tenant isolation controls.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | 0% (auth scaffold only; no RBAC, no isolation enforcement) | [`docs/poc-status.md`](poc-status.md) §RBAC row |
| Current | 0% — RBAC middleware not yet implemented | [`docs/gap-analysis.md`](gap-analysis.md) §G-S1 |
| Target (Day 30) | 100% of management and governance routes protected by `require_role` dependency; cross-tenant access blocked at service layer | [`docs/funding-readiness.md`](funding-readiness.md) §Phase 1 deliverables |

---

## KPI 7 — Audit Log Integrity

*Measures whether the audit log is tamper-evident (hash-chain linked) as required for sovereign governance.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | Not tamper-evident (append-only, no chain) | [`src/supreme_modeltx/platform_api/audit/log.py`](../src/supreme_modeltx/platform_api/audit/log.py) |
| Current | Not tamper-evident | [`docs/gap-analysis.md`](gap-analysis.md) §G-G1 |
| Target (Day 30) | SHA-256 hash-chain implemented; `verify_chain` utility passes on clean log and fails on tampered entry | [`docs/funding-readiness.md`](funding-readiness.md) §1.4 |

---

## KPI 8 — Reproducibility (Clean-checkout Demo)

*Measures whether the platform can be demonstrated end-to-end from a clean repository checkout — the gold standard for fund reviewer reproducibility.*

| Dimension | Value | Evidence |
|---|---|---|
| Baseline | Partial — setup script and smoke tests pass; full demo not documented | [`scripts/setup.sh`](../scripts/setup.sh), `tests/smoke/` |
| Current | Smoke tests and CPU training dry-run pass; `scripts/run_demo.sh` functional for model + API path | [`scripts/run_demo.sh`](../scripts/run_demo.sh) |
| Target (Day 90) | `bash scripts/run_demo.sh` completes with `Demo complete ✓` from clean checkout; includes GPU-backed inference example if GPU quota available | [`docs/funding-readiness.md`](funding-readiness.md) §3.5 |

---

## Summary table

| # | KPI | Baseline | Current | Target (Day 90) |
|---|---|---|---|---|
| 1 | TRL Stage | TRL 3 | TRL 4 | TRL 5 |
| 2 | Security gap closure (primary 4) | 0 / 4 | 0 / 4 | 4 / 4 by Day 30 |
| 3 | CI test pass rate | Not tracked | 100% (target) | ≥ 98% on `main` |
| 4 | Training loss (T-Dev-6L) | Not measured | ~7.83 (tiny slice) | ≤ 4.0 (GPU run, full corpus) |
| 5 | Perplexity (T-Dev-6L) | Not measured | ~2 471 (tiny slice) | ≤ 500 (GPU run, full corpus) |
| 6 | RBAC / API route coverage | 0% | 0% | 100% management routes by Day 30 |
| 7 | Audit log tamper-evidence | ❌ None | ❌ None | ✅ Hash-chain verified by Day 30 |
| 8 | Clean-checkout reproducibility | Partial | Smoke + dry-run | Full demo `✓` by Day 90 |

---

## Notes

- KPIs 4 and 5 (loss and perplexity) will be updated once the first GPU run completes. GPU quota is being requested in UK Azure South / UK West regions.
- KPIs 2, 6, and 7 are on the Phase 1 critical path (Days 1–30) and are fully achievable without GPU.
- All KPI targets are traceable to specific deliverables in [`docs/funding-readiness.md`](funding-readiness.md) and the gap backlog in [`docs/gap-analysis.md`](gap-analysis.md).
