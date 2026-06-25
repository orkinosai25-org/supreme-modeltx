# 90-Day Delivery Plan

Version: 0.1.0  
Last updated: 2026-06

---

## Overview

This plan covers the 90-day delivery roadmap for Supreme ModelTX under a UK Sovereign AI Fund allocation.  
The goal is to advance from a working scaffold with reproducible CPU training runs to a GPU-backed, benchmarked, and governance-hardened sovereign AI platform with a demonstrable public-sector use case.

---

## Assumptions

1. GPU allocation (minimum one A100 or equivalent, UK Azure region) is available from Day 1.
2. One or two engineers are working on the project full-time.
3. No additional hires are required within this 90-day window; specialist input (security, legal) is advisory.
4. Deliverables are released incrementally with reproducible evidence at each milestone.

---

## Milestones

### Phase 1 — Stabilise and benchmark (Days 1–30)

| # | Deliverable | Owner | Done when |
|---|---|---|---|
| 1.1 | GPU environment provisioned and validated (UK Azure region) | Eng | `scripts/validate_first_gpu_environment.sh` passes |
| 1.2 | T-Dev-6L first full GPU training run completed | Eng | Checkpoint + perplexity emitted to `artifacts/runs/` |
| 1.3 | Benchmark eval set expanded to ≥ 50 fixed prompts | Eng | `configs/benchmark_eval_set.json` updated; `scripts/evaluate.sh` runs clean |
| 1.4 | `.env.example` and `scripts/setup.sh` merged and documented | Eng | README setup section verified end-to-end |
| 1.5 | Baseline benchmark results published | Eng | `results/baseline.json` committed |

### Phase 2 — Governance and safety (Days 31–60)

| # | Deliverable | Owner | Done when |
|---|---|---|---|
| 2.1 | Control-plane approval workflow wired end-to-end (register → review → deploy) | Eng | Integration test passes; audit log emitted |
| 2.2 | Model card drafted (T-Dev-6L) per DSIT/CDEI template | Eng | `docs/model-card.md` merged |
| 2.3 | Red-team / adversarial input exercise documented | Eng + Advisor | Findings in `docs/evaluation.md` section 3.1 |
| 2.4 | Data provenance review and update of all manifests | Eng | Manifests include checksums and licence fields |
| 2.5 | Risk register reviewed and updated | PM/Eng | `docs/risk-register.md` version 0.2 merged |

### Phase 3 — Public-sector use case and demo (Days 61–90)

| # | Deliverable | Owner | Done when |
|---|---|---|---|
| 3.1 | Flagship public-sector use case implemented (knowledge-grounded Q&A) | Eng | `scripts/run_demo.sh` runs end-to-end with example output |
| 3.2 | Improved benchmark results published (GPU vs CPU baseline) | Eng | `results/latest.json` committed with delta vs baseline |
| 3.3 | API demo walkthrough (Swagger UI + curl examples) | Eng | README demo section and `scripts/run_demo.sh` tested |
| 3.4 | Deployment IaC reviewed for UK-sovereign constraints | Eng + Advisor | `infra/` and `deployment/iac/` reviewed; findings documented |
| 3.5 | 90-day summary report produced | PM | One-page PDF / markdown report for fund reviewers |

---

## Go / No-Go Criteria

| Gate | Criterion |
|---|---|
| Day 30 | GPU run completes; reproducible benchmark baseline exists |
| Day 60 | Governance workflow passes end-to-end; no critical red-team findings unaddressed |
| Day 90 | Demo runs from clean checkout; improved metrics exceed baseline by measurable margin |

---

## Budget Buckets (indicative)

| Category | Indicative share |
|---|---|
| GPU compute (Azure UK, A100 or NC-series) | ~50% |
| Engineering time (contract or in-house) | ~35% |
| Advisory (security, legal, DSIT alignment) | ~10% |
| Tooling and infrastructure (storage, CI minutes) | ~5% |

---

## Risks and Mitigations

See [`docs/risk-register.md`](risk-register.md) for the full risk register.

Key risks in this window:

| Risk | Impact | Mitigation |
|---|---|---|
| GPU quota delay (Azure UK region) | Phase 1 slips | Pre-request quota; fallback to CPU-only Phase 1 deliverables |
| Dependency on proprietary data sources | Evaluation quality limited | Use open datasets; document substitution plan |
| Regulatory / policy change (DSIT/AI Safety Institute) | Governance layer requires rework | Monitor DSIT publications; keep governance layer modular |

---

## References

- [`docs/gpu-readiness-scaling-plan.md`](gpu-readiness-scaling-plan.md) — GPU readiness details
- [`docs/risk-register.md`](risk-register.md) — full risk register
- [`docs/evaluation.md`](evaluation.md) — evaluation framework
- [`docs/architecture.md`](architecture.md) — system architecture
