# Gap Analysis: POC → Pilot / Deployment Readiness

> **Scope:** Evidence-backed blockers preventing transition from current POC posture to pilot and deployment readiness.  
> **Assessment date:** 2026-07-28  
> **Method:** Repository artifact review only — every gap references a concrete file path, test result, or doc.  
> **Primary evidence anchors:** [`docs/poc-status.md`](poc-status.md) (PR #59), [`docs/trl-assessment.md`](trl-assessment.md).

**Effort key:** S = days–1 week · M = 1–3 weeks · L = 3–6 weeks  
**Owner roles:** Platform · Security · MLOps · Product

---

## Current-state summary

`supreme-modeltx` has strong POC evidence across model core, API scaffolding, infra definitions, and documentation (TRL 4). Pilot readiness is blocked by four primary control-plane gaps: RBAC enforcement, policy engine, tamper-evident audit chaining, and enforced tenant isolation. Secondary blockers include production serving hardening, inference provider abstraction, and a CI-gated evaluation harness.

---

## Gap register

### Security gaps

| # | Description | Impact / Risk | Priority | Effort | Owner | Dependency |
|---|---|---|---|---|---|---|
| G-S1 | **RBAC middleware absent** — no per-role API guards; any valid key can call any route ([`docs/poc-status.md`](poc-status.md) §RBAC row, Issue #5) | Privileged model management and governance operations reachable by any authenticated caller; blocks sovereign-control claims | High | M | Security | G-S3 (identity scope) |
| G-S2 | **Multi-tenant isolation not enforced** — project model exists ([`src/supreme_modeltx/platform_api/tenants/models.py`](../src/supreme_modeltx/platform_api/tenants/models.py)) but cross-project data access is not blocked at service layer ([`docs/poc-status.md`](poc-status.md) §Multi-tenant row, Issue #4) | Cross-tenant data leakage; prevents multi-customer pilot | High | M | Platform | None |
| G-S3 | **Scoped token lifecycle incomplete** — revocation exists but expiry enforcement is absent ([`src/supreme_modeltx/platform_api/auth/keys.py`](../src/supreme_modeltx/platform_api/auth/keys.py), [`docs/poc-status.md`](poc-status.md) §Scoped token row, Issue #6) | Long-lived credentials increase blast radius; machine-auth hygiene non-compliant with public-sector requirements | High | S | Security | None |
| G-S4 | **Policy engine absent** — no configurable model/data/region/retention controls ([`docs/poc-status.md`](poc-status.md) §Policy engine row, Issue #8) | Sovereign and regulatory controls cannot be proven or audited in operation; TRL 5 blocker | High | L | Platform, Security | G-S1, G-S2 |

### Governance gaps

| # | Description | Impact / Risk | Priority | Effort | Owner | Dependency |
|---|---|---|---|---|---|---|
| G-G1 | **Audit log not tamper-evident** — events are append-only but no hash-chain linking ([`src/supreme_modeltx/platform_api/audit/log.py`](../src/supreme_modeltx/platform_api/audit/log.py), [`docs/poc-status.md`](poc-status.md) §Tamper-evident row, Issue #7) | Audit events can be silently altered; insufficient for incident forensics or grant/regulatory review | High | M | Security | None |
| G-G2 | **Policy decision logging absent** — no explicit allow/deny reason codes emitted from routing or governance layer ([`docs/trl-assessment.md`](trl-assessment.md) §Security/Governance) | Compliance evidence for sovereign-control decisions cannot be reconstructed post-incident | High | M | Security | G-S4 |

### MLOps / lifecycle gaps

| # | Description | Impact / Risk | Priority | Effort | Owner | Dependency |
|---|---|---|---|---|---|---|
| G-M1 | **Inference provider not abstracted** — `InferenceEngine` coupled to a single checkpoint path ([`src/supreme_modeltx/platform_api/api/engine.py`](../src/supreme_modeltx/platform_api/api/engine.py), [`docs/poc-status.md`](poc-status.md) §Inference integration row, Issue #10) | Adopting GPU/vLLM backends later requires invasive rewrites; testing across providers is blocked | Medium | M | MLOps | None |
| G-M2 | **Model promotion/rollback not operationalized** — registry stages exist but lifecycle transitions and rollback are manual ([`src/supreme_modeltx/platform_api/model_registry/registry.py`](../src/supreme_modeltx/platform_api/model_registry/registry.py), Issue #9) | Release management is unreliable; rollback confidence is low in pilot incidents | Medium | M | MLOps | G-M1 |
| G-M3 | **Evaluation harness incomplete** — perplexity is reported per run but no machine-readable baseline report, no CI promotion threshold ([`docs/evaluation.md`](evaluation.md), [`docs/poc-status.md`](poc-status.md) §Experiment tracking row, Issue #11) | Quality regressions can ship undetected; model release decisions are subjective | Medium | M | MLOps | G-M2 |

### Operations gaps

| # | Description | Impact / Risk | Priority | Effort | Owner | Dependency |
|---|---|---|---|---|---|---|
| G-O1 | **Production serving not hardened** — no SLO definitions, no autoscaling, no incident-response runbooks ([`docs/trl-assessment.md`](trl-assessment.md) §Ops Readiness, [`docs/poc-status.md`](poc-status.md) §Production serving row, Issue #12) | Reliability risk in pilot; weak confidence for fund/grant reviewers requiring operational evidence | High | L | Platform | G-M1 |
| G-O2 | **Usage metering rollups and cost views incomplete** — token ledger exists ([`src/supreme_modeltx/platform_api/usage/metering.py`](../src/supreme_modeltx/platform_api/usage/metering.py)) but no aggregated cost view or chargeback report (Issue #12) | Limited operational visibility; budget forecasting and per-tenant billing are manual | Medium | S | Product | None |

---

## Ranked execution backlog

| Rank | Gap | Priority | Effort | Owner | Dependency |
|---:|---|---|---|---|---|
| 1 | G-S2 — Enforce multi-tenant isolation boundaries | High | M | Platform | None |
| 2 | G-S3 — Complete scoped token expiry and lifecycle | High | S | Security | None |
| 3 | G-S1 — Implement RBAC middleware and per-role enforcement | High | M | Security | G-S3 |
| 4 | G-G1 — Add tamper-evident audit hash-chain and verification | High | M | Security | None |
| 5 | G-S4 — Implement policy engine v1 with decision logging | High | L | Platform, Security | G-S1, G-S2 |
| 6 | G-G2 — Wire policy decision logging (allow/deny reason codes) | High | M | Security | G-S4 |
| 7 | G-M1 — Inference provider abstraction (CPU/GPU swappable) | Medium | M | MLOps | None |
| 8 | G-M2 — Model registry promotion/rollback workflow | Medium | M | MLOps | G-M1 |
| 9 | G-M3 — Evaluation harness with CI-gated baseline reports | Medium | M | MLOps | G-M2 |
| 10 | G-O2 — Usage metering rollups and cost views | Medium | S | Product | None |
| 11 | G-O1 — Production serving hardening (SLOs, alerts, runbooks) | High | L | Platform | G-M1 |

---

## Prioritized execution sequence — next 4–6 weeks

### Week 1–2 · Security boundary foundation

- **G-S3** — Implement token expiry enforcement and rotation in [`auth/keys.py`](../src/supreme_modeltx/platform_api/auth/keys.py) (Issue #6). Unblocks G-S1.
- **G-S2** — Add cross-project access check middleware in [`tenants/`](../src/supreme_modeltx/platform_api/tenants/) (Issue #4). Independent; highest compliance risk.
- **G-G1** — Add SHA-256 hash-chain to [`audit/log.py`](../src/supreme_modeltx/platform_api/audit/log.py) + `verify_chain` utility (Issue #7). Independent; short effort.

### Week 2–3 · RBAC and policy enforcement

- **G-S1** — Implement role model and `require_role` FastAPI dependency; apply to management routes (Issue #5). Requires G-S3 identity scope.
- **G-O2** — Add usage rollup endpoint and chargeback view (Issue #12). Short effort; high Product visibility.

### Week 3–5 · Policy engine and governance logging

- **G-S4** — Build policy engine v1: configurable model/data/region/retention rules, explicit allow/deny evaluation (Issue #8). Depends on G-S1, G-S2.
- **G-G2** — Integrate decision logging into policy engine responses (Issue #8 extension). Depends on G-S4.

### Week 4–6 · Lifecycle, evaluation, and ops readiness

- **G-M1** — Extract inference provider interface; CPU and vLLM implementations (Issue #10). Independent.
- **G-M2** — Wire model promotion/rollback to registry stages with audit events (Issue #9). Depends on G-M1.
- **G-M3** — Build `eval/harness.py` and CI baseline report; add pass/fail threshold gate (Issue #11). Depends on G-M2.
- **G-O1** — Define SLOs, add alerting config, and draft incident-response runbook (Issue #12). Depends on G-M1.

### TRL gate check (end of week 6)

Security/Governance average ≥ 2.0, Ops Readiness ≥ 2.0, Validation Evidence ≥ 2.5 → **TRL 5 claim is supportable**.

---

## Immediate next implementation PR recommended

**PR title:** `feat: security boundary foundation — tenant isolation, token expiry, tamper-evident audit`  
**Scope:** G-S2 + G-S3 + G-G1 (Issues #4, #6, #7)  
**Why first:** These three gaps are mutually independent, have the shortest combined effort (M + S + M), and together remove the most critical security and governance blockers before RBAC and policy work begins.
