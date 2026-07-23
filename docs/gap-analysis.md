# Gap Analysis: POC → Pilot/Deployment Readiness

> **Scope:** Evidence-backed blockers preventing transition from current POC posture to pilot/deployment readiness.  
> **Assessment date:** 2026-07-23  
> **Method:** Repository artifact review only (code, tests, workflows, and docs linked below).

---

## Current-state summary

`supreme-modeltx` has strong POC evidence across model core, API scaffolding, infra definitions, and documentation, but pilot readiness is blocked by control-plane hardening, policy/governance enforcement, and operational validation gaps.

Primary evidence anchors:

- [`docs/poc-status.md`](poc-status.md)
- [`docs/trl-assessment.md`](trl-assessment.md)
- [`docs/repository-readiness-review.md`](repository-readiness-review.md)

---

## Domain gap analysis

### 1) Architecture gaps

| Gap | Evidence | Risk/Impact | Severity | Effort | Linked issue |
|---|---|---|---|---|---|
| Tenant/workspace domain boundaries are incomplete | [`src/supreme_modeltx/platform_api/tenants/models.py`](../src/supreme_modeltx/platform_api/tenants/models.py), [`docs/poc-status.md`](poc-status.md) | Cross-tenant leakage risk and unclear ownership of resources | High | Medium | #4 |
| Inference providers are not abstracted behind a stable contract | [`src/supreme_modeltx/platform_api/api/engine.py`](../src/supreme_modeltx/platform_api/api/engine.py), [`docs/poc-status.md`](poc-status.md) | GPU/runtime adoption later will require invasive rewrites | Medium | Medium | #10 |
| Model lifecycle promotion/rollback is not fully operationalized | [`src/supreme_modeltx/platform_api/model_registry/registry.py`](../src/supreme_modeltx/platform_api/model_registry/registry.py), [`docs/poc-status.md`](poc-status.md) | Release management is manual and rollback reliability is uncertain | Medium | Medium | #9 |

### 2) Security gaps

| Gap | Evidence | Risk/Impact | Severity | Effort | Linked issue |
|---|---|---|---|---|---|
| RBAC middleware and per-role enforcement are missing | [`docs/poc-status.md`](poc-status.md), [`docs/trl-assessment.md`](trl-assessment.md) | Privileged operations may be reachable by any authenticated key | High | Medium | #5 |
| Scoped token lifecycle (expiry/rotation/revocation controls) is incomplete | [`src/supreme_modeltx/platform_api/auth/keys.py`](../src/supreme_modeltx/platform_api/auth/keys.py), [`docs/poc-status.md`](poc-status.md) | Long-lived credentials increase blast radius and weaken machine auth hygiene | High | Medium | #6 |
| Policy constraints (model/data/region/retention) are not enforced at runtime | [`docs/trl-assessment.md`](trl-assessment.md), [`docs/poc-status.md`](poc-status.md) | Sovereign and regulatory controls cannot be proven in operation | High | High | #8 |

### 3) Operations gaps

| Gap | Evidence | Risk/Impact | Severity | Effort | Linked issue |
|---|---|---|---|---|---|
| Usage rollups, chargeback view, and operational runbooks are incomplete | [`src/supreme_modeltx/platform_api/usage/metering.py`](../src/supreme_modeltx/platform_api/usage/metering.py), [`docs/poc-status.md`](poc-status.md) | Limited operational visibility and slower incident triage/recovery | Medium | Low | #12 |
| Production serving hardening (SLOs/alerts/recovery drills) is not evidenced | [`docs/trl-assessment.md`](trl-assessment.md), [`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md) | Reliability risk in pilot workloads and weak operational confidence | High | High | #12 |

### 4) Testing/validation gaps

| Gap | Evidence | Risk/Impact | Severity | Effort | Linked issue |
|---|---|---|---|---|---|
| Reproducible evaluation harness with baseline reporting is missing | [`docs/evaluation.md`](evaluation.md), [`docs/poc-status.md`](poc-status.md) | Progress/regression decisions cannot be consistently measured | Medium | Medium | #11 |
| CI-friendly lightweight evaluation mode is not yet integrated | [`scripts/evaluate.sh`](../scripts/evaluate.sh), [`docs/poc-status.md`](poc-status.md) | Quality gates are less effective for release readiness | Medium | Medium | #11 |

### 5) Governance/compliance gaps

| Gap | Evidence | Risk/Impact | Severity | Effort | Linked issue |
|---|---|---|---|---|---|
| Audit trail is append-only but not tamper-evident | [`src/supreme_modeltx/platform_api/audit/log.py`](../src/supreme_modeltx/platform_api/audit/log.py), [`docs/poc-status.md`](poc-status.md) | Limited forensic trust and weaker governance posture | High | Medium | #7 |
| Policy decision logging with explicit reason codes is missing | [`docs/trl-assessment.md`](trl-assessment.md), [`docs/poc-status.md`](poc-status.md) | Compliance evidence for denied/allowed decisions is insufficient | High | High | #8 |

---

## Ranked backlog (POC → pilot)

| Rank | Gap | Domain | Severity | Effort | Suggested owner | Suggested sequence | Follow-on issue |
|---:|---|---|---|---|---|---|---|
| 1 | Implement tenant/workspace model and hard isolation boundaries | Architecture + Security | High | Medium | Platform API team | Phase 1 foundation | #4 |
| 2 | Add RBAC policy matrix and middleware enforcement | Security | High | Medium | Platform API + Security | Phase 1 (after #4 domain scoping) | #5 |
| 3 | Complete scoped API keys/tokens lifecycle controls | Security | High | Medium | Platform API + Security | Phase 1 (parallel with #5) | #6 |
| 4 | Implement policy engine v1 with decision logging | Governance + Security | High | High | Governance/Policy + Platform API | Phase 2 (depends on identity/tenancy controls) | #8 |
| 5 | Build tamper-evident append-only audit chain + verification | Governance/Compliance | High | Medium | Platform API + Compliance | Phase 2 (alongside #8) | #7 |
| 6 | Build model registry promotion/rollback workflow | Architecture + MLOps | Medium | Medium | MLOps + Platform API | Phase 3 | #9 |
| 7 | Add inference provider abstraction (CPU/API now, GPU later) | Architecture + MLOps | Medium | Medium | MLOps/Inference team | Phase 3 | #10 |
| 8 | Implement evaluation harness and baseline reporting | Testing/Validation | Medium | Medium | Eval/QA + MLOps | Phase 4 (gates for #9/#10) | #11 |
| 9 | Add usage metering rollups, cost views, and runbooks | Operations | Medium | Low | Platform Ops/SRE | Phase 4 | #12 |
| 10 | Pilot operations hardening (SLOs, alerts, recovery drills) | Operations | High | High | Platform Ops/SRE | Phase 5 readiness gate | #12 |

---

## Suggested implementation sequence

1. **Security boundary foundation:** #4 → #5/#6  
2. **Governance enforcement:** #8 → #7  
3. **Lifecycle architecture stabilization:** #9 → #10  
4. **Validation and operationalization:** #11 → #12

Dependency rationale:

- #8 policy controls are materially stronger once tenancy/RBAC identity boundaries are in place (#4/#5).
- #11 evaluation gates are most useful when lifecycle flows (#9) and provider contracts (#10) are stable.
- #12 pilot runbooks/SLOs should be finalized only after security/governance controls are enforceable.

---

## Mapping to follow-on implementation issues

All identified blockers map directly to the issue set from #58:

- **#4** Multi-tenant workspace domain model
- **#5** RBAC middleware and role enforcement
- **#6** Scoped API keys and token lifecycle
- **#7** Immutable audit trail with tamper-evidence
- **#8** Policy engine v1 for sovereign controls
- **#9** Model registry lifecycle promotion/rollback
- **#10** Inference provider abstraction
- **#11** Evaluation harness and baseline reporting
- **#12** Usage metering, chargeback, and operational runbooks

No blocker above is left without a direct implementation path.
