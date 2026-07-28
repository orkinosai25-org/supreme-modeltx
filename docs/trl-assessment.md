# TRL Self-Assessment — Supreme ModelTX

> **Scope:** Evidence-based readiness assessment against TRL 4–6 progression criteria for applied AI / sovereign platform programmes.  
> **Assessment date:** 2026-07-28  
> **Method:** Repository-artifact review only — every score is backed by a concrete file path, test, or document.  
> **Primary evidence anchor:** [`docs/poc-status.md`](poc-status.md) (merged PR #59 — POC evidence matrix).

---

## Scoring rubric

| Score | Meaning |
|---|---|
| 0 | No evidence in repository |
| 1 | Early scaffold / partial implementation |
| 2 | Working POC implementation with direct evidence (code + tests / docs) |
| 3 | Pilot-ready with hardening, operational controls, and relevant-environment validation |

### TRL stage thresholds used in this scorecard

| Stage | Label | Threshold |
|---|---|---|
| TRL 4 | Component validation in controlled environment | Average score **≥ 1.5**, core engineering and validation criteria **≥ 2** |
| TRL 5 | Validation in relevant environment | Average score **≥ 2.5**, security/governance and ops criteria **≥ 2** |
| TRL 6 | System demonstration in relevant environment | All criteria **≥ 2.5**, pilot deployed and documented |

---

## Criteria scorecard

### 1. Engineering Maturity

*Covers: code modularity, reproducibility, config-driven execution, CI pipeline.*

| Evidence | Score (0–3) | Rationale |
|---|---:|---|
| T-Dev-6L model architecture ([`src/supreme_modeltx/model_core/models/t_series/baseline.py`](../src/supreme_modeltx/model_core/models/t_series/baseline.py)), Pydantic v2 config schema ([`src/supreme_modeltx/model_core/config/schema.py`](../src/supreme_modeltx/model_core/config/schema.py)), training loop ([`src/supreme_modeltx/model_core/training/trainer.py`](../src/supreme_modeltx/model_core/training/trainer.py)), checkpoint management ([`src/supreme_modeltx/model_core/training/checkpoint.py`](../src/supreme_modeltx/model_core/training/checkpoint.py)), tokenizer workflow ([`src/supreme_modeltx/model_core/tokenizer/train.py`](../src/supreme_modeltx/model_core/tokenizer/train.py)), manifest-driven data pipeline ([`src/supreme_modeltx/model_core/data/manifest.py`](../src/supreme_modeltx/model_core/data/manifest.py)); FastAPI platform API with all major routers ([`src/supreme_modeltx/platform_api/api/app.py`](../src/supreme_modeltx/platform_api/api/app.py)); C# control-plane ([`control-plane/src/`](../control-plane/src/)); reproducible one-step setup ([`scripts/setup.sh`](../scripts/setup.sh)); CI workflows ([`.github/workflows/`](../.github/workflows/)). POC evidence matrix confirms all 16 Model Core rows ✅ Done. (Source: [`docs/poc-status.md`](poc-status.md) §1, §2, §3.) | **2.5** | Architecture is clearly layered (model core / platform API / control plane). All major modules implemented with stable boundaries. Reproducible from clean checkout. Pilot-scale hardening (distributed training recovery, production API error budgets) not yet evidenced. |

### 2. Validation Evidence

*Covers: experiment results, benchmark scores, test coverage, quality gates.*

| Evidence | Score (0–3) | Rationale |
|---|---:|---|
| CPU training run with loss + perplexity ([`docs/first-experiment-findings.md`](first-experiment-findings.md)); expanded run ([`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md)); GPU run plan + findings ([`docs/t_dev_6l_first_gpu_run.md`](t_dev_6l_first_gpu_run.md), [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md)); benchmark scorer ([`src/supreme_modeltx/model_core/eval/benchmark.py`](../src/supreme_modeltx/model_core/eval/benchmark.py)); run artifact schema ([`docs/run-artifacts.md`](run-artifacts.md)); 11 unit-test modules + smoke tests ([`tests/unit/`](../tests/unit/), [`tests/smoke/`](../tests/smoke/)). POC evidence matrix: Testing §6 all ✅ Done. (Source: [`docs/poc-status.md`](poc-status.md) §6.) | **2.0** | Multiple documented training runs with reproducible loss/perplexity curves. Broad unit-test coverage. No stable benchmark baseline yet; no CI-gated promotion thresholds; no live inference validation at relevant scale. |

### 3. Security / Governance

*Covers: authentication, RBAC, tenant isolation, audit trail, policy enforcement.*

| Evidence | Score (0–3) | Rationale |
|---|---:|---|
| scrypt-hashed API key issuance + revocation ([`src/supreme_modeltx/platform_api/auth/keys.py`](../src/supreme_modeltx/platform_api/auth/keys.py)); append-only audit log ([`src/supreme_modeltx/platform_api/audit/log.py`](../src/supreme_modeltx/platform_api/audit/log.py)); project/tenant model ([`src/supreme_modeltx/platform_api/tenants/models.py`](../src/supreme_modeltx/platform_api/tenants/models.py)). POC evidence matrix: RBAC ❌ Missing (Issue #5), policy engine ❌ Missing (Issue #8), tamper-evident audit ❌ Missing (Issue #7), multi-tenant cross-project blocking ⚠️ Partial. (Source: [`docs/poc-status.md`](poc-status.md) §Component Maturity Summary.) | **1.0** | Auth scaffold with secure key hashing and a partial audit log are present. RBAC middleware, policy engine, tamper-evident audit chaining, and enforced tenant isolation are all absent — four primary blockers to TRL 5. |

### 4. Ops Readiness

*Covers: infrastructure-as-code, deployment runbooks, SLOs, incident response.*

| Evidence | Score (0–3) | Rationale |
|---|---:|---|
| Azure Bicep templates ([`infra/main.bicep`](../infra/main.bicep), [`infra/rbac.bicep`](../infra/rbac.bicep), [`infra/modules/`](../infra/modules/)); Azure UK GPU runner runbook ([`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md)); VM scaling guide ([`docs/vm-scaling.md`](vm-scaling.md)); GPU readiness plan ([`docs/gpu-readiness-scaling-plan.md`](gpu-readiness-scaling-plan.md)); 90-day delivery plan ([`docs/delivery-plan-90d.md`](delivery-plan-90d.md)); risk register ([`docs/risk-register.md`](risk-register.md)). POC evidence matrix: Infrastructure §5 all ✅ Done. Production serving hardening ❌ Missing (Issue #12). (Source: [`docs/poc-status.md`](poc-status.md) §5, §Component Maturity Summary.) | **1.5** | IaC and provisioning runbooks are complete for Azure UK deployment. No production SLOs, no autoscaling configuration, no incident-response runbooks tied to live paths — required for TRL 5 ops evidence. |

### 5. Scalability Path

*Covers: distributed training hooks, GPU scaling plan, multi-tenant capacity isolation, horizontal serving.*

| Evidence | Score (0–3) | Rationale |
|---|---:|---|
| `torchrun` multi-GPU support + distributed setup ([`src/supreme_modeltx/model_core/training/distributed/setup.py`](../src/supreme_modeltx/model_core/training/distributed/setup.py)); mixed precision ([`src/supreme_modeltx/model_core/training/precision.py`](../src/supreme_modeltx/model_core/training/precision.py)); FSDP/DeepSpeed scaffold ([`src/supreme_modeltx/model_core/training/trainer.py`](../src/supreme_modeltx/model_core/training/trainer.py)); GPU corpus plan + first-subset manifest ([`docs/t_dev_6l_gpu_corpus_plan.md`](t_dev_6l_gpu_corpus_plan.md), [`data/manifests/t_dev_6l_gpu_corpus_v1_first_subset.yaml`](../data/manifests/t_dev_6l_gpu_corpus_v1_first_subset.yaml)); T-X multi-model orchestrator ([`tmodels/tx/orchestrator.py`](../tmodels/tx/orchestrator.py)); vLLM GPU inference scaffold ([`inference/vllm_server.py`](../inference/vllm_server.py)). POC evidence matrix: multi-tenant isolation ⚠️ Partial; experiment tracking ❌ Missing. (Source: [`docs/poc-status.md`](poc-status.md) §4, §Component Maturity Summary.) | **1.5** | Technical groundwork for distributed and GPU-scale execution is present and documented. Multi-tenant resource isolation and GPU-scale training validation have not yet been demonstrated end-to-end. |

---

## Aggregate result

| Criterion | Score (0–3) |
|---|---:|
| Engineering Maturity | 2.5 |
| Validation Evidence | 2.0 |
| Security / Governance | 1.0 |
| Ops Readiness | 1.5 |
| Scalability Path | 1.5 |
| **Total** | **8.5 / 15** |
| **Average** | **1.70 / 3** |

**Current TRL stage: TRL 4**  
The platform meets TRL 4 thresholds (engineering and validation criteria ≥ 2, repeatable lab-environment execution, documented POC evidence across all major sub-systems). It does **not** yet meet TRL 5 — security/governance and ops criteria remain at 1.0–1.5.

---

## What is needed to reach TRL 5

| Action required | Criterion raised | Target score |
|---|---|---|
| Implement RBAC middleware and per-role enforcement (Issue #5) | Security / Governance | → 2.0 |
| Enforce multi-tenant cross-project isolation at service layer (Issue #4) | Security / Governance | → 2.0 |
| Complete scoped API key expiry, rotation, and revocation (Issue #6) | Security / Governance | → 2.0 |
| Implement policy engine v1 with decision logging (Issue #8) | Security / Governance | → 2.5 |
| Add tamper-evident audit hash-chain and verification tooling (Issue #7) | Security / Governance | → 2.5 |
| Define and instrument SLOs, alerts, and rollback runbooks | Ops Readiness | → 2.5 |
| Document and execute pilot operational validation cycles | Ops Readiness | → 2.5 |
| Build evaluation harness with CI-gated baseline reports (Issue #11) | Validation Evidence | → 2.5 |

## What is needed to reach TRL 6

In addition to TRL 5 requirements:

| Action required | Criterion raised |
|---|---|
| Demonstrate multi-node GPU training at target T-101 scale | Scalability Path → 2.5 |
| Validate inference provider abstraction under pilot load (Issue #10) | Scalability Path → 2.5 |
| Complete model promotion/rollback workflow with lifecycle audit (Issue #9) | Engineering Maturity → 3.0 |
| Execute and document a full pilot deployment in an operationally relevant environment | All criteria |

---

## Confidence and limitations

- Every score reflects only repository-verifiable evidence; no off-repo claims are counted.
- Scores will be revised upward when missing controls are implemented and committed.
- See [`docs/gap-analysis.md`](gap-analysis.md) for the detailed blocker backlog with owner, effort, and dependency information.
