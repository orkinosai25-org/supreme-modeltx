# Submission Checklist — Supreme ModelTX

> **Scope:** Complete evidence index for UK Sovereign AI Fund submission.  
> **Application narrative:** [`docs/application-evidence-bundle.md`](application-evidence-bundle.md)  
> **Impact metrics:** [`docs/impact-metrics.md`](impact-metrics.md)  
> **Last updated:** 2026-08-03

---

## 1. Required Evidence Links

### 1.1 POC and technical readiness

| Evidence item | Link | Status |
|---|---|---|
| POC status matrix (all 16 model-core rows) | [`docs/poc-status.md`](poc-status.md) | ✅ Committed (PR #59) |
| TRL 4 self-assessment with per-criterion scores | [`docs/trl-assessment.md`](trl-assessment.md) | ✅ Committed |
| Gap analysis: POC → pilot/deployment | [`docs/gap-analysis.md`](gap-analysis.md) | ✅ Committed |
| First CPU training experiment findings | [`docs/first-experiment-findings.md`](first-experiment-findings.md) | ✅ Committed |
| Expanded CPU benchmarked experiment findings | [`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md) | ✅ Committed |
| GPU run plan (T-Dev-6L first GPU run) | [`docs/t_dev_6l_first_gpu_run.md`](t_dev_6l_first_gpu_run.md) | ✅ Committed |
| First GPU experiment findings tracker | [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md) | ✅ Committed |
| Run artifact schema and inspection guide | [`docs/run-artifacts.md`](run-artifacts.md) | ✅ Committed |
| Benchmark methodology | [`docs/benchmarking.md`](benchmarking.md) | ✅ Committed |

### 1.2 Architecture and sovereignty

| Evidence item | Link | Status |
|---|---|---|
| Platform architecture overview | [`docs/architecture/overview.md`](architecture/overview.md) | ✅ Committed |
| Sovereignty design principles | [`docs/sovereignty/principles.md`](sovereignty/principles.md) | ✅ Committed |
| Sovereign AI application brief | [`docs/sovereign-ai/application-brief.md`](sovereign-ai/application-brief.md) | ✅ Committed |
| Architecture reference (flat) | [`docs/architecture.md`](architecture.md) | ✅ Committed |

### 1.3 Infrastructure and deployment

| Evidence item | Link | Status |
|---|---|---|
| Azure Bicep IaC (UK South / UK West) | [`infra/main.bicep`](../infra/main.bicep), [`infra/rbac.bicep`](../infra/rbac.bicep) | ✅ Committed |
| Azure UK GPU runner provisioning runbook | [`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md) | ✅ Committed |
| VM scaling guide | [`docs/vm-scaling.md`](vm-scaling.md) | ✅ Committed |
| GPU readiness and scaling plan | [`docs/gpu-readiness-scaling-plan.md`](gpu-readiness-scaling-plan.md) | ✅ Committed |

### 1.4 Delivery plan and risk

| Evidence item | Link | Status |
|---|---|---|
| Funding readiness and 90-day execution plan | [`docs/funding-readiness.md`](funding-readiness.md) | ✅ Committed |
| 90-day delivery plan (v0.1) | [`docs/delivery-plan-90d.md`](delivery-plan-90d.md) | ✅ Committed |
| Risk register | [`docs/risk-register.md`](risk-register.md) | ✅ Committed |
| Repository readiness review | [`docs/repository-readiness-review.md`](repository-readiness-review.md) | ✅ Committed |

### 1.5 Evaluation and quality

| Evidence item | Link | Status |
|---|---|---|
| Evaluation framework | [`docs/evaluation.md`](evaluation.md) | ✅ Committed |
| Test results reference | [`docs/test-results.md`](test-results.md) | ✅ Committed |
| TRL assessment | [`docs/trl-assessment.md`](trl-assessment.md) | ✅ Committed |

---

## 2. CI / Run Links

| Workflow | Purpose | Link |
|---|---|---|
| `ci.yml` — Continuous integration | Lint, test, smoke test on every PR | [`.github/workflows/`](../.github/workflows/) |
| `first-gpu-experiment.yml` | GPU-backed T-Dev-6L training and benchmark | [`.github/workflows/first-gpu-experiment.yml`](../.github/workflows/first-gpu-experiment.yml) |
| Run artifacts (CPU baseline) | Loss curves, perplexity, checkpoint hashes | [`artifacts/runs/t_dev_6l_first_run/run_artifacts/`](../artifacts/runs/t_dev_6l_first_run/run_artifacts/) |
| Run artifacts (expanded run) | Benchmarked CPU run | [`artifacts/runs/t_dev_6l_expanded_run/run_artifacts/`](../artifacts/runs/t_dev_6l_expanded_run/run_artifacts/) |
| Training config (GPU) | Canonical config for first GPU run | [`configs/real_training/t_dev_6l_first_gpu_run.json`](../configs/real_training/t_dev_6l_first_gpu_run.json) |
| Evaluation script | Full evaluate.sh with baseline output | [`scripts/evaluate.sh`](../scripts/evaluate.sh) |
| Demo script | End-to-end reproducibility check | [`scripts/run_demo.sh`](../scripts/run_demo.sh) |

> **Note:** Live CI run links for the fund submission should be copied from the GitHub Actions tab at time of submission and pasted here.  
> **Placeholder:** `https://github.com/orkinosai25-org/supreme-modeltx/actions` — replace with specific run URL.

---

## 3. Partner / Customer Proof Placeholders

*These items are placeholders to be populated before final submission.*

| Item | Contact / Organisation | Notes | Status |
|---|---|---|---|
| Letters of support / intent | _[Public sector partner TBC]_ | Letters from UK government department, NHS trust, or regulated financial institution confirming interest in sovereign LLM platform | ⬜ Pending |
| Pilot engagement evidence | _[Organisation TBC]_ | Documented discovery session, requirements workshop, or early access agreement | ⬜ Pending |
| Technical advisor validation | _[Advisor TBC]_ | Statement from independent technical reviewer confirming architecture soundness and TRL assessment | ⬜ Pending |
| Commercial pipeline evidence | _[Sales / BD TBC]_ | Pipeline tracker or signed NDA/MoU with potential customers | ⬜ Pending |
| Ecosystem / partner integrations | _[Integration partner TBC]_ | Evidence of any technology partnership, system integrator interest, or reseller agreement | ⬜ Pending |

---

## 4. Final QA Checklist

Complete every item before submitting the application.

### 4.1 Application narrative

- [ ] `docs/application-evidence-bundle.md` — every claim is linked to a repository artifact
- [ ] No vague statements (e.g., "state-of-the-art", "industry-leading") without supporting evidence
- [ ] GPU constraint addressed explicitly: sovereign-control-plane-first, Phase 2 sequencing
- [ ] Application text is copy-paste ready (no internal placeholders remaining)
- [ ] TRL stage stated accurately (TRL 4 as of 2026-08-03)

### 4.2 Evidence completeness

- [ ] All rows in Section 1 (Required Evidence Links) resolve to committed files
- [ ] CI run links in Section 2 point to real, passing workflow runs
- [ ] Partner/customer proof placeholders (Section 3) either populated or explicitly noted as pending with a completion date

### 4.3 Technical accuracy

- [ ] `docs/trl-assessment.md` scores reflect latest repository state
- [ ] `docs/gap-analysis.md` gap statuses are current (no resolved gaps still marked as open)
- [ ] `docs/risk-register.md` reviewed and dated within 30 days of submission
- [ ] `docs/funding-readiness.md` 90-day plan dates are anchored to actual fund start date

### 4.4 Impact metrics

- [ ] `docs/impact-metrics.md` — all 5–8 KPIs have baseline, current, and target values populated
- [ ] No KPI target is unsubstantiated; each traces to a delivery-plan milestone

### 4.5 Repository hygiene

- [ ] `README.md` "Readiness & Evidence" section links to all three new docs
- [ ] No secrets or credentials committed (`.env` excluded via `.gitignore`)
- [ ] All referenced file paths exist in the repository (no broken links)
- [ ] `THIRD_PARTY_NOTICES.md` up to date for all dependencies used

### 4.6 Reproducibility check

- [ ] `bash scripts/setup.sh` completes without errors on a clean checkout
- [ ] `python -m pytest tests/smoke/ -v` passes
- [ ] `bash scripts/run_demo.sh` completes with `Demo complete ✓`
- [ ] At least one training-run artifact exists in `artifacts/runs/` with a committed `training_summary.json`

---

## 5. Submission Day Actions

1. Run full reproducibility check (Section 4.6 above).
2. Capture live CI run URL from GitHub Actions and update Section 2.
3. Confirm partner/customer proof items or document pending status with dates.
4. Final read-through of `docs/application-evidence-bundle.md` — no placeholders, all links valid.
5. Export or link this checklist as part of the submission appendix.
