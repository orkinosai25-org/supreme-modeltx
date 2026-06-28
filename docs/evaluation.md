# Evaluation Framework

Version: 0.1.0  
Last updated: 2026-06

---

## 1. Overview

This document describes the evaluation methodology for Supreme ModelTX, covering training metrics, benchmark scoring, safety assessments, and reproducibility standards expected for UK Sovereign AI Fund-aligned submissions.

---

## 2. Automatic Metrics

### 2.1 Training metrics

| Metric | Definition | Target (T-Dev-6L CPU baseline) |
|---|---|---|
| Training loss | Cross-entropy loss per token | Decreasing across steps |
| Validation loss | Held-out cross-entropy | < training loss (no overfitting) |
| Perplexity | `exp(val_loss)` | Decreasing; tracked per checkpoint |

Training run artifacts are written under `artifacts/runs/<run_name>/run_artifacts/`:

```text
training_summary.json     ← per-step loss, val loss, perplexity
training_summary.md       ← human-readable summary
samples.json / samples.md ← generated samples at each checkpoint
config_used.json          ← full config snapshot for reproducibility
```

See [`docs/run-artifacts.md`](run-artifacts.md) for the full schema.

### 2.2 Benchmark set

Fixed evaluation prompts are defined in `configs/benchmark_eval_set.json`.  
Canonical prompts for generation quality are in `configs/canonical_prompts.json`.  
Baseline performance records are in `configs/benchmark_baselines.json`.

Run the benchmark harness:

```bash
bash scripts/evaluate.sh
```

Or directly:

```bash
python -m supreme_modeltx.model_core.eval.perplexity \
  --config configs/real_training/t_dev_6l_first_run.json \
  --eval-manifest data/manifests/t_dev_6l_first_run.yaml
```

---

## 3. Safety and Alignment Evaluation

### 3.1 Failure mode catalogue

| Failure mode | Mitigation in current design |
|---|---|
| Hallucination / confabulation | Retrieval grounding layer (inference/retrieval_service.py) |
| Prompt injection | Input sanitisation hooks in platform API auth layer |
| PII leakage in outputs | Audit event logging; data manifest provenance tracking |
| Biased or harmful generation | Human-in-the-loop review gate in governance lifecycle |
| Uncontrolled deployment | Approval workflow in SMTX control plane (register → review → approve → deploy) |

### 3.2 Human-in-the-loop policy

All production deployments require:

1. **Registration** — model card submitted to registry with intended use and data provenance.
2. **Review gate** — human reviewer approves policy compliance before deployment state transitions.
3. **Audit trail** — all inference requests and usage events are logged to the immutable audit pipeline.

### 3.3 Data provenance

Training data lineage is captured in YAML manifests under `data/manifests/`. Each manifest records:

- source dataset references and checksums
- tokeniser version
- split sizes and sequence lengths
- preprocessing steps

---

## 4. Reproducibility Standards

| Requirement | Status |
|---|---|
| Config-driven training (no magic CLI overrides) | ✅ All training params in JSON config files under `configs/` |
| Artifact versioning per run | ✅ `run_artifacts/config_used.json` and checkpoints per run |
| Deterministic data loading | ✅ Manifest-driven, split-aware loaders |
| Test coverage (unit + smoke) | ✅ `tests/unit/` and `tests/smoke/` suites |
| CI pipeline for smoke tests | ✅ `.github/workflows/python-ci.yml` |
| GPU experiment reproducibility | ✅ Documented in `docs/first-gpu-experiment-findings.md` |

---

## 5. Evaluation Roadmap

Near-term milestones aligned with the 90-day delivery plan:

| Milestone | Target |
|---|---|
| Expand benchmark set to ≥50 fixed prompts | Day 30 |
| Add BLEU/ROUGE scoring for summarisation tasks | Day 45 |
| Formal red-team exercise against adversarial inputs | Day 60 |
| Model card draft aligned with DSIT/CDEI guidelines | Day 60 |
| Publish baseline + improved results comparison | Day 90 |

---

## 6. References

- [`docs/benchmarking.md`](benchmarking.md) — benchmarking workflow details
- [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md) — first GPU run evaluation findings
- [`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md) — expanded experiment findings
- [`configs/benchmark_eval_set.json`](../configs/benchmark_eval_set.json) — fixed evaluation prompts
- [`configs/benchmark_baselines.json`](../configs/benchmark_baselines.json) — baseline metrics
