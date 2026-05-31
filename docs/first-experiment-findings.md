# First Benchmarked Training Experiment — Findings

**Run name:** `t_dev_6l_first_run`  
**Date (UTC):** 2026-05-31  
**Git commit:** `6448046e86ef2fd05a8f4c5346f39342ba8f2d34`  
**Benchmark:** `smtx-mini-code-reasoning-v1`

---

## Experiment overview

This is the first end-to-end benchmarked training run in this repository. It validates the full
pipeline from data preparation through tokenizer training, model training, checkpoint sample
generation, and benchmark scoring.

| Parameter | Value |
|---|---|
| Model variant | `t-dev-6l` |
| Parameters | 6,556,928 (~6.6M) |
| Vocab size | 512 |
| Hidden size | 256 |
| Layers | 6 |
| Attention heads | 8 |
| Max sequence length | 128 |
| Precision | float32 (mixed-precision disabled) |
| Device | CPU |
| Training steps | 20 |
| Batch size | 2 |
| Gradient accumulation | 1 |
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| LR schedule | constant |
| Corpus | 20 synthetic sentences (16 train / 4 val) |
| Tokenizer | SentencePiece BPE, vocab=512, v-first-run |

---

## Training metrics

| Checkpoint (step) | Validation loss | Perplexity |
|---|---|---|
| step 10 | 5.9428 | 381.02 |
| step 20 | 6.0096 | 407.32 |

**Best checkpoint:** step 10 (lower validation loss).

Training loss decreased from 5.75 (step 5) to 4.70 (step 20), indicating the model was fitting
the training distribution. Validation loss increased slightly from step 10 to step 20 (5.94 →
6.01), a hallmark of overfitting on the tiny 20-sentence corpus.

---

## Benchmark results

Benchmark: `smtx-mini-code-reasoning-v1` (4 tasks: 2 code, 2 reasoning)

| Checkpoint | Overall score | Code score | Reasoning score | Matched tasks |
|---|---|---|---|---|
| step 10 | 0.00 | 0.00 | 0.00 | 4 / 4 |
| step 20 | 0.00 | 0.00 | 0.00 | 4 / 4 |

All 4 benchmark prompts were matched correctly in both checkpoint samples (no missing tasks). Task
scores were 0.0 across the board because the model has not yet learned to produce meaningful
completions given only 20 training steps on a 20-sentence synthetic corpus.

### Open baseline reference (not runtime-scored)

| Model | Overall | Code | Reasoning |
|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | 0.50 | 0.40 | 0.60 |
| Mistral-7B-Instruct-v0.2 | 0.80 | 0.75 | 0.85 |

---

## What improved

- **Benchmark plumbing is end-to-end verified.** The full pipeline — data prep, tokenizer training,
  model training, checkpoint generation, sample generation, and benchmark scoring — executes
  without errors.
- **Prompt alignment is correct.** All 4 benchmark eval-set prompts were found in both checkpoint
  sample files (`matched_task_count=4`, `missing_task_count=0`). A bug in the canonical-prompts
  loader (`parents[5]` → `parents[4]`) was identified and fixed during this run; without it,
  samples would use built-in stub prompts that do not align with the benchmark eval set.
- **Artifact schema is correct.** `training_summary.json`, `samples/*.json`, `samples.json`,
  `samples.md`, `training_summary.md`, and `config_used.json` are all well-formed and complete.
- **Training loss decreased.** Loss fell from 5.75 → 4.70 over 20 steps, confirming the optimizer
  and training loop work correctly.

## What did not improve (expected at this scale)

- **Benchmark scores remain 0.00.** Twenty steps on a 20-sentence corpus produce only noise-level
  completions. No meaningful code or reasoning was generated — this is the expected baseline for a
  randomly-initialised 6.6M-parameter model with virtually no training.
- **Validation loss rose slightly (step 10 → 20).** Overfitting on the tiny corpus caused
  generalisation to degrade after step 10.
- **Completion quality is very low.** Samples show repeated tokens (e.g. `h h h h h...`) or empty
  strings, consistent with an untrained model producing maximum-likelihood garbage.

## Coverage gaps

| Gap | Details |
|---|---|
| Scale | The experiment uses a 20-sentence synthetic corpus; a real evaluation requires a domain-relevant corpus of at least tens of thousands of tokens. |
| Steps | 20 training steps is insufficient for any measurable task capability. A minimum of several thousand steps is needed to see non-zero benchmark scores. |
| GPU path | This run used CPU (`float32`). The mixed-precision and CUDA code paths have not been exercised end-to-end in a benchmarked run. |
| Benchmark breadth | The `smtx-mini-code-reasoning-v1` benchmark contains only 4 tasks. Broader capability coverage (e.g. HumanEval, HellaSwag) is not measured. |
| Checkpoint retention | Large `.pt` files (76 MB each) are excluded from the repository by `.gitignore`; only the run metadata and sample JSON files are committed. |

## Next optimisation priorities

1. **Real corpus:** Replace the 20-sentence synthetic corpus with a domain-representative JSONL
   dataset (≥50K tokens). This is the single most impactful change for non-zero benchmark scores.
2. **More steps:** Run for at least 1,000–5,000 steps to observe meaningful loss reduction and task
   improvement over training time.
3. **GPU run:** Execute training with `precision.enabled=true` and `dtype=bfloat16` on a CUDA
   device to validate the mixed-precision path and measure wall-clock throughput.
4. **Benchmark expansion:** Add or link to standard external benchmarks (HumanEval, HellaSwag,
   ARC-Easy) to complement the internal directional signal.
5. **Checkpoint upload:** Integrate artifact storage (e.g. Azure Blob Storage or GitHub LFS) so
   checkpoint `.pt` files are retained and retrievable for downstream evaluation.

---

## Artifacts

| Artifact | Path |
|---|---|
| Training summary (JSON) | `artifacts/runs/t_dev_6l_first_run/run_artifacts/training_summary.json` |
| Training summary (Markdown) | `artifacts/runs/t_dev_6l_first_run/run_artifacts/training_summary.md` |
| Checkpoint samples (step 10) | `artifacts/runs/t_dev_6l_first_run/run_artifacts/samples/checkpoint_step_00000010_samples.json` |
| Checkpoint samples (step 20) | `artifacts/runs/t_dev_6l_first_run/run_artifacts/samples/checkpoint_step_00000020_samples.json` |
| Consolidated samples (JSON) | `artifacts/runs/t_dev_6l_first_run/run_artifacts/samples.json` |
| Consolidated samples (Markdown) | `artifacts/runs/t_dev_6l_first_run/run_artifacts/samples.md` |
| Config snapshot | `artifacts/runs/t_dev_6l_first_run/run_artifacts/config_used.json` |
| Benchmark results (JSON) | `artifacts/runs/t_dev_6l_first_run/benchmark_outputs/benchmark_results.json` |
| Benchmark results (Markdown) | `artifacts/runs/t_dev_6l_first_run/benchmark_outputs/benchmark_results.md` |
| Chained CI workflow | `.github/workflows/first-experiment.yml` |

Checkpoint binary files (`*.pt`, 76 MB each) are not committed; they are excluded via `.gitignore`.
To reproduce this run, execute the `First Benchmarked Experiment` GitHub Actions workflow or follow
the steps in `.github/workflows/first-experiment.yml` locally.
