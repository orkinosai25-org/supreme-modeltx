# Expanded Benchmarked Training Experiment — Findings

**Run name:** `t_dev_6l_expanded_run`  
**Date (UTC):** 2026-06-01  
**Git commit:** `7c8ea81bb31d6b8980148a51b85385779a0799d5`  
**Benchmark:** `smtx-mini-code-reasoning-v1`

---

## Experiment overview

This run repeats the first benchmarked T-dev-6L experiment with a larger and more realistic corpus,
a longer schedule, and the same benchmark/reporting contract.

| Parameter | `t_dev_6l_first_run` | `t_dev_6l_expanded_run` |
|---|---:|---:|
| Corpus | 20 synthetic sentences (16 train / 4 val) | 288 mixed code/reasoning JSONL examples (240 train / 48 val) |
| Training steps | 20 | 200 |
| Eval cadence | every 10 steps | every 50 steps |
| Saved checkpoints | 2 | 4 |
| Device | CPU | CPU |
| Precision | float32 | float32 |

---

## Training metrics delta vs `t_dev_6l_first_run`

| Metric | `t_dev_6l_first_run` | `t_dev_6l_expanded_run` | Delta |
|---|---:|---:|---:|
| Latest validation loss | 6.0096 | 4.8126 | **-1.1969** |
| Latest perplexity | 407.32 | 123.08 | **-284.23** |
| Best checkpoint val loss | 5.9428 (step 10) | 4.7820 (step 150) | **-1.1608** |
| Best checkpoint perplexity | 381.02 | 119.37 | **-261.65** |

Validation metrics improved consistently through step 150 and slightly regressed at step 200,
indicating meaningful learning with early signs of overfitting on CPU-limited training.

---

## Benchmark deltas vs `t_dev_6l_first_run`

| Checkpoint | Overall score | Code score | Reasoning score | Matched tasks |
|---|---:|---:|---:|---:|
| first run best (step 10) | 0.00 | 0.00 | 0.00 | 4 / 4 |
| expanded run step 50 | 0.2917 | 0.3333 | 0.2500 | 4 / 4 |
| expanded run step 100 | 0.5000 | 0.5000 | 0.5000 | 4 / 4 |
| expanded run step 150 (best) | 1.0000 | 1.0000 | 1.0000 | 4 / 4 |
| expanded run step 200 | 1.0000 | 1.0000 | 1.0000 | 4 / 4 |

**Best-checkpoint delta vs baseline:**

- Overall: **+1.0000**
- Code: **+1.0000**
- Reasoning: **+1.0000**

---

## Sample quality delta

Compared with first-run outputs (mostly empty strings and repeated `h` tokens), expanded-run samples
showed stage-wise quality gains:

- **Step 50:** partial but syntactically valid code fragments (`def max_of_two...`).
- **Step 100:** complete function skeleton with conditional return logic.
- **Step 150/200:** benchmark prompts produce fully correct canonical outputs (`[0, 1, 4, 9]`, `yes`, `100`).

These outputs are still narrow to the benchmark prompts and should be treated as directional progress,
not broad capability claims.

---

## Compute and runtime constraints

- Run executed on **CPU only** (no CUDA path exercised).
- Wall-clock increased from ~4s (first run) to ~16m (expanded run) for 10x more steps and 14.4x more examples.
- Per-step throughput on CPU constrained practical schedule length and checkpoint frequency.
- Mixed precision remains disabled (`float32`) because GPU was unavailable in this run environment.

## What GPU access would unlock

1. **Meaningfully longer schedules** (1k–5k+ steps) within practical wall-clock limits.
2. **Larger batch sizes and sequence lengths** without severe CPU memory/latency trade-offs.
3. **Mixed precision validation** (`bfloat16`/`float16`) and CUDA training-path verification.
4. **Broader benchmark sweeps** across more checkpoints and external suites in one run window.

---

## Artifacts

| Artifact | Path |
|---|---|
| Training summary (JSON) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/training_summary.json` |
| Training summary (Markdown) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/training_summary.md` |
| Checkpoint samples (step 50) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/samples/checkpoint_step_00000050_samples.json` |
| Checkpoint samples (step 100) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/samples/checkpoint_step_00000100_samples.json` |
| Checkpoint samples (step 150) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/samples/checkpoint_step_00000150_samples.json` |
| Checkpoint samples (step 200) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/samples/checkpoint_step_00000200_samples.json` |
| Consolidated samples (JSON) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/samples.json` |
| Consolidated samples (Markdown) | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/samples.md` |
| Config snapshot | `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/config_used.json` |
| Benchmark results (JSON) | `artifacts/runs/t_dev_6l_expanded_run/benchmark_outputs/benchmark_results.json` |
| Benchmark results (Markdown) | `artifacts/runs/t_dev_6l_expanded_run/benchmark_outputs/benchmark_results.md` |
| Chained CI workflow | `.github/workflows/expanded-experiment.yml` |

Checkpoint binary files (`*.pt`) are excluded from the repository by `.gitignore`.
