# First Self-Hosted GPU T-Dev-6L Experiment — Findings

**Run name:** `t_dev_6l_first_gpu_run`  
**Workflow:** `.github/workflows/first-gpu-experiment.yml`  
**Review date (UTC):** 2026-06-02

---

## Status

This findings document is now the canonical review location for the first self-hosted GPU
T-Dev-6L run.

At review time, the repository does not yet include committed artifacts under
`artifacts/runs/t_dev_6l_first_gpu_run/`, so CPU-vs-GPU deltas below are intentionally marked
as pending until the first completed self-hosted GPU execution outputs are preserved.

---

## Required artifact contract (to preserve from completed run)

| Artifact | Expected path |
|---|---|
| Training summary (JSON) | `artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts/training_summary.json` |
| Training summary (Markdown) | `artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts/training_summary.md` |
| Consolidated samples (JSON) | `artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts/samples.json` |
| Consolidated samples (Markdown) | `artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts/samples.md` |
| Checkpoint samples | `artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts/samples/checkpoint_step_*_samples.json` |
| Benchmark results (JSON) | `artifacts/runs/t_dev_6l_first_gpu_run/benchmark_outputs/benchmark_results.json` |
| Benchmark results (Markdown) | `artifacts/runs/t_dev_6l_first_gpu_run/benchmark_outputs/benchmark_results.md` |
| CPU-vs-GPU comparison (JSON) | `artifacts/runs/t_dev_6l_first_gpu_run/comparison_vs_cpu/comparison_vs_cpu.json` |
| CPU-vs-GPU comparison (Markdown) | `artifacts/runs/t_dev_6l_first_gpu_run/comparison_vs_cpu/comparison_vs_cpu.md` |

---

## CPU-vs-GPU deltas (pending first completed GPU run artifacts)

Comparison baselines:

- `t_dev_6l_first_run`
- `t_dev_6l_expanded_run`

| Metric | vs `t_dev_6l_first_run` | vs `t_dev_6l_expanded_run` |
|---|---:|---:|
| Best validation loss delta (GPU-CPU) | pending | pending |
| Best perplexity delta (GPU-CPU) | pending | pending |
| Benchmark overall delta | pending | pending |
| Benchmark code delta | pending | pending |
| Benchmark reasoning delta | pending | pending |
| Runtime / throughput delta | pending | pending |

Sample quality comparison remains pending until checkpoint sample artifacts for the first completed
GPU run are available for side-by-side review.

---

## Comparison honesty checks (must remain true before publishing deltas)

- Benchmark definition matches: `smtx-mini-code-reasoning-v1` contract.
- Tokenizer model path matches across compared runs (workflow enforces this).
- Corpus/materialization assumptions are stated explicitly (first subset manifest for GPU run).
- Artifact schema remains aligned with prior CPU runs (`training_summary.*`, `samples.*`,
  `benchmark_results.*`, `comparison_vs_cpu/*`).
- Any partial comparison is labeled as partial.

---

## Execution environment facts (to fill from first completed run)

- GPU class/count: pending
- VRAM envelope: pending
- Precision mode: pending
- Wall-clock runtime: pending
- Notable bottlenecks/failures: pending
- Resume/restart required: pending

---

## Next optimization target after first completed GPU run

1. Quantify throughput bottlenecks (data loader, checkpoint cadence, eval cadence).
2. Tune effective batch size vs VRAM headroom.
3. Validate benchmark movement against `t_dev_6l_expanded_run` best checkpoint while keeping
   tokenizer and benchmark assumptions constant.
