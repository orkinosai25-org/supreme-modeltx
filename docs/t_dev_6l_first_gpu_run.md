# First GPU-Optimized T-Dev-6L Run Plan

This document defines the candidate path for the first GPU-backed T-Dev-6L run.
It is **not GPU-validated**; readiness changes did not execute GPU experiments.

For an Azure UK self-hosted runner path, use
**[`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md)**.

## Canonical GPU config

- `configs/real_training/t_dev_6l_first_gpu_run.json`

Key settings vs prior CPU benchmark runs:

- model shape: 6 layers, hidden 512, FFN 2048, context 512
- batch size: 8
- gradient accumulation: 4
- schedule: 1,000 steps
- eval cadence: every 100 steps (`eval_max_batches=16`)
- checkpoint cadence: every 100 steps (`keep_last_n=5`)
- precision: BF16 enabled

## Hardware/runtime assumptions

### Minimum envelope

- 1 CUDA GPU with **24 GB+ VRAM**
- BF16-capable CUDA stack (if unavailable, use float16 config override)
- Python environment with:
  - `python -m pip install "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cu124`
  - `python -m pip install -e ".[foundation,dev]"`
  - tokenizer artifact at `artifacts/tokenizers/t-dev-6l/t-dev-6l/v1/tokenizer.model`
  - manifest at `data/manifests/t_dev_6l_gpu_corpus_v1_first_subset.yaml` (or a run-specific replacement)

### Target envelope

- 1 CUDA GPU with 40-80 GB VRAM
- single-process training only; FSDP and multi-GPU are unsupported until validated

### Practical constraints

- Smaller VRAM GPUs may require lowering `batch_size` and increasing `gradient_accumulation_steps`.
- Keep checkpoint path rooted at `.../checkpoints` so run artifacts remain under sibling `run_artifacts/`.

## Preflight validation (required before GPU launch)

Run:

```bash
SMTX_REQUIRE_GPU=1 python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json \
  --preflight
```

Preflight checks:

- config loads
- manifest/tokenizer/source paths resolve
- checkpoint + artifact directories are writable
- device/precision combination is valid
- GPU-designated commands cannot silently fall back to CPU
- FSDP and multi-process launches are rejected
- benchmark follow-on inputs (`configs/benchmark_eval_set.json`, `configs/benchmark_baselines.json`) exist
- artifact contract paths are still aligned (`config_used.json`, `training_summary.*`, `samples.*`)

## Launch commands

### Single GPU

```bash
SMTX_REQUIRE_GPU=1 python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json
```

### Multi-GPU

Unsupported until validated. Do not launch with `torchrun` or enable FSDP.

Before launch, run `python scripts/gpu_diagnostics.py --require-cuda` and archive
the driver, compute capability, CUDA/cuDNN versions and synchronized FP32/FP16/BF16
forward/backward numerical checks. A T4 requires FP16 rather than BF16 and a
smaller memory envelope; this changes settings and is not a hardware-only delta.
Legacy Docker images under `scripts/` are unsupported for this candidate path.

### GitHub Actions (GPU-backed + CPU comparison)

Use:

- `.github/workflows/first-gpu-experiment.yml`

This workflow is for a **self-hosted GPU runner** (`self-hosted`, `linux`, `x64`, `gpu` labels).
It is not runnable on standard GitHub-hosted CPU runners.

The workflow runs preflight, verifies CUDA availability, launches the canonical GPU config, runs benchmark scoring, and writes:

- `artifacts/runs/t_dev_6l_first_gpu_run/comparison_vs_cpu/comparison_vs_cpu.json`
- `artifacts/runs/t_dev_6l_first_gpu_run/comparison_vs_cpu/comparison_vs_cpu.md`

It also publishes the same comparison markdown to the GitHub Actions run summary for quick review.
Post-run findings should be documented in:

- `docs/first-gpu-experiment-findings.md`

The workflow restores an approved immutable tokenizer **after default clean
checkout**; it does not retrain at an existing path. Mount the approved file
read-only outside `GITHUB_WORKSPACE`, then provide workflow inputs
`tokenizer_source_path` (absolute mounted path) and `tokenizer_sha256` (approved
64-character digest). The source and restored copy must match that digest.
Pre-restoring the ignored checkout path before checkout will not survive cleaning.
Archive the SHA-256, not just the path/version. Summaries add
`artifact_identity` with model/config/settings/dataset/tokenizer and benchmark
hashes while retaining existing artifact filenames and summary fields.

CPU-vs-GPU hardware deltas require identical content hashes, code revision,
precision, seed, model shape, data, batch/accumulation, steps, validation, sample
generation and benchmark settings. Historical runs lacking hashes, or runs with
different settings (including the prior CPU configurations), are **noncomparable**.
Their deltas are null in JSON and N/A in Markdown; missing/nonfinite metrics are
never replaced with zero/NaN.

Benchmark reports record the hashes of the eval set and task baselines actually
read at scoring time, scorer code, and each scored sample's content/checkpoint
provenance. Those samples/checkpoints must bind to the training summary hashes;
pretraining canonical-file snapshots are insufficient. Historical benchmark reports
without scoring identity cannot yield hardware deltas.

> Scope note: workflow/documentation updates only define and expose the run path.  
> A completed milestone still requires an actual self-hosted GPU workflow execution and review of the emitted artifacts/summary outputs.

## Resume after interruption

Resume is automatic from the latest checkpoint in:

- `artifacts/runs/t_dev_6l_first_gpu_run/checkpoints/`

Or set an explicit checkpoint path via:

- `training.checkpoint.resume_from`

## Output contract (unchanged)

Outputs are written to:

- `artifacts/runs/t_dev_6l_first_gpu_run/checkpoints/`
- `artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts/`

Run artifacts include:

- `config_used.json`
- `training_summary.json`
- `training_summary.md`
- `samples.json`
- `samples.md`
- `samples/checkpoint_step_*_samples.json`

## Benchmark scoring

Run the canonical benchmark workflow against produced samples:

```bash
python -m supreme_modeltx.model_core.eval.benchmark \
  --eval-set configs/benchmark_eval_set.json \
  --baselines configs/benchmark_baselines.json \
  --samples-root artifacts/runs/t_dev_6l_first_gpu_run/run_artifacts \
  --output-dir artifacts/runs/t_dev_6l_first_gpu_run/benchmark_outputs
```

Expected benchmark outputs:

- `benchmark_results.json`
- `benchmark_results.md`
