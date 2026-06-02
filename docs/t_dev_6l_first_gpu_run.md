# First GPU-Optimized T-Dev-6L Run Plan

This document is the execution-ready package for the first canonical GPU-backed T-Dev-6L run.

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
  - `pip install -e ".[dev]"`
  - tokenizer artifact at `artifacts/tokenizers/t-dev-6l/t-dev-6l/v1/tokenizer.model`
  - manifest at `data/manifests/t_dev_6l_gpu_corpus_v1_first_subset.yaml` (or a run-specific replacement)

### Target envelope

- 1-4 CUDA GPUs with 40-80 GB VRAM each
- same config and artifact contract; scale with `torchrun` when needed

### Practical constraints

- Smaller VRAM GPUs may require lowering `batch_size` and increasing `gradient_accumulation_steps`.
- Keep checkpoint path rooted at `.../checkpoints` so run artifacts remain under sibling `run_artifacts/`.

## Preflight validation (required before GPU launch)

Run:

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json \
  --preflight
```

Preflight checks:

- config loads
- manifest/tokenizer/source paths resolve
- checkpoint + artifact directories are writable
- device/precision combination is valid
- benchmark follow-on inputs (`configs/benchmark_eval_set.json`, `configs/benchmark_baselines.json`) exist
- artifact contract paths are still aligned (`config_used.json`, `training_summary.*`, `samples.*`)

## Launch commands

### Single GPU

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json
```

### Multi-GPU

```bash
torchrun --nproc_per_node=4 -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json
```

### GitHub Actions (GPU-backed + CPU comparison)

Use:

- `.github/workflows/first-gpu-experiment.yml`

This workflow is for a **self-hosted GPU runner** (`self-hosted`, `linux`, `x64`, `gpu` labels).
It is not runnable on standard GitHub-hosted CPU runners.

The workflow runs preflight, verifies CUDA availability, launches the canonical GPU config, runs benchmark scoring, and writes:

- `artifacts/runs/t_dev_6l_first_gpu_run/comparison_vs_cpu/comparison_vs_cpu.json`
- `artifacts/runs/t_dev_6l_first_gpu_run/comparison_vs_cpu/comparison_vs_cpu.md`

It also publishes the same comparison markdown to the GitHub Actions run summary for quick review.

CPU-vs-GPU deltas are computed only when tokenizer model paths match across runs.

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
