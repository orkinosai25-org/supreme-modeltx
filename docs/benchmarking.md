# Baseline benchmark workflow

This repository includes a compact, repeatable benchmark for targeted **code** and **reasoning** checks.

## Canonical eval set

- Tasks: `configs/benchmark_eval_set.json`
- Prompt source for checkpoint sample generation: `configs/canonical_prompts.json`
- Baseline references: `configs/benchmark_baselines.json`

## Workflow

Use the GitHub Actions workflow:

- `.github/workflows/benchmark-baseline.yml`

It runs:

```bash
python -m supreme_modeltx.model_core.eval.benchmark \
  --eval-set configs/benchmark_eval_set.json \
  --baselines configs/benchmark_baselines.json \
  --samples-root run_artifacts \
  --output-dir benchmark_outputs
```

## Outputs

The workflow uploads a `benchmark-outputs` artifact containing:

- `benchmark_results.json` (structured benchmark data for automation/comparison)
- `benchmark_results.md` (human-readable summary)

## Methodology and limitations

- Local checkpoints are evaluated from `run_artifacts/samples/checkpoint_step_*_samples.json`.
- Scoring uses exact answer containment and keyword-ratio checks for deterministic repeatability.
- This is a compact directional benchmark; it does **not** replace full external benchmark suites.
- Open baseline values are reference points sourced from public model documentation.
