# Baseline benchmark workflow

This repository includes a compact, repeatable benchmark for targeted **code** and **reasoning** checks.
It is a **lightweight internal/canonical directional signal** and is **not equivalent** to frontier public benchmark suites.

## Canonical eval set

- Tasks: `configs/benchmark_eval_set.json`
- Prompt source for checkpoint sample generation: `configs/canonical_prompts.json`
- Baseline references: `configs/benchmark_baselines.json`

Baseline references are treated as **reference metadata** from public disclosures (model cards/docs), not runtime-evaluated outputs from this workflow.

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

The markdown summary explicitly separates:

- what was scored
- how scoring works
- what was not scored
- why results are only part of the overall evaluation picture

## Methodology and limitations

- Local checkpoints are discovered from the configured `--samples-root` using:
  - `samples/checkpoint_step_*_samples.json`
  - `**/samples/checkpoint_step_*_samples.json`
- Prompt alignment is exact string matching between eval-set prompt text and sample payload prompt text.
- Scoring uses exact answer containment and keyword-ratio checks for deterministic repeatability.
- This is a compact directional benchmark; it does **not** replace full external benchmark suites.
- Open baseline values are reference metadata sourced from public model documentation, not measured in this workflow.
