# Baseline Benchmark Summary

- Benchmark: `smtx-mini-code-reasoning-v1`
- Generated at (UTC): `2026-06-01T00:18:44.220101+00:00`
- Task count: `4`
- Scope: Lightweight internal/canonical directional benchmark for quick local checkpoint tracking; not equivalent to frontier public benchmark suites.
- Baseline semantics: selected_open_baselines are reference metadata from public model documentation and are not runtime-evaluated in this workflow.

## Best local checkpoint
- Path: `artifacts/runs/t_dev_6l_expanded_run/checkpoints/checkpoint_step_00000150.pt`
- Overall score: `1.0`
- Code score: `1.0`
- Reasoning score: `1.0`
- Matched tasks: `4` / `4`
- Missing tasks: `0`

## Selected open baselines (reference metadata, not runtime-scored here)
- **TinyLlama-1.1B-Chat-v1.0** (`https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0`): overall=0.5, code=0.4, reasoning=0.6
- **Mistral-7B-Instruct-v0.2** (`https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2`): overall=0.8, code=0.75, reasoning=0.85

## What was scored
- Local checkpoint samples found under `artifacts/runs/t_dev_6l_expanded_run/run_artifacts` using patterns `samples/checkpoint_step_*_samples.json, **/samples/checkpoint_step_*_samples.json`.
- Prompt mapping policy: Tasks are scored only when the eval-set prompt exactly matches a sample prompt string.

## How scoring works
- `keyword_ratio`: score = matched required keywords / total required keywords
- `contains`: score = 1.0 if expected answer appears in completion, else 0.0

## What was not scored
- Tasks without an exact prompt match in checkpoint samples are reported as missing.
- Baseline entries are not re-run or re-scored by this workflow.

## Why this is only part of the picture
- This benchmark is a lightweight internal/canonical directional signal, not a replacement for full external benchmark suites.
- Scores depend on deterministic checkpoint sample generation prompts and may not measure broad task generalisation.
- Open baseline scores are reference metadata sourced from public model cards or project documentation, not measured outputs from this workflow.
