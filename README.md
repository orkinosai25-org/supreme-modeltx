# supreme-modeltx

This repository has been restored to a training-oriented **Supreme Model TX** baseline focused on an LLM-style model-training workflow.

## What was restored

- A `src/supreme_modeltx` training codebase with a **T100 6-layer** transformer baseline.
- Training configuration in `configs/t100_6layer.json`.
- A runnable training entrypoint: `python -m src.supreme_modeltx.train`.
- GPU-aware execution logic (`--device auto` prefers CUDA when available).
- Mixed precision support on CUDA via `torch.autocast` + `GradScaler`.
- Checkpoint output to `checkpoints/t100_6layer_last.pt`.

## Quick start

```bash
python -m pip install torch
python -m src.supreme_modeltx.train --config configs/t100_6layer.json --device auto
```

Or run the helper script:

```bash
./scripts/train_t100.sh
```

## GPU-backed execution notes

- `--device auto` selects `cuda` when available.
- Set `CUDA_VISIBLE_DEVICES` to control GPU selection.
- AMP is enabled by default on CUDA (`use_amp: true` in config).

## Tests

```bash
python -m unittest discover -s tests -v
```
