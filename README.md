# Supreme Model T-X private foundation

Supreme ModelTX is an early-stage sovereign AI platform foundation. This private repository now includes a reproducible PyTorch development baseline for Supreme Model T-X that is ready for CPU smoke tests today and a controlled single-GPU pilot later.

## What is implemented now

- `pyproject.toml` packaging with constrained foundation dependencies
- CPU-safe training, evaluation, and inference configs under `configs/foundation/`
- checkpoint-aware PyTorch training scaffolding in `src/supreme_modeltx/foundation/training.py`
- JSONL dataset validation and split tooling in `src/supreme_modeltx/foundation/dataset_tools.py`
- versioned prompt regression harness in `src/supreme_modeltx/foundation/evaluation.py`
- GPU diagnostics in `scripts/gpu_diagnostics.py`
- private-run documentation in `/docs`

## What is not included

- proprietary datasets
- model weights or checkpoints
- credentials, API keys, or secrets
- any claim that Supreme Model T-X has already completed training

## Foundation layout

```text
src/supreme_modeltx/foundation/
  config.py            # training / inference / evaluation schemas
  dataset_tools.py     # JSONL validation, duplicate detection, splitting, optional scans
  evaluation.py        # versioned prompt harness with JSON results
  gpu_diagnostics.py   # CUDA and mixed-precision diagnostics
  inference.py         # stub or checkpoint-backed inference entrypoint
  tokenization.py      # deterministic smoke-test tokenizer
  training.py          # reproducible tiny-model training scaffold
```

## Install for local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[foundation,dev]"
```

For CPU-only PyTorch wheels in a fresh environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## CPU smoke test

Run the targeted smoke test:

```bash
python -m pytest tests/smoke/test_foundation_cpu_smoke.py -v
```

Run the training scaffold directly:

```bash
python -m supreme_modeltx.foundation.training --config configs/foundation/training-smoke.yaml
```

Run the evaluation harness:

```bash
python -m supreme_modeltx.foundation.evaluation --config configs/foundation/evaluation.yaml
```

Run dataset validation:

```bash
python -m supreme_modeltx.foundation.dataset_tools validate path/to/dataset.jsonl --scan-pii --scan-secrets
```

Run GPU diagnostics:

```bash
python scripts/gpu_diagnostics.py
```

## Single-GPU pilot readiness

Use `configs/foundation/training-single-gpu.yaml` as the starting point for the first approved GPU run. It enables:

- deterministic seeding
- configurable batch size and gradient accumulation
- optional BF16/FP16 mixed precision
- checkpoint save and resume
- validation loss logging
- graceful interruption handling

Before any real run:

1. validate private JSONL data with source and license metadata
2. split train/validation/test data and archive the generated manifest
3. run `scripts/gpu_diagnostics.py`
4. record the experiment with `docs/experiment-report-template.md`

## Docker

Build the CPU-safe development image:

```bash
docker build -t supreme-modeltx-foundation .
```

Default container command:

```bash
python -m supreme_modeltx.foundation.training --config configs/foundation/training-smoke.yaml
```

## Documentation

- `docs/gpu-runbook.md`
- `docs/data-governance.md`
- `docs/experiment-report-template.md`
- `docs/model-card-template.md`

## License

See `LICENSE`.
