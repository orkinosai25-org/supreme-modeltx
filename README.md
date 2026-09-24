# Supreme Model T-X private foundation

Supreme Model T-X is a private PyTorch-based language-model development foundation. This repository supports CPU smoke tests today and is ready for a controlled single-GPU pilot, but it is not a completed, production-trained, or production-deployed LLM.

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

## Install for local CPU development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[foundation,dev]"
```

For a fresh CPU-only environment:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<2.6"
pip install -e ".[foundation,dev]"
```

## Install for a cloud GPU host

Keep the CPU-safe path above for local development. For a CUDA host such as a single-GPU Lambda instance, install a CUDA-enabled PyTorch wheel explicitly instead of replacing the CPU image silently:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export PYTORCH_WHL_INDEX_URL="${PYTORCH_WHL_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
pip install --index-url "${PYTORCH_WHL_INDEX_URL}" "torch>=2.4,<2.6"
pip install -e ".[foundation,dev]"
```

Choose the wheel index that matches the host driver/runtime combination. Validate the host before training:

```bash
python scripts/gpu_diagnostics.py
python scripts/gpu_diagnostics.py --config configs/foundation/training-single-gpu.yaml --require-cuda
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

Run the strict foundation preflight for the single-GPU config:

```bash
python scripts/gpu_diagnostics.py --config configs/foundation/training-single-gpu.yaml --require-cuda
```

## Single-GPU pilot readiness

Use `configs/foundation/training-single-gpu.yaml` as the starting point for the first approved GPU run. It enables:

- deterministic seeding
- explicit `cuda_device_index` selection
- configurable batch size and gradient accumulation
- optional BF16/FP16 mixed precision
- checkpoint save and resume
- validation loss logging
- graceful interruption handling

Before any real run:

1. validate private JSONL data with source and license metadata
2. split train/validation/test data and archive the generated manifest
3. run `python scripts/gpu_diagnostics.py --config configs/foundation/training-single-gpu.yaml --require-cuda`
4. record the experiment with `docs/experiment-report-template.md`
5. start training with `python -m supreme_modeltx.foundation.training --config configs/foundation/training-single-gpu.yaml`

## Docker

Build the CPU-safe development image:

```bash
docker build -t supreme-modeltx-foundation .
```

Default container command:

```bash
python -m supreme_modeltx.foundation.training --config configs/foundation/training-smoke.yaml
```

The checked-in Dockerfiles remain CPU-safe. For the first single-GPU pilot, use the documented host-level CUDA installation path so the PyTorch wheel index and NVIDIA runtime stay explicit and reviewable.

## Documentation

- `docs/gpu-runbook.md`
- `docs/first-gpu-pilot-checklist.md`
- `docs/data-governance.md`
- `docs/experiment-report-template.md`
- `docs/model-card-template.md`

## License

See `LICENSE`.
