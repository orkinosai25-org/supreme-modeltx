# Supreme Model T-X private foundation

Supreme Model T-X is an early-stage sovereign AI platform foundation. This private repository now includes a reproducible PyTorch development baseline that is ready for CPU smoke tests today and a controlled single-GPU pilot later.

## What is implemented now

- `pyproject.toml` packaging with constrained foundation dependencies
- a lightweight FastAPI service with `/health/` and a non-root Docker runtime
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

## Day 1: run the API locally

Use Python 3.10–3.12 (Docker uses 3.11). The API needs no GPU,
PyTorch, model weights, or cloud credentials.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[api,dev]"
cp .env.example .env
set -a; source .env; set +a
smtx-serve
```

In another terminal:

```bash
curl --fail http://localhost:9000/health/
```

Expected response: `{"status":"ok","version":"0.1.0"}`.
Interactive API documentation is at `http://localhost:9000/docs`.
The health endpoint checks service liveness, not model readiness; inference
returns HTTP 503 until a checkpoint and tokenizer are configured.

`.env` is not loaded automatically. Its sample key and salt are for local
development only; replace both before sharing access. This is not a
production-hardened service.

Run lightweight validation without installing PyTorch:

```bash
python -m compileall src/supreme_modeltx
python -m pytest tests/unit/test_api_startup.py tests/unit/test_platform_api.py
```

## Install for PyTorch development

Training and checkpoint-backed inference are optional extensions of the same
`src/supreme_modeltx` package. API code lives under `platform_api/`, model
components under `model_core/`, and training/evaluation workflows under
`foundation/`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[api,foundation,dev]"
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

Build and run the lightweight API image (Docker Engine required):

```bash
docker build -t supreme-modeltx .
cp .env.example .env  # skip if already configured
docker run --rm --name smtx-api --env-file .env \
  -p 127.0.0.1:9000:9000 supreme-modeltx
```

In another terminal, use the same `curl` health check as above. Docker also
checks `/health/` automatically:

```bash
docker inspect --format '{{.State.Health.Status}}' smtx-api
```

The default image installs only the `api` extra from `pyproject.toml`, excludes
local secrets and model artifacts from its build context, and runs as a
non-root user. SQLite state is temporary unless a writable volume is mounted
and `SUPREME_MODELTX_PLATFORM_DB_PATH` is set to a path inside it.

For the existing CPU training scaffold, explicitly select the optional target:

```bash
docker build --target foundation -t supreme-modeltx-foundation .
docker run --rm supreme-modeltx-foundation
```

Its default command runs the small synthetic-data training smoke config; no
private corpus is bundled. This target is CPU-only, not a CUDA/GPU image.
Additional training dependencies, approved datasets, and GPU provisioning
remain separate from Day 1 API startup.

`pyproject.toml` is the package manifest; `requirements.txt` remains available
for the existing training/deployment scripts. Python CI validates both the
PyTorch tests and the lightweight API, and builds/runs the default Docker image.

## Documentation

- `docs/gpu-runbook.md`
- `docs/data-governance.md`
- `docs/experiment-report-template.md`
- `docs/model-card-template.md`

## License

See `LICENSE`.
