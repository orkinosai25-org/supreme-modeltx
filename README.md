# Supreme Model T-X private foundation

Supreme Model T-X is an early-stage governed pretrained-model evaluation,
adaptation, and deployment foundation supporting **Ebru**, a Turkish-first product
direction. The near-term priority is evaluating approved existing models under
operator control, not training a frontier model from scratch. CPU workflows and
custom T-series models remain useful research and smoke-test paths.

## What is implemented now

- `pyproject.toml` packaging with constrained foundation dependencies
- a lightweight FastAPI service with `/health/` and a non-root Docker runtime
- CPU-safe training, evaluation, and inference configs under `configs/foundation/`
- checkpoint-aware PyTorch training scaffolding in `src/supreme_modeltx/foundation/training.py`
- JSONL dataset validation and split tooling in `src/supreme_modeltx/foundation/dataset_tools.py`
- versioned prompt regression harness in `src/supreme_modeltx/foundation/evaluation.py`
- optional local provider contract in `src/supreme_modeltx/model_core/inference/providers.py`
  (lightweight imports; PyTorch loads only when the pretrained provider is selected)
- GPU diagnostics in `scripts/gpu_diagnostics.py`
- private-run documentation in `/docs`

These are engineering components, not evidence of production readiness or
Turkish-language suitability. CUDA code paths and GPU diagnostics exist, but
single-GPU, BF16, and multi-GPU validation are not established here.

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
returns HTTP 503 until the selected backend has its required local artifacts.

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

The `foundation/` smoke workflow trains **TinyPilotLM** using its own tokenizer,
configuration, and checkpoint format. It is separate from
`model_core/`'s **TSeriesBaseline** and its checkpoint/tokenizer contract.
Neither format is a Hugging Face pretrained snapshot; do not interchange them.
Canonical T-Dev-6L is approximately **58M parameters with 512-token context**.
Smaller CPU proxy experiments do not validate that canonical model on GPU.

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

## Optional local pretrained inference

The default API backend remains `SMTX_INFERENCE_BACKEND=pytorch`, using the
existing `model_core` checkpoint and tokenizer settings. The optional
`LocalTransformersProvider` requires the existing `foundation` extra:

```bash
python -m pip install -e ".[api,foundation]"
```

After approving and preparing the local snapshot described below:

```bash
export SMTX_INFERENCE_BACKEND=transformers
export SMTX_PRETRAINED_MODEL_PATH=/absolute/path/to/approved/local-snapshot
export SMTX_DEVICE=cpu  # or cuda on an approved, compatible GPU host
smtx-serve
```

Supply an **absolute path** to a complete local model/tokenizer snapshot; the
path is not a Hub model ID.
Pretrained inference uses strict `cpu`/`cuda` selection: requesting unavailable
CUDA must fail rather than silently fall back to CPU. This does not change the
legacy research device-selection behavior. The lightweight default API image
does not include this optional runtime or model weights.

Default dtype is **float32 on CPU** and **bfloat16 on CUDA**; CUDA BF16 requires
hardware support and is not validated by CPU tests. Loading uses
`local_files_only=True`, `trust_remote_code=False`, and safetensors weights only.
A tokenizer chat template is required. The service never automatically downloads
models or executes custom upstream model code. Artifact metadata is preserved in
the runtime registry description; registration does not constitute approval.

**Candidate only:** `Qwen/Qwen2.5-0.5B-Instruct` (candidate Apache-2.0 license,
not independently verified here) is a small
evaluation candidate, not an approved or deployed model. License review, an
immutable revision and artifact hashes, safety review, and Turkish evaluation
approval are required before use. No candidate scores or GPU runs are reported.
Upstream model-card verification was blocked by DNS in this task; explicit
upstream license approval remains required.

### Required snapshot manifest

The snapshot must contain `config.json`, `tokenizer_config.json`, all required
tokenizer assets (including a chat template), safetensors weights (and their
index/shards where applicable), and **`smtx-artifact.json`** with:

| Field | Required value |
|---|---|
| `source_model` | Reviewed upstream model identifier |
| `revision` | Immutable upstream commit: exactly 40 lowercase hexadecimal characters |
| `license` | Reviewed license identifier/description for this exact snapshot |
| `architecture` | Hugging Face `config.json` `model_type`, e.g. `qwen2` |
| `conversion_lineage` | List of reviewed conversion steps; `[]` only for an unconverted snapshot |
| `files_sha256` | Relative-path mapping to lowercase SHA-256 for **every** snapshot file except `smtx-artifact.json` |

Acquisition is a separate, operator-approved step. Existing Hugging Face tooling
can obtain a pinned revision, for example
`hf download "$SMTX_SOURCE_MODEL" --revision "$SMTX_APPROVED_REVISION" --local-dir "$SMTX_PRETRAINED_MODEL_PATH"`.
Keep the deployed snapshot immutable after preparation; no file may resolve
outside its directory. Review license/notices and the acquired files before
creating the manifest. Download metadata files inside the directory also count
toward the complete hash inventory.

Use the installed ecosystem tooling for acquisition and standard Python
`json`/`hashlib` for inventory generation, without a new helper script. Set the
reviewed metadata first; these are placeholders, **not approved candidate data**:

```bash
export SMTX_SOURCE_MODEL='<reviewed upstream model identifier>'
export SMTX_APPROVED_REVISION='<reviewed 40-character lowercase commit SHA>'
export SMTX_APPROVED_LICENSE='<reviewed license>'
export SMTX_CONVERSION_LINEAGE='[]'  # only if no conversion was performed

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re

root = Path(os.environ["SMTX_PRETRAINED_MODEL_PATH"])
assert root.is_absolute() and root.is_dir(), "Use an absolute local snapshot path"
root = root.resolve()
revision = os.environ["SMTX_APPROVED_REVISION"]
assert re.fullmatch(r"[0-9a-f]{40}", revision), "Review an immutable commit SHA"
lineage = json.loads(os.environ["SMTX_CONVERSION_LINEAGE"])
assert isinstance(lineage, list) and all(isinstance(step, str) for step in lineage)
config = json.loads((root / "config.json").read_text())
assert (root / "tokenizer_config.json").is_file()
assert any(root.rglob("*.safetensors")), "Safetensors weights are required"
manifest_path = root / "smtx-artifact.json"
assert not manifest_path.exists(), "Do not overwrite an existing approved manifest"
hashes = {}
for path in sorted(root.rglob("*")):
    if not path.is_file() or path == manifest_path:
        continue
    assert path.resolve().is_relative_to(root), "File escapes snapshot boundary"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
manifest = {
    "source_model": os.environ["SMTX_SOURCE_MODEL"],
    "revision": revision,
    "license": os.environ["SMTX_APPROVED_LICENSE"],
    "architecture": config["model_type"],
    "conversion_lineage": lineage,
    "files_sha256": hashes,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
```

Review the generated manifest and upstream provenance before deployment.
Self-generated hashes detect subsequent changes; they do not prove provenance
or approve a license. Loading verifies the complete file inventory/hashes,
architecture, and tokenizer chat template. This preparation example has not
been used to evaluate a pretrained model or execute a GPU run.
Shard indexes may reference only inventoried local safetensors files. Tokenizer
file overrides must name inventoried files and are resolved inside the snapshot;
positional tokenizer overrides and adapters pointing to external base models
are rejected. Use a reviewed, complete merged model snapshot.

### Opt-in Turkish lexical smoke evaluation

After license/revision review and snapshot/manifest preparation, the existing
foundation harness can use the same local provider:

```bash
python -m supreme_modeltx.foundation.evaluation --config configs/pretrained_baseline.json
```

The example config selects `backend=transformers`, `device=cpu`,
`dtype=float32`, and
`pretrained_model_path=/approved/models/qwen2.5-0.5b-instruct`.
Replace that illustrative path in the evaluation config with your approved
absolute snapshot path; the harness uses its config, not the API environment
variables above.

`configs/foundation/eval_cases/turkish-baseline-v1.yaml` contains four **lexical
smoke checks**, not a Turkish capability, safety, or product-readiness benchmark.
They do not satisfy the roadmap's larger frozen held-out evaluation gate.

The report at `artifacts/private_foundation/pretrained_baseline/results.json`
includes artifact provenance, suite hash, per-case token counts and latency,
generation throughput, and peak allocated CUDA VRAM when running on CUDA
(`null` on CPU). Setting the optional config field
`evaluation.compute_cost_per_hour_usd` adds an estimated generation compute cost;
its default is `null`, not a claim that compute is free.
Timing covers serial generation including tokenization/decoding, not model
loading, warmup, or idle time. It is not a serving SLO or total-run cost estimate.
The command is an opt-in procedure only: no model was downloaded or evaluated,
and no pretrained or GPU results were executed in this task.

## NVIDIA alignment: current versus planned

| Component | Status | Next evidence gate |
|---|---|---|
| PyTorch CUDA paths | Existing code; not GPU-validated | Approved single-GPU run with hardware, versions, errors, and metrics recorded |
| cuDNN | CUDA runtime library where supplied by the chosen PyTorch stack; not a separate model backend | Record compatible runtime versions; no benchmark claim |
| NVIDIA NeMo | Planned; not integrated | Licensed Turkish adaptation experiment with held-out comparison |
| NVIDIA Megatron Core | Planned; not integrated | Justified distributed experiment; no multi-GPU validation claimed |
| NVIDIA TensorRT-LLM | Planned; not integrated | Supported export/build and quality/latency/memory parity evaluation |
| ONNX | Planned; not integrated | Supported export and output-parity checks before deployment |

See [the gated roadmap](docs/roadmap.md) and
[GPU readiness plan](docs/gpu-readiness-scaling-plan.md).
Official resources: [NVIDIA Inception](https://www.nvidia.com/en-us/startups/),
[NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html),
[Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/),
[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/),
[CUDA](https://docs.nvidia.com/cuda/),
[cuDNN](https://docs.nvidia.com/deeplearning/cudnn/), and
[ONNX](https://onnx.ai/onnx/).

## Planned single-GPU research pilot

Use `configs/foundation/training-single-gpu.yaml` as a starting configuration for
a separately approved TinyPilotLM research run, not as a validated GPU recipe.
Its scaffolding includes:

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

Compute access is only one gate: licensing, dataset quality, held-out Turkish
evaluation, safety, reproducibility, and deployment controls remain necessary.

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

## Azure App Service: `sumotx2`

`sumotx2` uses **Linux Python code deployment**, not a container registry.
`.github/workflows/main_sumotx2.yml` builds/tests on Python **3.11** and deploys
on pushes to `main` or a manual run on `main`. Pull requests only build/test.
The Dockerfile and local `smtx-serve` command remain unchanged (port 9000).

Before the first deployment:

1. In GitHub repository **Settings → Secrets and variables → Actions**, set
   the repository variable `AZURE_RESOURCE_GROUP` to the resource group containing
   `sumotx2`.
2. Keep the three `AZUREAPPSERVICE_CLIENTID_*`, `AZUREAPPSERVICE_TENANTID_*`,
   and `AZUREAPPSERVICE_SUBSCRIPTIONID_*` secrets generated by Deployment Center
   and referenced in the workflow. The existing deployment identity needs
   **Website Contributor** on `sumotx2` and a federated credential for
   `repo:orkinosai25-org/supreme-modeltx:ref:refs/heads/main`
   (audience `api://AzureADTokenExchange`). No publishing password is needed.
3. In Azure **sumotx2 → Environment variables**, set `SUPREME_MODELTX_API_KEY`
   and `SUPREME_MODELTX_KEY_SALT` to separate, strong random values. The legacy
   `SMTX_API_KEY` / `SMTX_KEY_SALT` names also work; canonical names take precedence.
   The workflow refuses deployment if either value is missing or a development
   default. Never commit these values or upload `.env`; reset any previously
   exposed deployment credentials.

The workflow configures the Web App runtime as `PYTHON|3.11`, enables
`SCM_DO_BUILD_DURING_DEPLOYMENT=true`, sets the health check to `/health/`, and
uses this startup command:

```bash
python -m uvicorn supreme_modeltx.platform_api.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Azure's Oryx build installs the package via the deployment artifact's root
`requirements.txt` (copied from `deployment/azure/requirements.txt`: `.[api]`).
This includes the `src/` package and console entry points, without training
dependencies, local virtual environments, secrets, or model artifacts. The
repository's training-oriented root `requirements.txt` is not used for Azure.
For Linux Python code hosting, use port 8000; `WEBSITES_PORT` is a container
setting and is not needed here.

SQLite state is stored at `/home/smtx/platform.sqlite3`, outside the deployed
code directory, on App Service's persistent `/home` storage. Keep this MVP
at **one instance / one worker**; back up the database and migrate to a managed
database before scaling out.

Merge this PR, then watch **Actions → Build and deploy Python app to Azure
Web App - sumotx2**, or choose **Run workflow → main**. After deployment:

```bash
curl --fail https://<sumotx2-default-hostname>/health/
```

Use the actual default hostname from Azure Overview; the workflow also checks
this endpoint with retries. Expect `{"status":"ok","version":"0.1.0"}`; `/docs`
shows the API. Health is liveness only: inference still returns 503 until
model artifacts and optional inference dependencies are provisioned separately.
If deployment or startup fails, inspect the GitHub Actions job logs, Azure
Deployment Center logs (Oryx build), and App Service Log stream (Uvicorn startup).
This wires up the MVP, not a fully production-hardened inference service.

## Documentation

- `docs/gpu-runbook.md`
- `docs/data-governance.md`
- `docs/experiment-report-template.md`
- `docs/model-card-template.md`

## License

See `LICENSE`.
