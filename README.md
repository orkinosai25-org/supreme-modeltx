# Supreme ModelTX

**Supreme ModelTX** is a British sovereign LLM platform scaffold — a PyTorch-first model development stack paired with an API-first business access layer.

> **Sovereign AI, designed and built in Britain.**  
> Our platform gives UK organisations a path to owning their AI infrastructure end-to-end: from pretraining data and tokenisation through to governed deployment and auditable usage metering.

---

## UK Strategic Relevance

Supreme ModelTX directly addresses the UK government's stated priorities for AI sovereignty:

- **Domestic AI capability** — British-built, open-architecture LLM stack that is not dependent on any single US or non-UK cloud provider.
- **Data residency and control** — Designed to run inside customer-owned Azure subscriptions in UK South / UK West regions; data never leaves the customer boundary.
- **Governed deployment** — Control-plane architecture enforces policy, approval gates, and full audit trails before any model is deployed or inference is served — meeting public-sector accountability requirements.
- **Reproducible, auditable science** — Every training run is config-driven and emits structured artifacts; experiments are reproducible from a clean checkout.
- **Open foundation for public-sector use** — Platform API is OpenAI-compatible, enabling rapid integration into existing public-sector tooling, portals, and procurement patterns.

**Target users:** UK government departments, NHS, HMRC, MoD, defence contractors, critical national infrastructure operators, and regulated financial institutions requiring AI capability without vendor lock-in.

See [`docs/sovereign-ai/application-brief.md`](docs/sovereign-ai/application-brief.md) for the full sovereign AI application brief.

---

## Two first-class layers

```
supreme-modeltx/
├── src/supreme_modeltx/
│   ├── model_core/       ← PyTorch-native LLM development
│   └── platform_api/     ← API-first business access
```

### 1. `model_core` — Model Development Stack

| Module | Purpose |
|---|---|
| `config/` | Pydantic schema for model, training, data, and tokenizer settings |
| `models/t_series/` | T-series decoder-only transformers (T-Dev-6L baseline today; T-101 roadmap) |
| `models/common/` | RoPE, GQA attention, RMSNorm, SwiGLU building blocks |
| `training/` | Training loop, checkpoint, optimizer, scheduler, mixed precision, distributed |
| `data/` | Manifest-driven JSONL/Parquet/HF Datasets ingestion with sequence packing |
| `tokenizer/` | SentencePiece-oriented tokenizer workflow boundary |
| `eval/` | Perplexity, validation hooks |
| `inference/` | Checkpoint loading, autoregressive generation, nucleus/top-k sampling |

### 2. `platform_api` — Business API Platform

| Module | Purpose |
|---|---|
| `api/` | FastAPI application with OpenAPI docs (`/docs`) |
| `auth/` | API key issuance and verification |
| `tenants/` | Project / tenant management |
| `usage/` | Token usage metering and rate-limit scaffolding |
| `model_registry/` | Model catalogue with stage tracking |
| `deployment/` | Deployment lifecycle management |

---

## Quick start

### One-step setup

```bash
# Clone and set up everything (train + eval + dev extras):
git clone https://github.com/orkinosai25-org/supreme-modeltx.git
cd supreme-modeltx
bash scripts/setup.sh

# With API platform dependencies:
bash scripts/setup.sh --api
```

`scripts/setup.sh` installs the package in editable mode, verifies syntax, and copies `.env.example → .env` if not already present.

### Install manually

```bash
# Core (model_core only):
pip install -e ".[train]"

# API platform:
pip install -e ".[api]"

# Development (everything + tests):
pip install -e ".[train,api,eval,dev]"
```

### Smoke test — model instantiation and tiny forward pass

```bash
python -m pytest tests/smoke/ -v
```

### Smoke test — 2-step CPU training dry run

```bash
python -m supreme_modeltx.model_core.training.trainer --dry-run
```

### Train a versioned tokenizer (T-Dev-6L)

```bash
# local-first corpus training (defaults to INPUT_PATH=data/raw)
bash scripts/train_tokenizer.sh

# explicit run via module/entrypoint
python -m supreme_modeltx.model_core.tokenizer.train \
  --input-path data/raw \
  --artifact-root artifacts/tokenizers \
  --model-variant t-dev-6l \
  --version v1 \
  --vocab-size 32000
```

Tokenizer artifacts are versioned at:

```text
artifacts/tokenizers/<model-variant>/<version>/
  tokenizer.model
  tokenizer.vocab
  metadata.json
  training_corpus.txt
```

Point training/inference config at the produced model path:

```json
{
  "tokenizer": {
    "backend": "sentencepiece",
    "model_path": "artifacts/tokenizers/t-dev-6l/v1/tokenizer.model"
  },
  "data": {
    "tokenizer_path": "artifacts/tokenizers/t-dev-6l/v1/tokenizer.model"
  }
}
```

### First real T-Dev-6L training experiment (manifest + checkpoint + perplexity)

Canonical config:

- `configs/real_training/t_dev_6l_first_run.json`

This run uses:

- manifest: `data/manifests/t_dev_6l_first_run.yaml`
- tokenizer artifact: `artifacts/tokenizers/t-dev-6l/v1/tokenizer.model`
- checkpoint directory: `artifacts/runs/t_dev_6l_first_run/checkpoints/`

Run:

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_run.json
```

Expected outputs:

- checkpoint files: `artifacts/runs/t_dev_6l_first_run/checkpoints/step_*.pt`
- training loss logs
- validation logs with loss and perplexity (from `model_core/eval/perplexity.py`)
- resume logs when checkpoint files already exist (`Resumed from step ...`)
- structured run artifacts:
  - `artifacts/runs/t_dev_6l_first_run/run_artifacts/config_used.json`
  - `artifacts/runs/t_dev_6l_first_run/run_artifacts/training_summary.json`
  - `artifacts/runs/t_dev_6l_first_run/run_artifacts/training_summary.md`
  - `artifacts/runs/t_dev_6l_first_run/run_artifacts/samples.json`
  - `artifacts/runs/t_dev_6l_first_run/run_artifacts/samples.md`
  - `artifacts/runs/t_dev_6l_first_run/run_artifacts/samples/checkpoint_step_*_samples.json`

To compare runs, inspect each run's `run_artifacts/training_summary.json` for:

- tokenizer path/version
- checkpoint paths and best checkpoint
- latest validation loss and perplexity
- device, precision, eval cadence
- timestamps and git commit

See **[`docs/run-artifacts.md`](docs/run-artifacts.md)** for the full artifact schema, how to inspect outputs, and how to compare runs.

Example log lines:

```text
step=10/20 | loss=7.9315 | lr=3.00e-04
eval step=10/20 | val_loss=7.8122 | perplexity=2471.83
```

Limitations of this first real run:

- single-process local execution focus (not multi-node scale)
- tiny processed dataset slices for wiring validation, not benchmark quality

### Expanded benchmarked T-Dev-6L experiment (larger corpus + longer schedule)

Canonical config:

- `configs/real_training/t_dev_6l_expanded_run.json`

Run:

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_expanded_run.json
```

This run keeps the same run-artifact and benchmark contract while increasing corpus size and schedule
length. Outputs are written under:

- `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/`
- `artifacts/runs/t_dev_6l_expanded_run/benchmark_outputs/`

Comparison against `t_dev_6l_first_run` (loss, perplexity, sample quality, and benchmark deltas) is
documented in **[`docs/expanded-experiment-findings.md`](docs/expanded-experiment-findings.md)**.

The funded-compute follow-on plan that turns those benchmark results into a GPU-readiness and
application-scaling roadmap is documented in
**[`docs/gpu-readiness-scaling-plan.md`](docs/gpu-readiness-scaling-plan.md)**.

The next versioned corpus target for future GPU-backed experiments is documented in
**[`docs/t_dev_6l_gpu_corpus_plan.md`](docs/t_dev_6l_gpu_corpus_plan.md)** with its
reproducible manifest at **`data/manifests/t_dev_6l_gpu_corpus_v1.yaml`**.
The first approved materialized subset and training-ready manifest are:

- `data/manifests/t_dev_6l_gpu_corpus_v1_first_subset.yaml`
- `data/processed/t_dev_6l_gpu_corpus_v1/{train,validation}/`

### First GPU-optimized T-Dev-6L training run

Canonical GPU config:

- `configs/real_training/t_dev_6l_first_gpu_run.json`

Run preflight before consuming GPU time:

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json \
  --preflight
```

Launch training:

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json
```

Full hardware assumptions, resume flow, and benchmark scoring procedure:
**[`docs/t_dev_6l_first_gpu_run.md`](docs/t_dev_6l_first_gpu_run.md)**.
Run-result review tracker:
**[`docs/first-gpu-experiment-findings.md`](docs/first-gpu-experiment-findings.md)**.

To execute the first GPU-backed benchmarked run and automatically compare against
`t_dev_6l_first_run` and `t_dev_6l_expanded_run`, trigger:
**`.github/workflows/first-gpu-experiment.yml`**.
This workflow requires a self-hosted GPU runner (`self-hosted`, `linux`, `x64`, `gpu` labels).
For Azure UK provisioning/bootstrap and runner setup, use
**[`docs/azure-uk-gpu-runner-runbook.md`](docs/azure-uk-gpu-runner-runbook.md)**.

### Run the platform API

```bash
SUPREME_MODELTX_API_KEY=my-key uvicorn supreme_modeltx.platform_api.api.app:create_app --factory --reload
# Docs at: http://localhost:9000/docs
```

### Run all tests

```bash
python -m pytest tests/ -v
```

---

## Demo

A complete end-to-end demo can be run from a clean checkout after setup:

```bash
# Run model smoke test + CPU training dry run + API health check:
bash scripts/run_demo.sh

# API-only (skip if fastapi is not installed):
bash scripts/run_demo.sh --skip-api
```

Expected output:

```text
[1/2] Model instantiation, forward-pass, and training dry-run (smoke tests) ... PASSED
[2/2] Platform API health check ... PASSED (or SKIPPED if not installed)
Demo complete ✓
```

---

## Evaluation

```bash
# Run full test suite + syntax check + (optional) perplexity + benchmark:
bash scripts/evaluate.sh

# With a checkpoint:
export SMTX_CHECKPOINT_PATH=artifacts/runs/t_dev_6l_first_run/checkpoints/step_100.pt
export SMTX_TOKENIZER_PATH=artifacts/tokenizers/t-dev-6l/v1/tokenizer.model
bash scripts/evaluate.sh
```

Results are written to `results/latest.json`. See [`docs/evaluation.md`](docs/evaluation.md) for the full evaluation framework.

---

## Architecture

See [`docs/architecture/overview.md`](docs/architecture/overview.md) for a full description of the two-layer architecture.

See [`docs/sovereignty/principles.md`](docs/sovereignty/principles.md) for the sovereignty design principles.

---

## T-series model family

| Model | Status | Params | Context |
|---|---|---|---|
| **T-Dev-6L** | ✅ Scaffold complete, CPU smoke-testable | ~58M | 512 |
| **T-101** | 🔜 Architecture designed, training pending GPU allocation | 7B | 4096 |
| T-201 (reasoning) | 🗺 Roadmap | — | — |
| T-301 (retrieval) | 🗺 Roadmap | — | — |

---

## Repository structure

```
supreme-modeltx/
├── src/supreme_modeltx/          ← main Python package
│   ├── model_core/               ← LLM development layer
│   └── platform_api/             ← business API layer
├── tests/
│   ├── unit/                     ← config & schema unit tests
│   └── smoke/                    ← model instantiation & forward-pass smoke tests
├── docs/
│   ├── architecture/             ← architecture docs
│   ├── sovereignty/              ← sovereignty principles
│   ├── website/                  ← website-ready positioning copy
│   ├── sovereign-ai/             ← sovereign AI application brief
│   ├── pitch/                    ← pitch deck outline
│   └── positioning/              ← internal messaging guide
├── data/raw/                     ← sample pretraining data
├── control-plane/                ← C# ASP.NET Core governance control plane (retained)
├── infra/                        ← Bicep infrastructure definitions
├── pyproject.toml                ← package metadata and dependencies
├── THIRD_PARTY_NOTICES.md        ← open-source provenance
└── .github/workflows/            ← CI workflows
```

---

## Development

This repository is at **scaffold stage**: the architecture and module boundaries are established, core primitives are implemented and tested, and the design is ready for GPU-backed training experiments.

What is **not yet** in this repository:
- Trained model weights (pending GPU allocation)
- Production-scale tokenizer pipeline beyond the local-first versioned workflow
- Production-ready distributed training at scale (FSDP/DeepSpeed wiring is started)
- Production database backends for the platform API

See [`docs/architecture/overview.md`](docs/architecture/overview.md) for the full roadmap.

---

## Sovereignty

This platform is designed with sovereignty as a first principle — not just branding. See [`docs/sovereignty/principles.md`](docs/sovereignty/principles.md).

---

## Positioning and pitch materials

- Website overview: [`docs/website/overview.md`](docs/website/overview.md)
- Sovereign AI application brief: [`docs/sovereign-ai/application-brief.md`](docs/sovereign-ai/application-brief.md)
- Pitch deck outline: [`docs/pitch/pitch-deck-outline.md`](docs/pitch/pitch-deck-outline.md)
- Internal messaging guide: [`docs/positioning/messaging.md`](docs/positioning/messaging.md)
- Run artifact reference: [`docs/run-artifacts.md`](docs/run-artifacts.md)
- Baseline benchmark workflow: [`docs/benchmarking.md`](docs/benchmarking.md)
- First GPU T-Dev-6L run plan: [`docs/t_dev_6l_first_gpu_run.md`](docs/t_dev_6l_first_gpu_run.md)
- First GPU T-Dev-6L findings: [`docs/first-gpu-experiment-findings.md`](docs/first-gpu-experiment-findings.md)

## Readiness & Evidence

Evidence-backed readiness documents for fund reviewers and public-sector stakeholders:

- POC status matrix (PR #59): [`docs/poc-status.md`](docs/poc-status.md)
- TRL self-assessment (TRL 4–5 rubric + current score): [`docs/trl-assessment.md`](docs/trl-assessment.md)
- Gap analysis (POC → pilot/deployment): [`docs/gap-analysis.md`](docs/gap-analysis.md)
- Funding readiness (90-day plan, GPU-constrained strategy, milestones): [`docs/funding-readiness.md`](docs/funding-readiness.md)

---

## UK Sovereign AI Fund — documentation pack

- Architecture overview: [`docs/architecture.md`](docs/architecture.md)
- Evaluation framework: [`docs/evaluation.md`](docs/evaluation.md)
- 90-day delivery plan: [`docs/delivery-plan-90d.md`](docs/delivery-plan-90d.md)
- Risk register: [`docs/risk-register.md`](docs/risk-register.md)
- TRL self-assessment: [`docs/trl-assessment.md`](docs/trl-assessment.md)
- Gap analysis (POC → pilot): [`docs/gap-analysis.md`](docs/gap-analysis.md)
- Funding readiness: [`docs/funding-readiness.md`](docs/funding-readiness.md)

---

## Provenance

This repository builds on ideas and patterns from well-known open-source AI research projects. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full attribution.

The codebase is original. We are not a fork of DeepSeek, LLaMA, or any other project; we draw inspiration from open research in the same way that all serious LLM implementations do.

---

## Licence

See [LICENSE](LICENSE).
