# Supreme Model T-X private foundation

Supreme ModelTX is an early-stage sovereign AI platform foundation. This repository now includes a private PyTorch development baseline for Supreme Model T-X that is ready for CPU smoke tests today and a controlled single-GPU pilot later.

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

1. validate private JSONL data with source and licence metadata
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
