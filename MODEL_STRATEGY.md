# Model Strategy

> **Supreme ModelTX** — early-stage governed pretrained-model foundation supporting Ebru.
> This document separates the near-term evaluation/adaptation strategy from custom-model research.

---

## Vision

Supreme ModelTX aims to evaluate, adapt, and deploy approved pretrained models on
operator-controlled infrastructure for Ebru's Turkish-first workflows. Local
artifact custody does not require original pretraining or imply independence
from upstream licenses. Custom-model training remains a research path, not the
near-term product dependency. The foundation is designed to be:

- **Trainable** — reproducible training pipelines under our full control
- **Auditable** — data provenance and training logs are retained
- **Portable** — checkpoint format compatible with standard PyTorch loaders
- **Evidence-gated** — GPU and distributed scale require separate validation, not just configuration

## Near-term pretrained strategy

1. Review candidate licenses, intended use, safety, and provenance before acquisition.
2. Pin an immutable upstream revision and record hashes for a complete local snapshot.
3. Evaluate Turkish quality, instruction following, grounding, and safety on held-out prompts.
4. Consider adaptation only after a baseline and approved dataset exist.
5. Promote to a governed Ebru pilot only after quality and deployment gates pass.

`Qwen/Qwen2.5-0.5B-Instruct` is an **evaluation candidate** with a candidate
Apache-2.0 license, not independently verified here; explicit upstream
license/revision review and Turkish evaluation approval
remain pending. No candidate benchmark results or GPU executions are claimed.
The optional local Transformers API backend is selected with
`SMTX_INFERENCE_BACKEND=transformers`, `SMTX_PRETRAINED_MODEL_PATH` pointing to a
local snapshot, and strict `SMTX_DEVICE=cpu|cuda`. The existing `pytorch` backend
remains the default. This is inference scaffolding, not a pretrained adaptation
pipeline or automatic download service.

The opt-in `configs/pretrained_baseline.json` routes the existing foundation
evaluation harness to this provider on CPU/float32. Its four-case Turkish lexical
smoke suite is wiring coverage, not held-out Turkish product validation. The
report can capture provenance, tokens, generation latency/throughput, CUDA VRAM,
and an optional generation-only compute-cost estimate; no model has been
downloaded or evaluated in this task. See [README](README.md) for preparation
and the command.

---

## T-Series Model Family

The T-series is a custom-model research track. Configurations are not trained
weights, capability results, or delivery commitments.

| Model | Params | Context | Status | Notes |
|---|---|---|---|---|
| **T-Dev-6L** | ~58M | 512 | Research/smoke | Canonical `configs/t_dev_6l.json`; smaller CPU proxies are separate |
| **T-101** | ~7B | Legacy config only | Unvalidated legacy configuration | No validated training or serving result |
| **T-201 / T-301 / T-501** | Uncommitted larger targets | Uncommitted | Research concepts | Not funded programmes or promised checkpoints |
| **T-X** | Uncommitted | Uncommitted | Research concept | No committed MoE programme |

Deployment location and hardware require operator approval; UK-aligned compute
is a design objective, not evidence of an allocated GPU cluster.

---

## Architecture Decisions

### Decoder-Only Transformer
The T-series uses a standard causal decoder-only transformer. This aligns with the GPT/LLaMA/Mistral generation of open models and allows straightforward autoregressive inference.

### Pre-Norm with RMSNorm
Pre-normalization (`x → norm(x) → sublayer + x`) is more stable than post-norm during training. RMSNorm is used instead of LayerNorm: it is computationally cheaper and performs equivalently in practice.

### RoPE Positional Embeddings
Rotary Position Embeddings (RoPE) are applied directly to query and key projections inside attention. This gives relative-position awareness without a fixed learned embedding table, and generalises better to longer sequences at inference time.

### Grouped Query Attention (GQA)
From T-101 upwards, GQA reduces the KV cache memory footprint relative to multi-head attention, enabling longer context windows and higher batch sizes without proportional memory growth.

### SwiGLU Feed-Forward Network
The FFN uses `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)` gating. This consistently outperforms ReLU-FFN on language modelling benchmarks and is standard in modern sovereign LLM stacks.

### BF16 Mixed Precision
The research stack includes BF16/FP16 autocast configuration. Precision support,
stability, and performance must be measured on the selected CUDA host; CPU smoke
evidence does not validate GPU mixed precision.

---

## Training Pipeline

```
data/raw/  →  manifest.yaml  →  DataSource adapters  →  sequence packing
                                                       ↓
tokenizer/workflow.py  →  SentencePiece / HF tokenizer
                                                       ↓
training/trainer.py  →  gradient accumulation + BF16 autocast + grad clip
                     →  checkpoint save/load (training/checkpoint.py)
                     →  AdamW with weight decay groups (training/optimizer.py)
                     →  cosine LR schedule with warmup (training/scheduler.py)
                                                       ↓
eval/perplexity.py  →  ValidationHook → perplexity on held-out set
```

Distributed initialization scaffolding exists in
`training/distributed/setup.py`. It is not proof of complete distributed
training, FSDP wrapping, or multi-GPU validation.

`foundation/` trains TinyPilotLM with its own checkpoint format. The diagram
above describes `model_core`'s TSeriesBaseline path; the two formats and a
Transformers pretrained snapshot are not interchangeable.

---

## Data Strategy

1. **Governed corpus first** — `data/raw/` contains research seed data, not an approved product corpus; Turkish adaptation requires separate provenance and license review.
2. **Manifest-driven** — `DataManifest` (YAML/JSON) describes data sources, types, and weights. Sources can be JSONL, plaintext, Parquet, or HuggingFace Datasets.
3. **Sequence packing** — greedy packing (PaLM-style) maximises GPU utilisation by filling context windows.
4. **Future**: deduplicated web crawl filtered for quality; UK public sector documents; code; reasoning chains.

---

## Tokenizer Strategy

- SentencePiece BPE tokenizer, vocabulary size 32 000 (dev) → 65 536 (T-101+)
- Trained on the sovereign corpus; extended with domain-specific tokens as needed
- HuggingFace tokenizers supported as an optional backend for ecosystem compatibility
- Tokenizer artefacts versioned alongside model checkpoints

---

## Evaluation Harness

- **Perplexity** on a held-out validation set — primary scalar loss proxy during training
- **lm-eval-harness** integration planned for zero-shot task benchmarks (ARC, HellaSwag, MMLU)
- **Turkish-first evaluation** planned for Ebru; task and safety thresholds must be agreed before runs

---

## NVIDIA / ONNX alignment status

| Component | Current status | Intended investigation |
|---|---|---|
| CUDA / PyTorch | Code paths exist; not GPU-validated | Single-GPU baseline with reproducible measurements |
| cuDNN | Runtime library where provided by the compatible PyTorch CUDA stack | Record versions; not an inference backend |
| NeMo | Planned; not integrated | Governed pretrained adaptation |
| Megatron Core | Planned; not integrated | Distributed research only when justified |
| TensorRT-LLM | Planned; not integrated | Optimized serving after supported-model and parity checks |
| ONNX | Planned; not integrated | Export portability and runtime parity checks |

The [roadmap](docs/roadmap.md) defines measurable promotion gates, not fixed
cluster sizes or training-token commitments. See the
[GPU readiness plan](docs/gpu-readiness-scaling-plan.md) for official resources.

---

## What Was Reused from Existing SMTX vs. Newly Introduced

| Component | Decision | Notes |
|---|---|---|
| `control-plane/` (C# ASP.NET Core) | **Retained as-is** | Governance control plane; no changes |
| `infra/` (Bicep modules) | **Retained as-is** | Infrastructure-as-code for Azure; preserved |
| `training/train_t101.py` | **Retained** | Older training script; new canonical path is `src/supreme_modeltx/model_core/training/trainer.py` |
| `api/` (legacy FastAPI app) | **Retained** | Original API layer; new canonical path is `src/supreme_modeltx/platform_api/` |
| `data/raw/` | **Retained** | Sovereign corpus seed data |
| `tmodels/` configs | **Retained** | Model configuration stubs for T-101 → T-X |
| `src/supreme_modeltx/` | **All new** | Original code — model core, platform API, utils, tests |
| `configs/t_dev_6l.json` | **New** | Canonical T-Dev-6L config for the new model core |
| `.github/workflows/python-ci.yml` | **New** | CI for the Python package (unit + smoke tests) |
| `docs/architecture/`, `docs/sovereignty/` | **New** | Architecture and sovereignty documentation |
| `THIRD_PARTY_NOTICES.md` | **New** | Provenance tracking for all inspirations |

---

## Provenance and Licensing

All code in `src/supreme_modeltx/` is original. Architectural inspiration is drawn from published research (RoPE, GQA, SwiGLU, RMSNorm) and open-source projects (LLaMA 2, Mistral, LitGPT, PaLM sequence packing). No code has been copied verbatim. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full provenance.
