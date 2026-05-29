# Model Strategy

> **Supreme ModelTX** — British sovereign AI platform.  
> This document describes the T-series model roadmap, training strategy, and architectural direction.

---

## Vision

Supreme ModelTX trains and operates its own language models on sovereign UK-aligned infrastructure. We do not depend on a third-party hosted model at inference time. The model engine is designed to be:

- **Trainable** — reproducible training pipelines under our full control
- **Auditable** — data provenance and training logs are retained
- **Portable** — checkpoint format compatible with standard PyTorch loaders
- **Scalable** — architecture supports single-GPU development through multi-node distributed training

---

## T-Series Model Family

The T-series is Supreme ModelTX's flagship model family. Each variant is a dense decoder-only transformer.

| Model | Params | Context | Status | Notes |
|---|---|---|---|---|
| **T-Dev-6L** | ~58M | 2 048 | ✅ Available (dev) | Smoke-testable on CPU; scaffold + training dry-run validated |
| **T-101** | ~1B | 4 096 | 🔜 Staged | Architecture defined; training not yet started |
| **T-201** | ~7B | 8 192 | 🗺 Planned | Requires A100/H100 cluster |
| **T-301** | ~30B | 16 384 | 🗺 Planned | Requires multi-node GPU |
| **T-501** | ~70B | 32 768 | 🗺 Research | Long-horizon target |
| **T-X** | MoE | 32 768+ | 🗺 Research | Mixture-of-Experts variant |

All production models target UK sovereign GPU compute (Azure UK South, on-premises H100 racks, or equivalent).

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
Training uses `torch.autocast(dtype=torch.bfloat16)`. BF16 avoids the FP16 overflow instability while matching its training throughput benefits. FP32 GradScaler is retained for compatibility.

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

Distributed training is supported via `torchrun` and `init_distributed()` in `training/distributed/setup.py`.

---

## Data Strategy

1. **Sovereign corpus first** — `data/raw/` contains seed data curated for UK-relevant topics.
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
- **UK-specific benchmarks** in development (UK law, NHS clinical text, government documents)

---

## Scaling Plan

| Phase | Compute | Model | Training tokens |
|---|---|---|---|
| 0 — Dev | CPU / single GPU | T-Dev-6L | N/A (smoke test) |
| 1 — Pre-training T-101 | 8× A100 | T-101 (~1B) | 100B tokens |
| 2 — Instruction tuning | 8× A100 | T-101-Chat | 1B tokens (curated) |
| 3 — Scale to T-201 | 64× H100 | T-201 (~7B) | 1T tokens |
| 4 — Reasoning variants | 64× H100 | T-201-R | Reinforcement + SFT |

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
