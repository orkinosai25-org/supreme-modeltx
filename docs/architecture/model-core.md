# Model Core Architecture

## Overview

`model_core` is the PyTorch-native custom-model research layer for defining,
training, evaluating, and locally serving TSeriesBaseline. It is not the
near-term pretrained-model product strategy or a validated production runtime.

It has **no dependency on** the platform_api layer and can be used standalone.

The broader foundation supports governed pretrained-model evaluation,
adaptation planning, and deployment for Ebru. Its optional Transformers API
backend loads an approved local snapshot, separately from this custom-model
engine. `foundation/training.py` defines **TinyPilotLM**, with its own smoke
tokenizer and checkpoint format; those checkpoints cannot be substituted for
`model_core` checkpoints or pretrained snapshots.

---

## Module breakdown

### `config/`
Pydantic v2 configuration schema (`schema.py`).

- `ModelConfig` — vocabulary, dimensions, attention heads, RoPE, dtype
- `TrainingConfig` — steps, batch size, accumulation, checkpoint, precision, optimiser, scheduler, distributed
- `DataConfig` — manifest path, format, packing, tokenizer
- `TokenizerConfig` — backend, vocab size, special tokens
- `SMTXConfig` — root config combining all sub-configs; loads from JSON or YAML

```python
from supreme_modeltx.model_core.config.schema import SMTXConfig
cfg = SMTXConfig.from_file("config.yaml")
```

### `models/`

#### `common/`
- `attention.py` — `GroupedQueryAttention`, `precompute_freqs_cis`, `apply_rope`
  - Implements RoPE (Su et al., 2023)
  - Implements GQA (Ainslie et al., 2023)
  - Uses `torch.nn.functional.scaled_dot_product_attention` (FlashAttention-compatible)

#### `t_series/`
- `baseline.py` — `TSeriesBaseline` (T-Dev-6L)
  - 6 layers, 512 hidden, 8 heads (default, ~58M params)
  - RMSNorm pre-norm
  - SwiGLU FFN
  - Tied or untied output embeddings
  - Produces `{"logits": ..., "loss": ...}` dict

```python
model = TSeriesBaseline.dev_model()           # T-Dev-6L (~58M params)
model = TSeriesBaseline.from_config(cfg.model) # custom config
```

### `training/`
- `trainer.py` — main training loop; `train(cfg, dry_run=False)` and CLI entrypoint
- `checkpoint.py` — save/load/resume/prune checkpoints
- `optimizer.py` — AdamW with weight-decay param groups; gradient clipping
- `scheduler.py` — cosine / linear / constant LR with warmup
- `precision.py` — `torch.autocast` context and `GradScaler` for BF16/FP16
- `distributed/setup.py` — `init_distributed`, `is_main_process`, `cleanup_distributed`

Training is launched via:
```bash
# Single process (CPU or GPU):
python -m supreme_modeltx.model_core.training.trainer --config config.json

# Dry run (2 steps, no checkpoint):
python -m supreme_modeltx.model_core.training.trainer --dry-run
```

Example research experiment command (not evidence of a GPU run):

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_run.json
```

This canonical run wires:

- manifest-backed ingestion from `data/manifests/t_dev_6l_first_run.yaml`
- SentencePiece tokenizer artifact at `artifacts/tokenizers/t-dev-6l/v1/tokenizer.model`
- checkpoint save/resume at `artifacts/runs/t_dev_6l_first_run/checkpoints/step_*.pt`
- validation loss + perplexity logs via `eval/evaluate_perplexity`

### `data/`
- `manifest.py` — `DataManifest` / `DataSource` Pydantic schema; load from YAML/JSON
- `sources.py` — `iter_source()` adapter for jsonl / text / parquet / hf_dataset backends
- `preprocessing.py` — `tokenize_and_pack()` with greedy sequence packing

### `tokenizer/`
- `workflow.py` — `TokenizerWorkflow` (SentencePiece or HF tokenizers backend); `train_sentencepiece()` and `train_versioned_sentencepiece()`
- `train.py` — CLI entrypoint for local-first, versioned SentencePiece tokenizer training

Tokenizer training artifacts are written under:

```text
artifacts/tokenizers/<model-variant>/<version>/
  tokenizer.model
  tokenizer.vocab
  metadata.json
  training_corpus.txt
```

Inputs can come from:
- local text files/directories (`--input-path`)
- manifest-declared sources (`--manifest-path`) using the existing `data.manifest` and `data.sources` adapters

Training + inference config should reference the produced `.model` via:
- `tokenizer.model_path`
- `data.tokenizer_path` (overrides tokenizer.model_path when set in trainer flow)

### `eval/`
- `perplexity.py` — `evaluate_perplexity()` and `ValidationHook`
  - trainer uses this path for periodic validation metrics (`val_loss`, `perplexity`)

### `inference/`
- `engine.py` — `InferenceEngine`: load checkpoint, autoregressive generate
- `sampling.py` — `sample_tokens()`: greedy / temperature / top-k / top-p

---

## T-Dev-6L: the canonical development model

T-Dev-6L is the first concrete T-series model:

| Property | Value |
|---|---|
| Layers | 6 |
| Hidden size | 512 |
| Attention heads | 8 (MHA) |
| FFN width (SwiGLU) | 2048 |
| Context length | 512 (canonical; extensions require validation) |
| Parameters | ~58M |
| Dtype | BF16 configured for research, FP32 for CPU smoke; GPU precision unvalidated |

It is designed to be:
- **CPU-runnable** for smoke tests and CI
- **CUDA-configurable** for separately approved research experiments; not GPU-validated
- **A research baseline**, not evidence that the legacy ~7B T-101 config works

Distributed initialization scaffolding does not establish complete
distributed-gradient synchronization, FSDP support, or multi-GPU validation.
NeMo, Megatron Core, TensorRT-LLM, and ONNX are all **planned, not integrated**.
cuDNN is a runtime library of a compatible CUDA/PyTorch stack, not a model backend.

---

## Roadmap for model_core

- [x] SentencePiece tokenizer training pipeline on sovereign corpus (local-first, versioned artifacts)
- [x] CPU proxy experiment artifacts for manifest/tokenizer/checkpoint/perplexity plumbing (not canonical GPU validation)
- [ ] Governed pretrained baseline and held-out Turkish evaluation supporting Ebru
- [ ] Approved single-GPU validation with hardware/version/precision and reproducibility evidence
- [ ] Adaptation and optimized serving experiments after baseline approval
- [ ] Distributed research only after need and budget are justified; T-101 is an unvalidated legacy ~7B config, not a delivery commitment

Promotion gates are defined in [the roadmap](../roadmap.md) and
[GPU readiness plan](../gpu-readiness-scaling-plan.md).
