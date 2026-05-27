# Model Core Architecture

The `model_core` package implements the sovereign LLM engine for `supreme-modeltx`.

## T-Series Model Family

The T-Series is a family of dense, decoder-only transformers of increasing scale:

| Model | Hidden | Layers | Heads | FFN | Context |
|-------|--------|--------|-------|-----|---------|
| T101  | 256    | 4      | 4     | 1024 | 512    |
| T201  | 512    | 8      | 8     | 2048 | 1024   |
| T301  | 768    | 12     | 12    | 3072 | 2048   |
| T501  | 1024   | 24     | 16    | 4096 | 4096   |
| T-X   | TBD    | TBD    | TBD   | TBD  | TBD    |

All members share the same architecture class (`TSeriesModel`) and config schema (`ModelConfig`).

## Architecture Choices

### RoPE (Rotary Positional Embeddings)
Position information is encoded as a rotation applied to query and key vectors, enabling better generalisation to lengths beyond the training context.

### SwiGLU FFN
The feed-forward network uses a gated activation: `FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))`. This improves training stability and final model quality compared to a standard GELU FFN.

### RMSNorm
Pre-normalisation with Root Mean Square normalisation (instead of LayerNorm) reduces computation while maintaining training stability.

### Causal Attention Mask
A causal (autoregressive) upper-triangular mask prevents each position from attending to future positions.

## Config Schema

All hyperparameters are validated at construction time using Pydantic v2:

```python
from supreme_modeltx.model_core.config import ModelConfig, T_SERIES_CONFIGS

# Use a preset
cfg = T_SERIES_CONFIGS["t301"]

# Or define a custom config
cfg = ModelConfig(
    model_id="my-custom-model",
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=2048,
    vocab_size=32000,
    max_position_embeddings=2048,
)
```

Validation rules:
- `hidden_size` must be divisible by `num_attention_heads`
- `hidden_size`, `vocab_size`, and other dimensions have minimum values
- `hidden_act` is restricted to `"gelu"`, `"relu"`, or `"silu"`

## Training Pipeline

```python
from supreme_modeltx.model_core.config import TrainingConfig
from supreme_modeltx.model_core.models import TSeriesModel
from supreme_modeltx.model_core.training import Trainer

model = TSeriesModel(cfg)
train_cfg = TrainingConfig(
    run_name="t101-pretrain-v1",
    model_id="t101",
    output_dir="/checkpoints/t101",
    dataset_path="/data/corpus",
    max_steps=100_000,
)
trainer = Trainer(model, train_cfg, train_dataloader)
trainer.train()
```

For GPU-scale training, wrap `model` and `train_dataloader` with Accelerate before passing to `Trainer`.

## Evaluation

```python
from supreme_modeltx.model_core.eval import EvalHarness

harness = EvalHarness(model, device="cuda")
ppl = harness.compute_perplexity(eval_dataloader)
```

## Inference

```python
from supreme_modeltx.model_core.inference import InferenceEngine

engine = InferenceEngine(model, device="cuda")
generated_ids = engine.generate(input_ids, max_new_tokens=128, top_p=0.95)
```

## Roadmap

- [ ] Flash Attention 2 integration
- [ ] Grouped-Query Attention (GQA)
- [ ] MoE (Mixture of Experts) variant
- [ ] Instruction tuning and RLHF support
- [ ] UK domain fine-tuning (law, finance, government)
- [ ] Speculative decoding for inference throughput
