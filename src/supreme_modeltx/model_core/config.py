"""Config schema for supreme-modeltx model core."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Configuration for a T-Series model."""

    model_id: str = Field(description="Unique model identifier, e.g. 't101', 't201'.")
    hidden_size: int = Field(default=768, ge=64, description="Transformer hidden dimension.")
    num_hidden_layers: int = Field(default=12, ge=1, description="Number of transformer layers.")
    num_attention_heads: int = Field(default=12, ge=1, description="Number of attention heads.")
    intermediate_size: int = Field(default=3072, ge=64, description="FFN intermediate dimension.")
    vocab_size: int = Field(default=32000, ge=256, description="Vocabulary size.")
    max_position_embeddings: int = Field(default=2048, ge=64, description="Maximum sequence length.")
    hidden_act: Literal["gelu", "relu", "silu"] = Field(
        default="silu", description="Activation function used in the FFN."
    )
    rms_norm_eps: float = Field(default=1e-5, gt=0, description="Epsilon for RMS normalisation.")
    initializer_range: float = Field(default=0.02, gt=0, description="Std for weight initialisation.")
    tie_word_embeddings: bool = Field(default=False, description="Tie input/output embeddings.")
    use_cache: bool = Field(default=True, description="Enable KV cache during generation.")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="bfloat16", description="Default parameter dtype."
    )

    @model_validator(mode="after")
    def check_heads_divide_hidden(self) -> "ModelConfig":
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        return self


class TrainingConfig(BaseModel):
    """Configuration for a training run."""

    run_name: str = Field(description="Experiment / run name.")
    model_id: str = Field(description="Target model identifier.")
    output_dir: str = Field(description="Directory for checkpoints and logs.")
    dataset_path: str = Field(description="Path or HuggingFace dataset identifier.")
    max_steps: int = Field(default=10_000, ge=1)
    warmup_steps: int = Field(default=500, ge=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    per_device_train_batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    save_steps: int = Field(default=1_000, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=True)
    seed: int = Field(default=42)


# Preset configurations for T-Series model family
T_SERIES_CONFIGS: dict[str, ModelConfig] = {
    "t101": ModelConfig(
        model_id="t101",
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        vocab_size=32000,
        max_position_embeddings=512,
    ),
    "t201": ModelConfig(
        model_id="t201",
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=1024,
    ),
    "t301": ModelConfig(
        model_id="t301",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
    ),
    "t501": ModelConfig(
        model_id="t501",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        max_position_embeddings=4096,
    ),
}
