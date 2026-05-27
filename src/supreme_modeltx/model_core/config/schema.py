"""
model_core/config/schema.py — Pydantic configuration schema for
model, training, and data settings.

Inspired by the clean config patterns from LitGPT and torchtitan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Model configuration ────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Core hyperparameters for a decoder-only transformer model."""

    model_family: str = Field("t-series", description="Model family identifier.")
    model_variant: str = Field("t-dev-6l", description="Variant tag, e.g. t-dev-6l, t101.")
    version: str = Field("0.1.0")

    vocab_size: int = Field(32_000, ge=256)
    hidden_size: int = Field(512, ge=64)
    num_hidden_layers: int = Field(6, ge=1)
    num_attention_heads: int = Field(8, ge=1)
    num_key_value_heads: int = Field(8, ge=1, description="GQA groups; equals num_attention_heads for MHA.")
    intermediate_size: int = Field(2048, ge=64, description="FFN intermediate (SwiGLU) width.")
    max_position_embeddings: int = Field(2048, ge=64)

    rms_norm_eps: float = Field(1e-5, gt=0.0)
    rope_theta: float = Field(10_000.0, gt=0.0)
    hidden_act: Literal["swiglue", "gelu", "relu"] = "swiglue"

    tie_word_embeddings: bool = False
    torch_dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    initializer_range: float = Field(0.02, gt=0.0)

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @field_validator("num_key_value_heads")
    @classmethod
    def _kv_heads_divides_attn(cls, v: int, info) -> int:  # noqa: N805
        heads = info.data.get("num_attention_heads", v)
        if heads % v != 0:
            raise ValueError(
                f"num_key_value_heads ({v}) must divide num_attention_heads ({heads})"
            )
        return v


# ── Training configuration ─────────────────────────────────────────────────────

class CheckpointConfig(BaseModel):
    """Checkpoint save / resume settings."""

    save_dir: str = "checkpoints"
    save_every_n_steps: int = Field(500, ge=1)
    keep_last_n: int = Field(3, ge=1, description="Number of recent checkpoints to retain.")
    resume_from: Optional[str] = None


class PrecisionConfig(BaseModel):
    """Mixed-precision / BF16 training settings."""

    enabled: bool = True
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class OptimizerConfig(BaseModel):
    """Optimiser settings."""

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = Field(3e-4, gt=0.0)
    weight_decay: float = Field(0.1, ge=0.0)
    beta1: float = Field(0.9, ge=0.0, le=1.0)
    beta2: float = Field(0.95, ge=0.0, le=1.0)
    eps: float = Field(1e-8, gt=0.0)
    grad_clip: float = Field(1.0, ge=0.0)


class SchedulerConfig(BaseModel):
    """LR scheduler settings."""

    name: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = Field(100, ge=0)
    min_lr_ratio: float = Field(0.1, ge=0.0, le=1.0)


class DistributedConfig(BaseModel):
    """Distributed training settings."""

    backend: Literal["nccl", "gloo", "auto"] = "auto"
    fsdp_enabled: bool = False
    fsdp_sharding_strategy: Literal["full", "shard_grad_op", "no_shard"] = "full"


class TrainingConfig(BaseModel):
    """Full training loop configuration."""

    max_steps: int = Field(10_000, ge=1)
    batch_size: int = Field(8, ge=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    seed: int = 42

    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    log_every_n_steps: int = Field(50, ge=1)
    eval_every_n_steps: int = Field(500, ge=1)


# ── Data configuration ─────────────────────────────────────────────────────────

class DataConfig(BaseModel):
    """Dataset manifest and preprocessing settings."""

    manifest_path: Optional[str] = None
    data_dirs: list[str] = Field(default_factory=list)
    format: Literal["jsonl", "parquet", "hf_dataset", "text"] = "jsonl"
    hf_dataset_name: Optional[str] = None
    hf_dataset_split: str = "train"
    max_seq_len: int = Field(2048, ge=64)
    pack_sequences: bool = True
    num_workers: int = Field(4, ge=0)
    tokenizer_path: Optional[str] = None


# ── Tokenizer configuration ────────────────────────────────────────────────────

class TokenizerConfig(BaseModel):
    """Tokenizer (SentencePiece-oriented) workflow settings."""

    backend: Literal["sentencepiece", "hf_tokenizers"] = "sentencepiece"
    model_path: Optional[str] = None
    vocab_size: int = Field(32_000, ge=256)
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"


# ── Root config ────────────────────────────────────────────────────────────────

class SMTXConfig(BaseModel):
    """Root configuration combining all sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "SMTXConfig":
        """Load configuration from a JSON or YAML file."""
        path = Path(path)
        raw = path.read_text()
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return cls.model_validate(data)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())
