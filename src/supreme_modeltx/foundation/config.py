from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class FoundationModelConfig(BaseModel):
    model_name: str = "supreme-model-tx-pilot"
    vocab_size: int = Field(128, ge=32)
    hidden_size: int = Field(64, ge=32)
    num_layers: int = Field(2, ge=1)
    num_attention_heads: int = Field(4, ge=1)
    max_seq_len: int = Field(32, ge=8)
    dropout: float = Field(0.1, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def validate_attention_shape(self) -> "FoundationModelConfig":
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self


class DatasetConfig(BaseModel):
    train_path: str | None = None
    validation_path: str | None = None
    test_path: str | None = None
    required_fields: list[str] = Field(
        default_factory=lambda: ["prompt", "response", "source", "license"]
    )
    train_ratio: float = Field(0.8, gt=0.0, lt=1.0)
    validation_ratio: float = Field(0.1, gt=0.0, lt=1.0)
    test_ratio: float = Field(0.1, gt=0.0, lt=1.0)
    scan_pii: bool = False
    scan_secrets: bool = False

    @model_validator(mode="after")
    def validate_ratios(self) -> "DatasetConfig":
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train/validation/test ratios must sum to 1.0")
        return self


class TrainingLoopConfig(BaseModel):
    seed: int = 7
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(4, ge=1)
    gradient_accumulation_steps: int = Field(2, ge=1)
    max_steps: int = Field(6, ge=1)
    learning_rate: float = Field(5e-4, gt=0.0)
    weight_decay: float = Field(0.01, ge=0.0)
    mixed_precision: Literal["off", "bf16", "fp16"] = "off"
    validation_interval: int = Field(2, ge=1)
    checkpoint_interval: int = Field(2, ge=1)
    keep_last_n_checkpoints: int = Field(3, ge=1)
    max_grad_norm: float = Field(1.0, gt=0.0)
    output_dir: str = "artifacts/private_foundation/smoke_run"
    resume_from: str | None = None
    auto_resume_latest: bool = True
    train_samples: int = Field(32, ge=4)
    validation_samples: int = Field(8, ge=2)


class InferenceConfig(BaseModel):
    backend: Literal["stub", "checkpoint"] = "stub"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    checkpoint_path: str | None = None
    response_prefix: str = "Draft pilot response:"
    max_new_tokens: int = Field(48, ge=1)


class ExpectedBehaviourConfig(BaseModel):
    required_substrings: list[str] = Field(default_factory=list)
    forbidden_substrings: list[str] = Field(default_factory=list)
    exact_match: str | None = None
    max_characters: int | None = Field(default=None, ge=1)


class PromptTestCase(BaseModel):
    case_id: str
    prompt: str
    tags: list[str] = Field(default_factory=list)
    expected: ExpectedBehaviourConfig


class PromptSuiteConfig(BaseModel):
    suite_name: str
    version: str
    cases: list[PromptTestCase] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    suite_path: str
    output_path: str = "artifacts/private_foundation/evaluation/results.json"
    inference_config_path: str | None = None


class TrainingRunConfig(BaseModel):
    model: FoundationModelConfig = Field(default_factory=FoundationModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingLoopConfig = Field(default_factory=TrainingLoopConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingRunConfig":
        return load_model(path, cls)


class InferenceRunConfig(BaseModel):
    model: FoundationModelConfig = Field(default_factory=FoundationModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "InferenceRunConfig":
        return load_model(path, cls)


class EvaluationRunConfig(BaseModel):
    evaluation: EvaluationConfig
    model: FoundationModelConfig = Field(default_factory=FoundationModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationRunConfig":
        return load_model(path, cls)


def load_model(path: str | Path, model_type: type[BaseModel]) -> BaseModel:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    return model_type.model_validate(payload)


def save_model(path: str | Path, model: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(mode="json")
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
