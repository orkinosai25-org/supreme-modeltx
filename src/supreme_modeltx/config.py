from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    vocab_size: int = 256
    sequence_length: int = 128
    batch_size: int = 16
    embedding_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    ffw_hidden_dim: int = 1536
    dropout: float = 0.1
    learning_rate: float = 3e-4
    max_steps: int = 200
    eval_interval: int = 25
    eval_batches: int = 4
    checkpoint_dir: str = "checkpoints"
    gradient_clip_norm: float = 1.0
    use_amp: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    base = TrainingConfig().to_dict()
    unknown_keys = sorted(set(raw.keys()) - set(base.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    base.update(raw)
    return TrainingConfig(**base)
