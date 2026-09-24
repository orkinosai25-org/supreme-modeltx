from __future__ import annotations

import argparse
import json
import logging
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from supreme_modeltx.foundation.config import TrainingRunConfig
from supreme_modeltx.foundation.dataset_tools import load_jsonl
from supreme_modeltx.foundation.tokenization import CharacterTokenizer

logger = logging.getLogger("supreme_modeltx.foundation.training")


class TinyPilotLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_attention_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        logits = self.head(self.norm(hidden))
        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return output


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_path(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"checkpoint_step_{step:08d}.pt"


def training_state_path(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"checkpoint_step_{step:08d}.state.pt"


def training_metadata_path(output_dir: str | Path, step: int) -> Path:
    return training_state_path(output_dir, step).with_suffix(".json")


def _step_from_checkpoint_path(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)(?:\.state)?\.pt$", path.name)
    if not match:
        raise ValueError(f"Unrecognized checkpoint filename: {path.name}")
    return int(match.group(1))


def _autocast_context(device: torch.device, mode: str):
    if mode == "off":
        return nullcontext()
    if device.type == "cuda":
        dtype = torch.bfloat16 if mode == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and mode == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _build_scaler(device: torch.device, mode: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and mode == "fp16")


def _generate_synthetic_samples(num_samples: int, seq_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randint(1, vocab_size - 1) for _ in range(seq_len)]
        for _ in range(num_samples)
    ]


def _encode_records(path: str | Path, *, vocab_size: int, seq_len: int) -> list[list[int]]:
    tokenizer = CharacterTokenizer(vocab_size)
    records = load_jsonl(path)
    sequences: list[list[int]] = []
    for record in records:
        prompt = str(record.get("prompt", ""))
        response = str(record.get("response", ""))
        combined = f"{prompt}\n{response}".strip()
        sequences.append(tokenizer.encode(combined, max_length=seq_len))
    if not sequences:
        raise ValueError(f"Dataset contains no usable records: {path}")
    return sequences


def _prepare_samples(cfg: TrainingRunConfig) -> tuple[list[list[int]], list[list[int]]]:
    seq_len = cfg.model.max_seq_len
    vocab_size = cfg.model.vocab_size
    if cfg.dataset.train_path:
        train_samples = _encode_records(cfg.dataset.train_path, vocab_size=vocab_size, seq_len=seq_len)
    else:
        train_samples = _generate_synthetic_samples(
            cfg.training.train_samples,
            seq_len,
            vocab_size,
            cfg.training.seed,
        )

    if cfg.dataset.validation_path:
        validation_samples = _encode_records(cfg.dataset.validation_path, vocab_size=vocab_size, seq_len=seq_len)
    else:
        validation_samples = _generate_synthetic_samples(
            cfg.training.validation_samples,
            seq_len,
            vocab_size,
            cfg.training.seed + 1,
        )
    return train_samples, validation_samples


def _make_batch(samples: list[list[int]], *, step: int, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    total = len(samples)
    if total == 0:
        raise ValueError("Cannot build a batch from an empty dataset.")
    batch = [samples[(step * batch_size + offset) % total] for offset in range(batch_size)]
    tensor = torch.tensor(batch, dtype=torch.long, device=device)
    return {"input_ids": tensor, "labels": tensor.clone()}


def _save_checkpoint(
    *,
    cfg: TrainingRunConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    device: torch.device,
    metrics: list[dict[str, Any]],
    interrupted: bool = False,
) -> Path:
    path = checkpoint_path(cfg.training.output_dir, step)
    state_path = training_state_path(cfg.training.output_dir, step)
    metadata_path = training_metadata_path(cfg.training.output_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
        },
        path,
    )
    torch.save(
        {
            "step": step,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "interrupted": interrupted,
        },
        state_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "config": cfg.model_dump(mode="json"),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoints = sorted(path.parent.glob("checkpoint_step_*.pt"), key=_step_from_checkpoint_path)
    model_checkpoints = [candidate for candidate in checkpoints if ".state." not in candidate.name]
    for old_path in model_checkpoints[:-cfg.training.keep_last_n_checkpoints]:
        old_path.unlink()
        sibling_state = training_state_path(old_path.parent.parent, _step_from_checkpoint_path(old_path))
        sibling_meta = training_metadata_path(old_path.parent.parent, _step_from_checkpoint_path(old_path))
        if sibling_state.exists():
            sibling_state.unlink()
        if sibling_meta.exists():
            sibling_meta.unlink()
    return path


def _resolve_resume_path(cfg: TrainingRunConfig) -> Path | None:
    if cfg.training.resume_from:
        return Path(cfg.training.resume_from)
    if not cfg.training.auto_resume_latest:
        return None
    checkpoint_dir = Path(cfg.training.output_dir) / "checkpoints"
    candidates = sorted(
        (
            candidate
            for candidate in checkpoint_dir.glob("checkpoint_step_*.pt")
            if ".state." not in candidate.name
        ),
        key=_step_from_checkpoint_path,
    )
    return candidates[-1] if candidates else None


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    step = _step_from_checkpoint_path(path)
    model_state = torch.load(path, map_location=device, weights_only=True)
    training_state = torch.load(
        training_state_path(path.parent.parent, step),
        map_location=device,
        weights_only=True,
    )
    metrics_path = training_metadata_path(path.parent.parent, step)
    metrics: list[dict[str, Any]] = []
    if metrics_path.exists():
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = list(metadata.get("metrics", []))
    model.load_state_dict(model_state["model_state"])
    optimizer.load_state_dict(training_state["optimizer_state"])
    scaler.load_state_dict(training_state.get("scaler_state", {}))
    return int(training_state.get("step", 0)), metrics


def _evaluate(model: nn.Module, validation_samples: list[list[int]], *, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        total_batches = max(1, (len(validation_samples) + batch_size - 1) // batch_size)
        for batch_index in range(total_batches):
            batch = _make_batch(validation_samples, step=batch_index, batch_size=batch_size, device=device)
            losses.append(float(model(**batch)["loss"].detach().cpu().item()))
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainingRunConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_determinism(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    train_samples, validation_samples = _prepare_samples(cfg)

    model = TinyPilotLM(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = _build_scaler(device, cfg.training.mixed_precision)
    metrics: list[dict[str, Any]] = []
    start_step = 0

    resume_path = _resolve_resume_path(cfg)
    if resume_path and resume_path.exists():
        start_step, metrics = _load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        logger.info("resumed checkpoint=%s step=%s", resume_path, start_step)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = None
    current_step = start_step
    try:
        for step in range(start_step + 1, cfg.training.max_steps + 1):
            current_step = step
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for accumulation_index in range(cfg.training.gradient_accumulation_steps):
                batch = _make_batch(
                    train_samples,
                    step=(step - 1) * cfg.training.gradient_accumulation_steps + accumulation_index,
                    batch_size=cfg.training.batch_size,
                    device=device,
                )
                with _autocast_context(device, cfg.training.mixed_precision):
                    output = model(**batch)
                    loss = output["loss"] / cfg.training.gradient_accumulation_steps
                total_loss += float(loss.detach().cpu().item())
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            metric: dict[str, Any] = {"step": step, "train_loss": total_loss}
            if step % cfg.training.validation_interval == 0 or step == cfg.training.max_steps:
                metric["validation_loss"] = _evaluate(
                    model,
                    validation_samples,
                    batch_size=cfg.training.batch_size,
                    device=device,
                )
                logger.info(
                    "step=%s train_loss=%.4f validation_loss=%.4f",
                    step,
                    metric["train_loss"],
                    metric["validation_loss"],
                )
            else:
                logger.info("step=%s train_loss=%.4f", step, metric["train_loss"])
            metrics.append(metric)

            if step % cfg.training.checkpoint_interval == 0 or step == cfg.training.max_steps:
                latest_checkpoint = _save_checkpoint(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    device=device,
                    metrics=metrics,
                )
    except KeyboardInterrupt:
        latest_checkpoint = _save_checkpoint(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=current_step,
            device=device,
            metrics=metrics,
            interrupted=True,
        )
        logger.info("training interrupted; checkpoint saved to %s", latest_checkpoint)
        status = "interrupted"
    else:
        status = "completed"

    summary = {
        "status": status,
        "device": str(device),
        "seed": cfg.training.seed,
        "steps_completed": metrics[-1]["step"] if metrics else current_step,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "latest_training_state": str(latest_checkpoint.with_suffix(".state.pt")) if latest_checkpoint else None,
        "metrics_path": str(output_dir / "metrics.jsonl"),
    }
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric) + "\n")
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supreme Model T-X private foundation trainer.")
    parser.add_argument(
        "--config",
        default="configs/foundation/training-smoke.yaml",
        help="Path to a training config file.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = TrainingRunConfig.from_file(args.config)
    print(json.dumps(train(config), indent=2))


if __name__ == "__main__":
    main()
