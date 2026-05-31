"""
model_core/training/trainer.py — Main training loop entrypoint.

A clean, readable training loop for the supreme-modeltx model core.
Supports:
  - Single-process CPU/GPU training
  - Distributed (torch.distributed) via torchrun
  - Mixed precision (BF16/FP16)
  - Gradient accumulation
  - Checkpoint save/resume
  - Periodic validation

Usage (single process):
    python -m supreme_modeltx.model_core.training.trainer --config config.json

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=8 -m supreme_modeltx.model_core.training.trainer \\
        --config config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from supreme_modeltx.model_core.config.schema import SMTXConfig
from supreme_modeltx.model_core.data.manifest import DataManifest
from supreme_modeltx.model_core.data.preprocessing import tokenize_and_pack
from supreme_modeltx.model_core.data.sources import iter_source
from supreme_modeltx.model_core.eval.perplexity import evaluate_perplexity
from supreme_modeltx.model_core.inference.engine import InferenceEngine
from supreme_modeltx.model_core.models.t_series.baseline import TSeriesBaseline
from supreme_modeltx.model_core.tokenizer.workflow import TokenizerWorkflow
from supreme_modeltx.model_core.training.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from supreme_modeltx.model_core.training.distributed.setup import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
)
from supreme_modeltx.model_core.training.optimizer import build_optimizer, clip_gradients
from supreme_modeltx.model_core.training.precision import get_autocast_context, get_grad_scaler
from supreme_modeltx.model_core.training.scheduler import build_scheduler
from supreme_modeltx.utils.device import get_device
from supreme_modeltx.utils.logging import configure_logging

logger = logging.getLogger("supreme_modeltx.train")


# ── Minimal synthetic dataset for smoke/dev runs ───────────────────────────────

def _synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Generate a random token batch (for smoke runs without real data)."""
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {"input_ids": ids, "labels": ids}


def _dummy_data_iter(cfg: SMTXConfig, device: torch.device) -> Iterator[dict[str, torch.Tensor]]:
    """Infinite iterator of synthetic batches (used when no real data is configured)."""
    while True:
        yield _synthetic_batch(
            batch_size=cfg.training.batch_size,
            seq_len=min(cfg.data.max_seq_len, cfg.model.max_position_embeddings),
            vocab_size=cfg.model.vocab_size,
            device=device,
        )


def _iter_manifest_text(manifest: DataManifest, *, split: str) -> Iterator[str]:
    """Infinite text iterator over manifest sources."""
    matching_sources = [source for source in manifest.sources if source.split == split]
    if not matching_sources:
        raise ValueError(f"Manifest has no sources for split='{split}'")
    while True:
        for source in matching_sources:
            yield from iter_source(source)


def _manifest_data_iter(
    cfg: SMTXConfig,
    device: torch.device,
    *,
    split: str,
    batch_size: int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Infinite iterator of token batches from configured manifest sources."""
    tokenizer_path = cfg.data.tokenizer_path or cfg.tokenizer.model_path
    if not tokenizer_path:
        raise ValueError(
            "A tokenizer path is required for manifest-based training "
            "(set data.tokenizer_path or tokenizer.model_path)."
        )

    manifest = DataManifest.from_file(cfg.data.manifest_path)
    if not manifest.sources:
        raise ValueError(f"Manifest has no sources: {cfg.data.manifest_path}")

    tokenizer = TokenizerWorkflow(tokenizer_path, backend=cfg.tokenizer.backend)
    seq_len = min(cfg.data.max_seq_len, cfg.model.max_position_embeddings)
    packed_iter = tokenize_and_pack(
        text_iter=_iter_manifest_text(manifest, split=split),
        tokenizer_fn=tokenizer.as_callable(),
        max_seq_len=seq_len,
        pack=cfg.data.pack_sequences,
        eos_id=cfg.model.eos_token_id,
    )
    pad_id = cfg.model.pad_token_id
    effective_batch_size = batch_size or cfg.training.batch_size

    while True:
        sequences = list(next(packed_iter) for _ in range(effective_batch_size))
        input_ids = torch.full((effective_batch_size, seq_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq = seq[:seq_len]
            input_ids[i, : seq.numel()] = seq
        input_ids = input_ids.to(device)
        yield {"input_ids": input_ids, "labels": input_ids.clone()}


def _build_data_iter(cfg: SMTXConfig, device: torch.device) -> Iterator[dict[str, torch.Tensor]]:
    """Build the configured training data iterator."""
    if cfg.data.manifest_path:
        logger.info("Using manifest dataset (%s): %s", cfg.data.train_split, cfg.data.manifest_path)
        return _manifest_data_iter(cfg, device, split=cfg.data.train_split)
    logger.info("No manifest configured; using synthetic data iterator.")
    return _dummy_data_iter(cfg, device)


def _resolve_run_artifact_dir(cfg: SMTXConfig) -> Path:
    checkpoint_dir = Path(cfg.training.checkpoint.save_dir)
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    artifact_dir = run_dir / "run_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _extract_tokenizer_version(tokenizer_path: str | None) -> str | None:
    if not tokenizer_path:
        return None
    metadata_path = Path(tokenizer_path).parent / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = metadata.get("version")
    return str(version) if version else None


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _generate_checkpoint_samples(
    cfg: SMTXConfig,
    checkpoint_path: Path,
    artifact_dir: Path,
    *,
    generated_at_utc: str,
) -> Path | None:
    tokenizer_path = cfg.data.tokenizer_path or cfg.tokenizer.model_path
    if not tokenizer_path:
        return None

    samples_dir = artifact_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = TokenizerWorkflow(tokenizer_path, backend=cfg.tokenizer.backend)
    engine = InferenceEngine(
        model_config=cfg.model,
        checkpoint_path=checkpoint_path,
        dtype="float32",
    )
    prompts = [
        "Sovereign AI enables",
        "In this training run, the model should",
    ]
    prompt_rows: list[dict[str, Any]] = []
    tokenizer_vocab_size = tokenizer.vocab_size

    def _safe_decode(ids: list[int]) -> str:
        if not ids:
            return ""
        safe_ids = [token if 0 <= token < tokenizer_vocab_size else cfg.model.eos_token_id for token in ids]
        return tokenizer.decode(safe_ids)

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            continue
        input_ids = torch.tensor(prompt_ids, dtype=torch.long)
        output_ids = engine.generate(
            input_ids=input_ids,
            max_new_tokens=24,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            eos_id=cfg.model.eos_token_id,
        ).tolist()
        completion_ids = output_ids[len(prompt_ids):]
        prompt_rows.append(
            {
                "prompt": prompt,
                "prompt_token_count": len(prompt_ids),
                "completion_token_count": len(completion_ids),
                "completion_text": _safe_decode(completion_ids),
                "full_output_text": _safe_decode(output_ids),
            }
        )

    if not prompt_rows:
        return None

    sample_payload = {
        "checkpoint_path": str(checkpoint_path),
        "generated_at_utc": generated_at_utc,
        "generation": {
            "max_new_tokens": 24,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
        },
        "samples": prompt_rows,
    }
    sample_path = samples_dir / f"{checkpoint_path.stem}_samples.json"
    sample_path.write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")
    return sample_path


def _write_run_summary(
    cfg: SMTXConfig,
    artifact_dir: Path,
    *,
    started_at_utc: str,
    ended_at_utc: str,
    validation_history: list[dict[str, Any]],
    checkpoint_paths: list[str],
    sample_artifact_paths: list[str],
) -> None:
    config_path = artifact_dir / "config_used.json"
    config_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    latest_val = validation_history[-1] if validation_history else None
    tokenizer_path = cfg.data.tokenizer_path or cfg.tokenizer.model_path
    tokenizer_version = _extract_tokenizer_version(tokenizer_path)
    summary = {
        "run_name": Path(cfg.training.checkpoint.save_dir).parent.name,
        "timestamps": {
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
        },
        "git_commit": _get_git_commit(),
        "config_path": str(config_path),
        "tokenizer": {
            "path": tokenizer_path,
            "version": tokenizer_version,
            "backend": cfg.tokenizer.backend,
        },
        "checkpoint_paths": checkpoint_paths,
        "validation_history": validation_history,
        "latest_validation_loss": latest_val["val_loss"] if latest_val else None,
        "latest_perplexity": latest_val["perplexity"] if latest_val else None,
        "sample_artifact_paths": sample_artifact_paths,
    }

    summary_json_path = artifact_dir / "training_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md_path = artifact_dir / "training_summary.md"
    summary_md_path.write_text(
        "\n".join(
            [
                "# Training Run Summary",
                "",
                f"- Run: `{summary['run_name']}`",
                f"- Started (UTC): `{started_at_utc}`",
                f"- Ended (UTC): `{ended_at_utc}`",
                f"- Git commit: `{summary['git_commit'] or 'unavailable'}`",
                f"- Config: `{config_path}`",
                f"- Tokenizer path: `{tokenizer_path or 'unavailable'}`",
                f"- Tokenizer version: `{tokenizer_version or 'unknown'}`",
                f"- Latest validation loss: `{summary['latest_validation_loss']}`",
                f"- Latest perplexity: `{summary['latest_perplexity']}`",
                "",
                "## Checkpoints",
            ]
            + [f"- `{path}`" for path in checkpoint_paths]
            + ["", "## Sample artifacts"]
            + [f"- `{path}`" for path in sample_artifact_paths]
        )
        + "\n",
        encoding="utf-8",
    )


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: SMTXConfig, *, dry_run: bool = False) -> None:
    """Run the training loop described by *cfg*.

    Args:
        cfg: Fully-populated :class:`SMTXConfig`.
        dry_run: If True, run for 2 steps then return (useful for CI smoke tests).
    """
    configure_logging()
    is_dist = init_distributed(cfg.training.distributed.backend)
    device = get_device()

    torch.manual_seed(cfg.training.seed)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    run_artifact_dir = None if dry_run else _resolve_run_artifact_dir(cfg)
    validation_history: list[dict[str, Any]] = []
    sample_artifact_paths: list[str] = []

    model = TSeriesBaseline.from_config(cfg.model).to(device)
    if is_main_process():
        logger.info(
            "Model: %s | params: %s",
            cfg.model.model_variant,
            f"{model.num_parameters():,}",
        )

    # Distributed wrapping
    if is_dist and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = build_optimizer(model, cfg.training.optimizer)
    scheduler = build_scheduler(optimizer, cfg.training.scheduler, cfg.training.max_steps)
    scaler = get_grad_scaler(cfg.training.precision.dtype, device.type)

    # Resume from checkpoint if available
    start_step = 0
    resume_path = cfg.training.checkpoint.resume_from or (
        str(find_latest_checkpoint(cfg.training.checkpoint.save_dir))
        if find_latest_checkpoint(cfg.training.checkpoint.save_dir)
        else None
    )
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)
        logger.info("Resumed from step %d", start_step)

    data_iter = _build_data_iter(cfg, device)
    val_data_iter: Iterator[dict[str, torch.Tensor]] | None = None
    if cfg.data.manifest_path and cfg.data.validation_split:
        try:
            val_data_iter = _manifest_data_iter(
                cfg,
                device,
                split=cfg.data.validation_split,
                batch_size=max(1, cfg.training.batch_size),
            )
            logger.info(
                "Validation enabled (split=%s, every=%d steps, max_batches=%d)",
                cfg.data.validation_split,
                cfg.training.eval_every_n_steps,
                cfg.training.eval_max_batches,
            )
        except ValueError as exc:
            logger.warning("Validation disabled: %s", exc)

    max_steps = 2 if dry_run else cfg.training.max_steps
    accum_steps = cfg.training.gradient_accumulation_steps
    grad_clip = cfg.training.optimizer.grad_clip

    optimizer.zero_grad()
    running_loss = 0.0

    for step in range(start_step, max_steps):
        batch = next(data_iter)
        is_accum_step = (step + 1) % accum_steps != 0

        with get_autocast_context(cfg.training.precision.dtype, device_type=device.type):
            out = model(
                input_ids=batch["input_ids"],
                labels=batch.get("labels"),
            )
            loss = out["loss"] / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * accum_steps

        if not is_accum_step or step == max_steps - 1:
            if grad_clip > 0.0:
                if scaler:
                    scaler.unscale_(optimizer)
                clip_gradients(model, grad_clip)

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            avg_loss = running_loss / accum_steps
            running_loss = 0.0

            if is_main_process() and (step + 1) % cfg.training.log_every_n_steps == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "step=%d/%d | loss=%.4f | lr=%.2e",
                    step + 1, max_steps, avg_loss, lr,
                )

        if (
            is_main_process()
            and val_data_iter is not None
            and (step + 1) % cfg.training.eval_every_n_steps == 0
        ):
            val_loss, perplexity = evaluate_perplexity(
                model=model,
                batch_iter=val_data_iter,
                device=device,
                max_batches=cfg.training.eval_max_batches,
            )
            logger.info(
                "eval step=%d/%d | val_loss=%.4f | perplexity=%.2f",
                step + 1,
                max_steps,
                val_loss,
                perplexity,
            )
            validation_history.append(
                {
                    "step": step + 1,
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Save checkpoint
        if (
            is_main_process()
            and not dry_run
            and (step + 1) % cfg.training.checkpoint.save_every_n_steps == 0
        ):
            checkpoint_path = save_checkpoint(
                step + 1,
                model,
                optimizer,
                scheduler,
                save_dir=cfg.training.checkpoint.save_dir,
                keep_last_n=cfg.training.checkpoint.keep_last_n,
            )
            generated_at_utc = datetime.now(timezone.utc).isoformat()
            if run_artifact_dir is not None:
                try:
                    sample_path = _generate_checkpoint_samples(
                        cfg,
                        checkpoint_path,
                        run_artifact_dir,
                        generated_at_utc=generated_at_utc,
                    )
                except Exception as exc:  # pragma: no cover - best-effort artifact generation
                    logger.warning("Sample generation failed for %s: %s", checkpoint_path, exc)
                    sample_path = None
                if sample_path is not None:
                    sample_artifact_paths.append(str(sample_path))

    if is_main_process() and run_artifact_dir is not None:
        checkpoint_paths = sorted(
            str(path)
            for path in Path(cfg.training.checkpoint.save_dir).glob("checkpoint_step_*.pt")
        )
        _write_run_summary(
            cfg,
            run_artifact_dir,
            started_at_utc=started_at_utc,
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
            validation_history=validation_history,
            checkpoint_paths=checkpoint_paths,
            sample_artifact_paths=sample_artifact_paths,
        )
        logger.info("Training complete.")
    cleanup_distributed()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="supreme-modeltx training entrypoint")
    parser.add_argument("--config", type=str, help="Path to config JSON/YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 steps then exit.")
    args = parser.parse_args()

    if args.config:
        cfg = SMTXConfig.from_file(args.config)
    else:
        logger.warning("No config file provided — using default SMTXConfig (smoke run).")
        cfg = SMTXConfig()

    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
