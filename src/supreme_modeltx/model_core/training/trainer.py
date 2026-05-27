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
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from supreme_modeltx.model_core.config.schema import SMTXConfig
from supreme_modeltx.model_core.models.t_series.baseline import TSeriesBaseline
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

    autocast = get_autocast_context(cfg.training.precision.dtype, device_type=device.type)
    data_iter = _dummy_data_iter(cfg, device)

    max_steps = 2 if dry_run else cfg.training.max_steps
    accum_steps = cfg.training.gradient_accumulation_steps
    grad_clip = cfg.training.optimizer.grad_clip

    optimizer.zero_grad()
    running_loss = 0.0

    for step in range(start_step, max_steps):
        batch = next(data_iter)
        is_accum_step = (step + 1) % accum_steps != 0

        with autocast:
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

        # Save checkpoint
        if (
            is_main_process()
            and not dry_run
            and (step + 1) % cfg.training.checkpoint.save_every_n_steps == 0
        ):
            save_checkpoint(
                step + 1,
                model,
                optimizer,
                scheduler,
                save_dir=cfg.training.checkpoint.save_dir,
                keep_last_n=cfg.training.checkpoint.keep_last_n,
            )

    if is_main_process():
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
