"""
train_t101.py — T‑101 Base Model Training Script

CPU-first distributed training with optional GPU acceleration.

Compute profiles (set via TRAINING_PROFILE env var or --profile flag):
  cpu-single-node   — default; single process, no GPU required
  cpu-distributed   — multi-node CPU via torch.distributed + Gloo backend
  gpu-accelerated   — optional; requires CUDA quota; falls back to cpu if unavailable

Usage:
  # CPU single-node (default):
  python training/train_t101.py --config training/config_cpu_single.json

  # CPU distributed (launched by torchrun):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
           --master_addr=<ip> --master_port=29500 \\
           training/train_t101.py --config training/config_cpu_distributed.json

  # GPU (optional, when quota available):
  deepspeed --num_gpus=8 training/train_t101.py --config training/config_gpu_accelerated.json

Requirements:
    pip install torch transformers datasets accelerate sentencepiece peft
    # deepspeed only needed for gpu-accelerated profile:
    pip install deepspeed>=0.14.0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.train_t101")

# ── Supported compute profiles ────────────────────────────────────────────────
COMPUTE_PROFILES = {"cpu-single-node", "cpu-distributed", "gpu-accelerated"}


class ProgressLoggingCallback(TrainerCallback):
    """Log training progress to stdout so Azure Batch captures it."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        total = state.max_steps
        pct = 100.0 * step / total if total > 0 else 0.0
        parts = [f"step={step}/{total} ({pct:.1f}%)"]
        for k in ("loss", "learning_rate", "epoch"):
            if k in logs:
                parts.append(f"{k}={logs[k]:.6g}")
        logger.info("PROGRESS | " + " | ".join(parts))

    def on_save(self, args, state, control, **kwargs):
        logger.info("CHECKPOINT saved at step %d → %s", state.global_step, args.output_dir)


def detect_profile() -> str:
    """Determine the compute profile in priority order:
    1. TRAINING_PROFILE environment variable (set by Azure Batch task or CI)
    2. GPU available → gpu-accelerated
    3. WORLD_SIZE > 1 in environment → cpu-distributed
    4. Default → cpu-single-node
    """
    env_profile = os.environ.get("TRAINING_PROFILE", "").strip()
    if env_profile in COMPUTE_PROFILES:
        logger.info("Compute profile set by TRAINING_PROFILE env var: %s", env_profile)
        return env_profile
    if torch.cuda.is_available():
        logger.info("CUDA detected — selecting gpu-accelerated profile")
        return "gpu-accelerated"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        logger.info("WORLD_SIZE=%d detected — selecting cpu-distributed profile", world_size)
        return "cpu-distributed"
    logger.info("No GPU detected, single process — selecting cpu-single-node profile")
    return "cpu-single-node"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the T‑101 base model.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config_cpu_single.json",
        help="Path to the training configuration JSON file.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(COMPUTE_PROFILES),
        default=None,
        help=(
            "Compute profile override. Options: cpu-single-node (default), "
            "cpu-distributed, gpu-accelerated. "
            "Can also be set via TRAINING_PROFILE env var."
        ),
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_peft(model: AutoModelForCausalLM, peft_cfg: dict) -> AutoModelForCausalLM:
    """Wrap model with LoRA adapters for parameter-efficient CPU training.

    Freezes 90–99 % of weights so CPU gradient sync is cheap.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required for parameter-efficient training. "
            "Install it with: pip install peft"
        ) from exc

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_cfg.get("lora_r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied — trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total > 0 else 0.0,
    )
    return model


def build_tokenizer(model_config_path: str) -> AutoTokenizer:
    """Load or initialize the BPE tokenizer for T‑101."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_config_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_config_path: str) -> AutoModelForCausalLM:
    """Instantiate the T‑101 model from config."""
    config = AutoConfig.from_pretrained(model_config_path)
    model = AutoModelForCausalLM.from_config(config)
    return model


def preprocess_dataset(tokenizer: AutoTokenizer, cfg: dict):
    """Load and tokenize the training dataset."""
    data_cfg = cfg["data"]
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["validation_file"],
        },
    )

    max_seq_length = data_cfg.get("max_seq_length", 2048)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_cfg.get("preprocessing_num_workers", 4),
        remove_columns=raw_datasets["train"].column_names,
    )
    tokenized.set_format("torch")
    return tokenized


def resolve_output_dir(train_cfg: dict) -> str:
    """Return checkpoint output directory.

    Priority order:
    1. CHECKPOINT_DIR environment variable (set by Azure Batch / container)
    2. config output_dir value
    3. /mnt/checkpoints/t101-baby (default Azure file-share path)
    """
    env_dir = os.environ.get("CHECKPOINT_DIR")
    if env_dir:
        return env_dir
    configured = train_cfg.get("output_dir")
    if configured:
        return configured
    return "/mnt/checkpoints/t101-baby"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # ── Resolve compute profile ───────────────────────────────────────────────
    # CLI flag overrides env var and auto-detect.
    if args.profile:
        os.environ["TRAINING_PROFILE"] = args.profile
    profile = detect_profile()
    cfg_profile = cfg.get("profile")
    if cfg_profile and cfg_profile != profile:
        logger.warning(
            "Config file declares profile '%s' but resolved profile is '%s'. "
            "Using resolved profile.",
            cfg_profile,
            profile,
        )

    # GPU fallback: resolve before any logging so the active profile is correct.
    if profile == "gpu-accelerated" and not torch.cuda.is_available():
        logger.warning(
            "Profile is 'gpu-accelerated' but no CUDA device found. "
            "Falling back to cpu-distributed if WORLD_SIZE > 1, else cpu-single-node."
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        profile = "cpu-distributed" if world_size > 1 else "cpu-single-node"

    is_cpu = profile in {"cpu-single-node", "cpu-distributed"}
    is_distributed = profile == "cpu-distributed"

    logger.info("Active compute profile: %s", profile)
    if is_cpu and not torch.cuda.is_available():
        logger.info("No GPU quota detected — running fully on CPU (as expected).")

    train_cfg = cfg["training"]
    set_seed(train_cfg.get("seed", 42))

    output_dir = resolve_output_dir(train_cfg)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoints will be saved to: %s", output_dir)

    model_config_path = cfg.get("model_config", "tmodels/t101")
    logger.info("Loading tokenizer from: %s", model_config_path)
    tokenizer = build_tokenizer(model_config_path)

    logger.info("Building model from: %s", model_config_path)
    model = build_model(model_config_path)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameter count: %s", f"{num_params:,}")

    # ── Parameter-efficient fine-tuning (LoRA) — strongly recommended on CPU ─
    # Default: LoRA ON for CPU (makes distributed gradient sync practical),
    # LoRA OFF for GPU (full fine-tuning leverages CUDA memory/speed).
    # Override via `use_peft` in the training config.
    use_peft = train_cfg.get("use_peft", is_cpu)
    if use_peft:
        peft_cfg = cfg.get("peft", {})
        model = apply_peft(model, peft_cfg)
    else:
        logger.info("Full fine-tuning (LoRA disabled).")

    logger.info("Preprocessing dataset …")
    tokenized_datasets = preprocess_dataset(tokenizer, cfg)
    logger.info(
        "Dataset ready — train: %d samples, val: %d samples",
        len(tokenized_datasets["train"]),
        len(tokenized_datasets["validation"]),
    )

    # ── Build TrainingArguments — CPU-safe defaults ───────────────────────────
    # bf16 requires CUDA; disable it for all CPU profiles.
    use_bf16 = train_cfg.get("bf16", False) and not is_cpu
    use_fp16 = train_cfg.get("fp16", False) and not is_cpu

    # deepspeed only applies for gpu-accelerated profile
    deepspeed_cfg = cfg.get("deepspeed") if not is_cpu else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=use_bf16,
        fp16=use_fp16,
        no_cuda=is_cpu,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        evaluation_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", train_cfg.get("save_steps", 100)),
        deepspeed=deepspeed_cfg,
        report_to="none",
        logging_dir=str(Path(output_dir) / "logs"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
    )

    if is_distributed:
        # torch.distributed must already be initialised by torchrun before
        # the Trainer is constructed. Log for observability.
        logger.info(
            "Distributed CPU training — rank=%s world_size=%s backend=gloo",
            os.environ.get("RANK", "0"),
            os.environ.get("WORLD_SIZE", "1"),
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[ProgressLoggingCallback()],
    )

    logger.info("Starting training (profile=%s) …", profile)
    train_result = trainer.train()

    logger.info("Training complete — saving final checkpoint …")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(
        "T‑101 training complete. profile=%s | saved to %s | steps=%d | loss=%.4f",
        profile,
        output_dir,
        metrics.get("train_steps", 0),
        metrics.get("train_loss", float("nan")),
    )


if __name__ == "__main__":
    main()
