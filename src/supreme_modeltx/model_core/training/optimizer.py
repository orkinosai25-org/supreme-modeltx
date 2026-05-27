"""
model_core/training/optimizer.py — Optimizer and gradient-clipping helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from supreme_modeltx.model_core.config.schema import OptimizerConfig


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    """Construct the optimizer from config.

    Separates weight-decay groups (no decay for biases and norm params).
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = cfg.name.lower()
    if name in ("adamw", "adam"):
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=cfg.lr, momentum=0.9)

    raise ValueError(f"Unknown optimizer: {cfg.name!r}")


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients and return the pre-clip global norm."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()
