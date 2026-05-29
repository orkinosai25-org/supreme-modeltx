"""
model_core/eval/perplexity.py — Perplexity and validation evaluation.

Perplexity = exp(average negative log-likelihood per token).

This module provides:
  - evaluate_perplexity(): compute perplexity on a token batch iterator
  - A simple ValidationHook class for integration in the training loop

Provenance:
  Standard perplexity formulation; no external license concerns.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    batch_iter: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Compute perplexity on a sequence of batches.

    Each batch must contain ``"input_ids"`` and ``"labels"`` tensors.

    Args:
        model: The language model (must return dict with ``"loss"`` key).
        batch_iter: Iterator of batches.
        device: Device for computation.
        max_batches: Maximum number of batches to evaluate (None = all).

    Returns:
        Perplexity as a float.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(batch_iter):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]

        # Estimate number of non-padding tokens
        n_tokens = (labels != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    perplexity = math.exp(min(avg_nll, 100.0))  # clamp to avoid overflow
    logger.info("Perplexity: %.2f (avg NLL: %.4f, tokens: %d)", perplexity, avg_nll, total_tokens)
    model.train()
    return perplexity


class ValidationHook:
    """Simple hook that can be called from the training loop.

    Args:
        model: The model to evaluate.
        val_batch_fn: Callable returning an iterator of validation batches.
        device: Computation device.
        eval_every_n_steps: Evaluate every N training steps.
        max_batches: Limit validation to this many batches.
    """

    def __init__(
        self,
        model: nn.Module,
        val_batch_fn: Callable[[], Iterable[dict[str, torch.Tensor]]],
        device: torch.device,
        eval_every_n_steps: int = 500,
        max_batches: int = 50,
    ) -> None:
        self.model = model
        self.val_batch_fn = val_batch_fn
        self.device = device
        self.eval_every_n_steps = eval_every_n_steps
        self.max_batches = max_batches

    def __call__(self, step: int) -> float | None:
        """Call at each training step; returns perplexity when evaluated, else None."""
        if step % self.eval_every_n_steps == 0:
            return evaluate_perplexity(
                self.model,
                self.val_batch_fn(),
                device=self.device,
                max_batches=self.max_batches,
            )
        return None
