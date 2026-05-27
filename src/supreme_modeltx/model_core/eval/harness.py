"""Evaluation harness for T-Series models.

Provides perplexity measurement and a minimal few-shot accuracy
evaluation loop compatible with standard benchmark formats.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EvalHarness:
    """Run evaluation metrics against a trained T-Series model.

    Supported metrics:
    - **perplexity** — cross-entropy loss exponentiated over a held-out set
    - **accuracy** — exact-match accuracy for classification/QA benchmarks
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """Compute perplexity over a token-level dataloader."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss: torch.Tensor = outputs["loss"]

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss)
        logger.info("Perplexity: %.4f (avg_loss=%.4f)", ppl, avg_loss)
        return ppl

    @torch.no_grad()
    def compute_accuracy(self, examples: list[dict[str, Any]]) -> float:
        """Compute exact-match accuracy.

        Each example must have ``"input_ids"`` (list[int]) and
        ``"label"`` (int, the correct next-token id).
        """
        self.model.eval()
        correct = 0
        for ex in examples:
            input_ids = torch.tensor([ex["input_ids"]], device=self.device)
            outputs = self.model(input_ids=input_ids)
            predicted = outputs["logits"][0, -1, :].argmax().item()
            if predicted == ex["label"]:
                correct += 1
        acc = correct / max(len(examples), 1)
        logger.info("Accuracy: %.4f (%d/%d)", acc, correct, len(examples))
        return acc
