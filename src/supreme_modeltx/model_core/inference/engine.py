"""Local inference engine for T-Series models.

Supports greedy and sampling-based generation with optional KV-cache.
Compatible with the platform_api serving layer.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps a TSeriesModel for text generation.

    Provides a :meth:`generate` method that accepts a list of token IDs
    and returns the generated continuation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        eos_id: int | None = None,
    ) -> list[int]:
        """Generate up to ``max_new_tokens`` tokens from ``input_ids``.

        Args:
            input_ids: Prompt token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = no scaling).
            top_p: Nucleus sampling probability threshold.
            eos_id: Stop generation when this token is produced.

        Returns:
            List of generated token IDs (not including the prompt).
        """
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=ids)
            logits = outputs["logits"][0, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cum_probs - sorted_probs) > top_p
                sorted_probs[cutoff] = 0.0
                sorted_probs /= sorted_probs.sum()
                next_token = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_token = torch.argmax(probs).item()

            generated.append(int(next_token))
            if eos_id is not None and next_token == eos_id:
                break

            ids = torch.cat(
                [ids, torch.tensor([[next_token]], device=self.device)], dim=1
            )

        return generated

    def generate_text(
        self,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> str:
        """End-to-end text generation with encode/decode."""
        input_ids = tokenizer.encode(prompt)
        eos_id = tokenizer.eos_id()
        generated_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_id=eos_id,
        )
        return tokenizer.decode(generated_ids)
