"""
model_core/inference/sampling.py — Token sampling utilities.

Provides:
  - Greedy decoding
  - Temperature scaling
  - Top-k filtering
  - Top-p (nucleus) sampling

Provenance:
  Standard sampling algorithms; implementation is original.
  Top-p/Top-k methods follow the formulations in Holtzman et al. (2020)
  "The Curious Case of Neural Text Degeneration".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample the next token from *logits*.

    Args:
        logits: Shape ``(vocab_size,)`` or ``(1, vocab_size)``.
        temperature: Scaling factor. ``0`` means greedy (argmax).
        top_k: If > 0, keep only the top-k logits.
        top_p: Nucleus filtering threshold (1.0 = disabled).

    Returns:
        Scalar token id tensor.
    """
    if logits.dim() == 2:
        logits = logits[0]  # (vocab_size,)

    if temperature == 0.0:
        return torch.argmax(logits)

    logits = logits / temperature

    if top_k > 0:
        logits = _top_k_filter(logits, top_k)

    if top_p < 1.0:
        logits = _top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out logits outside the top-k."""
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out logits outside the nucleus (top-p)."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens that push cumulative prob above the threshold
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove_mask] = float("-inf")
    # Restore original ordering
    return sorted_logits.scatter(0, sorted_indices, sorted_logits)
