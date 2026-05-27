"""
model_core/models/common/attention.py — Attention primitives.

Implements:
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA) with optional key-value head reduction
  - A causal self-attention layer compatible with the T-series baseline

Design notes:
  - Uses torch.nn.functional.scaled_dot_product_attention when available
    (PyTorch ≥ 2.0) for memory-efficient Flash-Attention-style kernels.
  - Supports both MHA (num_kv_heads == num_heads) and GQA.
  - RoPE follows the original Su et al. 2023 formulation.

Provenance:
  RoPE implementation inspired by the open-source implementations in
  LLaMA (Meta AI) and Mistral — see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rotary Position Embeddings ─────────────────────────────────────────────────

def _rope_freqs(dim: int, theta: float = 10_000.0) -> torch.Tensor:
    """Compute the inverse frequencies for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return inv_freq


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query/key tensors.

    Args:
        x: Shape ``(B, seq_len, n_heads, head_dim)``.
        freqs_cis: Precomputed complex-valued frequencies, shape ``(seq_len, head_dim // 2)``.

    Returns:
        Tensor with the same shape as *x* after RoPE rotation.
    """
    # Reshape to complex and apply rotation
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = x_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(rotated).reshape(x.shape).type_as(x)


def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10_000.0) -> torch.Tensor:
    """Precompute RoPE complex-valued frequency tensor.

    Returns shape ``(seq_len, head_dim // 2)`` complex64.
    """
    inv_freq = _rope_freqs(head_dim, theta)
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)


# ── Grouped Query Attention ────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """Causal self-attention with optional Grouped Query Attention (GQA).

    When ``num_kv_heads == num_heads`` this reduces to standard MHA.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (must divide ``num_heads``).
        max_seq_len: Maximum sequence length (for RoPE precompute).
        rope_theta: RoPE base frequency.
        dropout: Attention dropout probability (applied only in training).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10_000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_groups = num_heads // num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Precomputed RoPE frequencies — registered as buffer so they move with the model
        freqs = precompute_freqs_cis(self.head_dim, max_seq_len, theta=rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape ``(B, T, C)``.
            attention_mask: Optional boolean mask, shape ``(B, 1, T, T)``.

        Returns:
            Output tensor, same shape as *x*.
        """
        B, T, C = x.shape

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, kv_heads * head_dim)
        v = self.v_proj(x)

        # Reshape to (B, T, n_heads, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, self.freqs_cis[:T])
        k = apply_rope(k, self.freqs_cis[:T])

        # Transpose to (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads to match Q for GQA
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Use PyTorch's fused SDPA when available (Flash Attention on GPU, efficient on CPU)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)
