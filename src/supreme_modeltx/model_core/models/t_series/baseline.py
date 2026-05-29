"""
model_core/models/t_series/baseline.py — T-Dev-6L: Small decoder-only transformer.

This is the canonical development / smoke-test model for supreme-modeltx.
It preserves the spirit of the T-series 6-layer baseline from the earlier
SMTX design, rebuilt from scratch as a clean PyTorch-native implementation.

Architecture overview:
  - Decoder-only (causal) transformer
  - RoPE positional embeddings (no learned position embeddings)
  - RMSNorm (pre-norm)
  - SwiGLU feed-forward network
  - Grouped Query Attention (GQA)
  - Tied or untied output embeddings

Default configuration: 6 layers, 512 hidden, 8 heads → ~25M parameters.
This fits on CPU for smoke runs and is GPU-scalable.

Provenance:
  Architecture inspired by LLaMA 2 (Meta AI, arXiv:2307.09288) and
  Mistral 7B (Mistral AI, arXiv:2310.06825). Implementation is original.
  See THIRD_PARTY_NOTICES.md for full provenance details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from supreme_modeltx.model_core.config.schema import ModelConfig
from supreme_modeltx.model_core.models.common.attention import GroupedQueryAttention


# ── RMS Normalisation ──────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ── SwiGLU Feed-Forward Network ────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: FFN(x) = swish(gate(x)) ⊙ up(x), projected by down.

    Reference: Noam Shazeer, "GLU Variants Improve Transformer" (2020).
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer Block ──────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """A single pre-norm decoder transformer block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            max_seq_len=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
        )
        self.ffn = SwiGLUFFN(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── T-Dev-6L Decoder Model ─────────────────────────────────────────────────────

class TSeriesBaseline(nn.Module):
    """T-Dev-6L: a small, clean decoder-only transformer for development and smoke tests.

    This model is the canonical starting point for the supreme-modeltx T-series.
    It is trainable on CPU (smoke runs), scalable to GPU (full training), and
    serves as the structural template for larger T-series variants.

    Args:
        config: :class:`ModelConfig` instance with hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None  # use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights using truncated normal with config std."""
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token ids, shape ``(B, T)``.
            attention_mask: Optional mask, shape ``(B, 1, T, T)`` or ``(B, T)``.
            labels: Optional shifted labels for causal LM loss, shape ``(B, T)``.
                    If provided, cross-entropy loss is returned.

        Returns:
            dict with keys:
              - ``"logits"``: shape ``(B, T, vocab_size)``
              - ``"loss"`` (optional): scalar CE loss when ``labels`` is given.
        """
        x = self.embed_tokens(input_ids)

        # Prepare causal mask if caller passes a 2-D padding mask
        attn_mask: torch.Tensor | None = None
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert (B, T) → (B, 1, T, T) boolean causal mask
            B, T = attention_mask.shape
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            pad = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_mask = causal.unsqueeze(0).unsqueeze(0) & pad

        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)

        x = self.norm(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        out: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # Shift for causal LM: predict token t+1 from token t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            out["loss"] = loss

        return out

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "TSeriesBaseline":
        """Construct from a :class:`ModelConfig`."""
        return cls(config)

    @classmethod
    def dev_model(cls) -> "TSeriesBaseline":
        """Return the canonical T-Dev-6L smoke-test model (~25M params)."""
        cfg = ModelConfig(
            model_variant="t-dev-6l",
            vocab_size=32_000,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=2048,
            max_position_embeddings=512,
        )
        return cls(cfg)
