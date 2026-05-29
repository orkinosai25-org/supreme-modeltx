"""
model_core/inference/engine.py — Inference engine: checkpoint loading and generation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from supreme_modeltx.model_core.config.schema import ModelConfig
from supreme_modeltx.model_core.models.t_series.baseline import TSeriesBaseline
from supreme_modeltx.model_core.training.checkpoint import load_checkpoint
from supreme_modeltx.model_core.inference.sampling import sample_tokens
from supreme_modeltx.utils.device import get_device

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Load a trained TSeriesBaseline checkpoint and run generation.

    Args:
        model_config: :class:`ModelConfig` matching the checkpoint.
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Computation device (default: auto-detected).
        dtype: Inference dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        self.device = device or get_device()
        self.model_config = model_config

        model = TSeriesBaseline.from_config(model_config)
        load_checkpoint(checkpoint_path, model, map_location=self.device)
        model.eval()

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.float32)
        self.model = model.to(self.device).to(self.dtype)
        logger.info("InferenceEngine ready on %s (dtype=%s)", self.device, dtype)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation from *input_ids*.

        Args:
            input_ids: Prompt token ids, shape ``(1, T)`` or ``(T,)``.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling (0 = disabled).
            eos_id: Stop generation when this token is produced.

        Returns:
            Token id tensor of shape ``(T + generated,)``.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        eos_id = eos_id if eos_id is not None else self.model_config.eos_token_id

        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            # Truncate context to max_position_embeddings
            ctx = generated[:, -self.model_config.max_position_embeddings:]
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                out = self.model(input_ids=ctx)
            next_token = sample_tokens(
                out["logits"][:, -1, :],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break

        return generated[0]
