"""
platform_api/api/engine.py — InferenceEngine singleton for the platform API.

Loads a checkpoint-backed InferenceEngine at application startup when the
required environment variables are present:

    SMTX_CHECKPOINT_PATH   — Path to the .pt model checkpoint (required)
    SMTX_TOKENIZER_PATH    — Path to a SentencePiece .model tokenizer file (required)
    SMTX_MODEL_CONFIG_PATH — Path to a model config JSON/YAML file (optional; defaults apply)
    SMTX_INFERENCE_DTYPE   — Inference dtype: bfloat16 | float16 | float32 (default: bfloat16)

When the env vars are absent or the files do not exist the engine is left
unloaded and inference endpoints respond with HTTP 503.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from supreme_modeltx.platform_api.model_registry.registry import ModelEntry, ModelRegistry

if TYPE_CHECKING:
    from supreme_modeltx.model_core.inference.engine import InferenceEngine
    from supreme_modeltx.model_core.tokenizer.workflow import TokenizerWorkflow

logger = logging.getLogger(__name__)

_engine_backend: Optional["_InferenceBackend"] = None


class _InferenceBackend:
    """Wraps a loaded :class:`InferenceEngine` and :class:`TokenizerWorkflow`."""

    def __init__(
        self,
        engine: "InferenceEngine",
        tokenizer: "TokenizerWorkflow",
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer

    def generate_from_messages(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> tuple[str, int, int]:
        """Run inference on a list of ChatMessage objects.

        Args:
            messages: List of :class:`ChatMessage` objects with ``role`` and
                ``content`` attributes.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            A 3-tuple ``(generated_text, prompt_tokens, completion_tokens)``.
        """
        prompt = _format_prompt(messages)
        prompt_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        generated = self.engine.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        generated_ids = generated.tolist()
        new_ids = generated_ids[len(prompt_ids):]
        completion_text = self.tokenizer.decode(new_ids)
        return completion_text, len(prompt_ids), len(new_ids)


def _format_prompt(messages: list) -> str:
    """Convert a list of ChatMessage objects into a flat prompt string."""
    parts = [f"{m.role}: {m.content}" for m in messages]
    return "\n".join(parts) + "\nassistant:"


def get_engine() -> Optional[_InferenceBackend]:
    """Return the active inference backend, or ``None`` if not configured."""
    return _engine_backend


def initialize_engine() -> None:
    """Load the InferenceEngine and TokenizerWorkflow from environment variables.

    Called once at application startup.  Safe to call multiple times (no-op if
    already loaded).
    """
    global _engine_backend
    if _engine_backend is not None:
        return

    checkpoint_path = os.environ.get("SMTX_CHECKPOINT_PATH", "")
    tokenizer_path = os.environ.get("SMTX_TOKENIZER_PATH", "")
    model_config_path = os.environ.get("SMTX_MODEL_CONFIG_PATH", "")
    dtype = os.environ.get("SMTX_INFERENCE_DTYPE", "bfloat16")

    if not checkpoint_path or not tokenizer_path:
        logger.info(
            "InferenceEngine not configured — set SMTX_CHECKPOINT_PATH and "
            "SMTX_TOKENIZER_PATH to enable local inference."
        )
        return

    if not Path(checkpoint_path).exists():
        logger.warning("Checkpoint not found at %s — inference disabled.", checkpoint_path)
        return

    if not Path(tokenizer_path).exists():
        logger.warning("Tokenizer not found at %s — inference disabled.", tokenizer_path)
        return

    try:
        from supreme_modeltx.model_core.config.schema import ModelConfig, SMTXConfig
        from supreme_modeltx.model_core.inference.engine import InferenceEngine
        from supreme_modeltx.model_core.tokenizer.workflow import TokenizerWorkflow

        if model_config_path and Path(model_config_path).exists():
            model_config = SMTXConfig.from_file(model_config_path).model
        else:
            model_config = ModelConfig()

        engine = InferenceEngine(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
            dtype=dtype,
        )
        tokenizer = TokenizerWorkflow(tokenizer_path)
        _engine_backend = _InferenceBackend(engine=engine, tokenizer=tokenizer)
        _register_loaded_model(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            model_config_path=model_config_path,
            dtype=dtype,
            model_family=model_config.model_family,
            model_variant=model_config.model_variant,
            context_length=model_config.max_position_embeddings,
        )
        logger.info("InferenceEngine loaded from checkpoint: %s", checkpoint_path)
    except Exception:
        logger.exception("Failed to load InferenceEngine — inference disabled.")


def _register_loaded_model(
    *,
    checkpoint_path: str,
    tokenizer_path: str,
    model_config_path: str,
    dtype: str,
    model_family: str,
    model_variant: str,
    context_length: int,
) -> None:
    """Persist metadata for the currently served checkpoint-backed model."""
    checkpoint = Path(checkpoint_path).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    model_id = os.environ.get("SMTX_MODEL_ID", f"served-{checkpoint.stem}")
    model_name = os.environ.get("SMTX_MODEL_NAME", checkpoint.stem)
    raw_stage = os.environ.get("SMTX_MODEL_STAGE", "production")
    stage = raw_stage if raw_stage in {"development", "staging", "production", "deprecated"} else "production"

    registry = ModelRegistry()
    registry.register(
        ModelEntry(
            id=model_id,
            name=model_name,
            family=model_family,
            variant=model_variant,
            stage=stage,
            description="Runtime-served checkpoint model.",
            context_length=context_length,
            checkpoint_path=str(checkpoint),
            tokenizer_path=str(tokenizer),
            model_config_path=str(Path(model_config_path).resolve()) if model_config_path else None,
            inference_dtype=dtype,
            is_available=True,
        )
    )
