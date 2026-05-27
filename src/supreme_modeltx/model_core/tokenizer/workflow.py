"""
model_core/tokenizer/workflow.py — Tokenizer workflow boundary.

Provides a unified TokenizerWorkflow class that wraps either:
  - A trained SentencePiece model (.model file)
  - A HuggingFace tokenizer (via the `tokenizers` library)

This module defines the workflow boundary — callers should always
use TokenizerWorkflow rather than importing sentencepiece directly,
so the underlying implementation can be swapped without API changes.

SentencePiece training:
  modeltx-tokenizer-train is a thin wrapper around the spm.SentencePieceTrainer,
  which should be trained on the sovereign pretraining corpus before
  any model training begins.

Provenance:
  SentencePiece: Kudo & Richardson, 2018 (Apache-2.0).
  HuggingFace tokenizers: HuggingFace (Apache-2.0).
  See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class TokenizerWorkflow:
    """Unified tokenizer interface for model training and inference.

    Args:
        model_path: Path to a SentencePiece ``.model`` file or HF tokenizer directory.
        backend: ``"sentencepiece"`` or ``"hf_tokenizers"``.
    """

    def __init__(self, model_path: str | Path, backend: str = "sentencepiece") -> None:
        self.model_path = Path(model_path)
        self.backend = backend
        self._sp = None
        self._hf = None
        self._load()

    def _load(self) -> None:
        if self.backend == "sentencepiece":
            try:
                import sentencepiece as spm
                self._sp = spm.SentencePieceProcessor()
                self._sp.Load(str(self.model_path))
                logger.info("SentencePiece tokenizer loaded: %s", self.model_path)
            except ImportError as exc:
                raise ImportError(
                    "sentencepiece is required: pip install sentencepiece"
                ) from exc
        elif self.backend == "hf_tokenizers":
            try:
                from tokenizers import Tokenizer
                self._hf = Tokenizer.from_file(str(self.model_path))
                logger.info("HF tokenizer loaded: %s", self.model_path)
            except ImportError as exc:
                raise ImportError(
                    "tokenizers is required: pip install tokenizers"
                ) from exc
        else:
            raise ValueError(f"Unknown tokenizer backend: {self.backend!r}")

    @property
    def vocab_size(self) -> int:
        if self._sp is not None:
            return self._sp.GetPieceSize()
        if self._hf is not None:
            return self._hf.get_vocab_size()
        raise RuntimeError("No tokenizer loaded.")

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token ids."""
        if self._sp is not None:
            return self._sp.EncodeAsIds(text)
        if self._hf is not None:
            return self._hf.encode(text).ids
        raise RuntimeError("No tokenizer loaded.")

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids to a string."""
        if self._sp is not None:
            return self._sp.DecodeIds(ids)
        if self._hf is not None:
            return self._hf.decode(ids)
        raise RuntimeError("No tokenizer loaded.")

    def as_callable(self) -> Callable[[str], list[int]]:
        """Return the encode method as a plain callable."""
        return self.encode


def train_sentencepiece(
    input_files: list[str],
    model_prefix: str,
    vocab_size: int = 32_000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
) -> None:
    """Train a SentencePiece tokenizer on *input_files*.

    Args:
        input_files: List of text file paths for training.
        model_prefix: Output prefix; produces ``<prefix>.model`` and ``<prefix>.vocab``.
        vocab_size: Target vocabulary size.
        character_coverage: Coverage of characters (0.9995 works well for Latin scripts).
        model_type: ``"bpe"``, ``"unigram"``, ``"word"``, or ``"char"``.
    """
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required: pip install sentencepiece") from exc

    input_str = ",".join(input_files)
    spm.SentencePieceTrainer.Train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )
    logger.info("SentencePiece tokenizer trained → %s.model", model_prefix)
