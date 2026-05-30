"""
model_core/tokenizer/workflow.py — Tokenizer workflow boundary.

Provides a unified TokenizerWorkflow class that wraps either:
  - A trained SentencePiece model (.model file)
  - A HuggingFace tokenizer (via the `tokenizers` library)

This module defines the workflow boundary — callers should always
use TokenizerWorkflow rather than importing sentencepiece directly,
so the underlying implementation can be swapped without API changes.

SentencePiece training:
  smtx-tokenizer-train is a thin wrapper around the spm.SentencePieceTrainer,
  which should be trained on the sovereign pretraining corpus before
  any model training begins.

Provenance:
  SentencePiece: Kudo & Richardson, 2018 (Apache-2.0).
  HuggingFace tokenizers: HuggingFace (Apache-2.0).
  See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from supreme_modeltx.model_core.data.manifest import DataManifest
from supreme_modeltx.model_core.data.sources import iter_source

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenizerArtifacts:
    """Paths for one trained, versioned tokenizer artifact set."""

    artifact_dir: Path
    model_path: Path
    vocab_path: Path
    metadata_path: Path
    corpus_path: Path
    version: str


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

    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
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
        hard_vocab_limit=False,
    )
    logger.info("SentencePiece tokenizer trained → %s.model", model_prefix)


def _iter_text_inputs(input_paths: Iterable[str]) -> Iterable[str]:
    for path_str in input_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer input path not found: {path}")
        if path.is_dir():
            for text_file in sorted(path.rglob("*.txt")):
                yield text_file.read_text(encoding="utf-8", errors="replace")
        else:
            yield path.read_text(encoding="utf-8", errors="replace")


def _iter_manifest_inputs(manifest_path: str) -> Iterable[str]:
    manifest = DataManifest.from_file(manifest_path)
    for source in manifest.sources:
        yield from iter_source(source)


def train_versioned_sentencepiece(
    *,
    input_paths: list[str] | None = None,
    manifest_path: str | None = None,
    artifact_root: str | Path = "artifacts/tokenizers",
    model_variant: str = "t-dev-6l",
    version: str | None = None,
    vocab_size: int = 32_000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
) -> TokenizerArtifacts:
    """Train and save a versioned SentencePiece tokenizer artifact bundle."""
    inputs = input_paths or []
    if not inputs and not manifest_path:
        raise ValueError("Provide at least one tokenizer input source (input_paths or manifest_path).")

    resolved_version = version or datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    artifact_dir = Path(artifact_root) / model_variant / resolved_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = artifact_dir / "training_corpus.txt"
    source_count = 0
    with corpus_path.open("w", encoding="utf-8") as corpus_file:
        for text in _iter_text_inputs(inputs):
            text = text.strip()
            if text:
                corpus_file.write(text)
                corpus_file.write("\n")
                source_count += 1
        if manifest_path:
            for text in _iter_manifest_inputs(manifest_path):
                text = str(text).strip()
                if text:
                    corpus_file.write(text)
                    corpus_file.write("\n")
                    source_count += 1

    if source_count == 0:
        raise ValueError("Tokenizer corpus is empty after reading local inputs.")

    model_prefix = str(artifact_dir / "tokenizer")
    train_sentencepiece(
        input_files=[str(corpus_path)],
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
    )

    model_path = artifact_dir / "tokenizer.model"
    vocab_path = artifact_dir / "tokenizer.vocab"
    metadata_path = artifact_dir / "metadata.json"
    metadata = {
        "backend": "sentencepiece",
        "model_variant": model_variant,
        "version": resolved_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "corpus_path": str(corpus_path),
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "model_type": model_type,
        "input_paths": inputs,
        "manifest_path": manifest_path,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TokenizerArtifacts(
        artifact_dir=artifact_dir,
        model_path=model_path,
        vocab_path=vocab_path,
        metadata_path=metadata_path,
        corpus_path=corpus_path,
        version=resolved_version,
    )
