"""Reproducible data pipeline for pre-training and fine-tuning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class DataPipeline:
    """Loads and tokenises text datasets for training.

    Supports:
    - Plain-text files (one document per line or multi-line)
    - JSONL files with a configurable ``text_field``
    - HuggingFace dataset identifiers (requires ``datasets`` library)
    """

    def __init__(
        self,
        source: str,
        tokenizer: object,
        max_seq_len: int = 2048,
        text_field: str = "text",
        seed: int = 42,
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field
        self.seed = seed

    def _iter_texts_from_file(self, path: Path) -> Iterator[str]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    yield obj.get(self.text_field, "")
        else:
            with path.open(encoding="utf-8") as f:
                yield f.read()

    def _iter_texts(self) -> Iterator[str]:
        path = Path(self.source)
        if path.exists():
            if path.is_dir():
                for file in sorted(path.rglob("*.txt")):
                    yield from self._iter_texts_from_file(file)
                for file in sorted(path.rglob("*.jsonl")):
                    yield from self._iter_texts_from_file(file)
            else:
                yield from self._iter_texts_from_file(path)
        else:
            try:
                from datasets import load_dataset  # type: ignore[import]

                ds = load_dataset(self.source, split="train")
                for example in ds:
                    yield example[self.text_field]
            except Exception as exc:
                logger.error("Failed to load dataset %r: %s", self.source, exc)
                raise

    def build_token_chunks(self) -> list[list[int]]:
        """Tokenise all texts and split into fixed-length chunks."""
        chunks: list[list[int]] = []
        buffer: list[int] = []
        for text in self._iter_texts():
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            while len(buffer) >= self.max_seq_len:
                chunks.append(buffer[: self.max_seq_len])
                buffer = buffer[self.max_seq_len :]
        if buffer:
            chunks.append(buffer)
        logger.info("Built %d token chunks from %r", len(chunks), self.source)
        return chunks
