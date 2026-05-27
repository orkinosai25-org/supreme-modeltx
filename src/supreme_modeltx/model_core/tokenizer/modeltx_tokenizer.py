"""ModelTX tokenizer wrapper.

Wraps HuggingFace ``tokenizers`` / ``sentencepiece`` to provide a
unified interface for the supreme-modeltx engine. Training a custom
BPE vocabulary is supported via :meth:`train_from_iterator`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_SPECIAL_TOKENS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}


class ModelTXTokenizer:
    """Byte-Pair Encoding tokenizer for the T-Series model family.

    On first use, either load a pre-trained tokenizer from ``vocab_path``
    or call :meth:`train_from_iterator` to train from scratch.
    """

    def __init__(self, vocab_path: str | None = None) -> None:
        self._tokenizer = None
        if vocab_path is not None:
            self._load(vocab_path)

    def _load(self, vocab_path: str) -> None:
        from tokenizers import Tokenizer  # type: ignore[import]

        self._tokenizer = Tokenizer.from_file(vocab_path)
        logger.info("Loaded tokenizer from %s", vocab_path)

    def train_from_iterator(
        self,
        texts: Iterable[str],
        vocab_size: int = 32000,
        save_path: str | None = None,
    ) -> None:
        """Train a BPE tokenizer from an iterable of text strings."""
        from tokenizers import Tokenizer  # type: ignore[import]
        from tokenizers.models import BPE  # type: ignore[import]
        from tokenizers.trainers import BpeTrainer  # type: ignore[import]
        from tokenizers.pre_tokenizers import ByteLevel  # type: ignore[import]

        tokenizer = Tokenizer(BPE(unk_token=_SPECIAL_TOKENS["unk_token"]))
        tokenizer.pre_tokenizer = ByteLevel()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(_SPECIAL_TOKENS.values()),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer
        logger.info("Trained BPE tokenizer (vocab_size=%d)", vocab_size)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(save_path)
            logger.info("Saved tokenizer to %s", save_path)

    def encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialised. Call train_from_iterator or pass vocab_path.")
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialised.")
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialised.")
        return self._tokenizer.get_vocab_size()

    def bos_id(self) -> int:
        return self._tokenizer.token_to_id(_SPECIAL_TOKENS["bos_token"])

    def eos_id(self) -> int:
        return self._tokenizer.token_to_id(_SPECIAL_TOKENS["eos_token"])
