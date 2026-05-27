"""
model_core/data/preprocessing.py — Tokenisation, sequence packing, and batching.

Responsibilities:
  - Tokenise raw text strings using a tokenizer callable.
  - Optionally pack multiple short sequences into fixed-length chunks
    (greedy bin-packing, as in PaLM / LLaMA training).
  - Yield token ID tensors suitable for the training loop.

Provenance:
  Sequence-packing approach inspired by the PaLM paper (Chowdhery et al., 2022)
  and the packed_dataset implementation in lit-gpt (Lightning AI, Apache-2).
  See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

import torch

logger = logging.getLogger(__name__)


def tokenize_and_pack(
    text_iter: Iterator[str],
    tokenizer_fn: Callable[[str], list[int]],
    max_seq_len: int,
    pack: bool = True,
    eos_id: int = 2,
) -> Iterator[torch.Tensor]:
    """Tokenise and optionally pack text into fixed-length token sequences.

    Args:
        text_iter: Iterator yielding raw text strings.
        tokenizer_fn: Callable that maps a string to a list of integer token ids.
        max_seq_len: Target sequence length.
        pack: If True, use greedy bin-packing across documents. If False,
              truncate or pad each document independently.
        eos_id: EOS token id appended between packed documents.

    Yields:
        1-D :class:`torch.Tensor` of token ids with length ``max_seq_len``.
    """
    if pack:
        yield from _packed(text_iter, tokenizer_fn, max_seq_len, eos_id)
    else:
        for text in text_iter:
            ids = tokenizer_fn(text)[:max_seq_len]
            yield torch.tensor(ids, dtype=torch.long)


def _packed(
    text_iter: Iterator[str],
    tokenizer_fn: Callable[[str], list[int]],
    max_seq_len: int,
    eos_id: int,
) -> Iterator[torch.Tensor]:
    """Greedy packing: accumulate tokens into max_seq_len chunks."""
    buf: list[int] = []
    for text in text_iter:
        ids = tokenizer_fn(text) + [eos_id]
        buf.extend(ids)
        while len(buf) >= max_seq_len:
            yield torch.tensor(buf[:max_seq_len], dtype=torch.long)
            buf = buf[max_seq_len:]
    # Yield final partial chunk if non-empty
    if buf:
        yield torch.tensor(buf, dtype=torch.long)
