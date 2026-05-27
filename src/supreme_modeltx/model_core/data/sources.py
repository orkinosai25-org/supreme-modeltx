"""
model_core/data/sources.py — Data source adapters.

Provides a unified iterator interface over different data backends:
  - JSONL files (e.g. HF-style instruction/conversation datasets)
  - Plain text files / directories
  - Parquet files
  - HuggingFace Datasets (hf_dataset) — optional import

Each adapter yields raw text strings.  Downstream preprocessing
(tokenisation, packing) happens in preprocessing.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from supreme_modeltx.model_core.data.manifest import DataSource

logger = logging.getLogger(__name__)


def iter_source(source: DataSource) -> Iterator[str]:
    """Yield raw text strings from a :class:`DataSource`."""
    backend = source.backend
    if backend == "jsonl":
        yield from _iter_jsonl(source)
    elif backend == "text":
        yield from _iter_text(source)
    elif backend == "parquet":
        yield from _iter_parquet(source)
    elif backend == "hf_dataset":
        yield from _iter_hf_dataset(source)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def _iter_jsonl(source: DataSource) -> Iterator[str]:
    path = Path(source.path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(source.text_field, "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSON line in %s", f)


def _iter_text(source: DataSource) -> Iterator[str]:
    path = Path(source.path)
    files: Iterable[Path]
    if path.is_dir():
        files = sorted(path.rglob("*.txt"))
    else:
        files = [path]
    for f in files:
        yield f.read_text(encoding="utf-8", errors="replace")


def _iter_parquet(source: DataSource) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required for parquet sources: pip install pyarrow") from exc
    path = Path(source.path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    for f in files:
        table = pq.read_table(f)
        field = source.text_field
        if field not in table.column_names:
            logger.warning("Field %r not found in %s; skipping.", field, f)
            continue
        for val in table[field].to_pylist():
            if val:
                yield str(val)


def _iter_hf_dataset(source: DataSource) -> Iterator[str]:
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise ImportError(
            "datasets is required for hf_dataset sources: pip install datasets"
        ) from exc
    ds = hf_load(source.hf_name, split=source.hf_split)
    for row in ds:
        text = row.get(source.text_field, "")
        if text:
            yield text
