from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "data" / "processed" / "t_dev_6l_gpu_corpus_v1"
TRAIN_DIR = CORPUS_ROOT / "train"
VAL_DIR = CORPUS_ROOT / "validation"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "t_dev_6l_gpu_corpus_v1_first_subset.yaml"

CODE_PATTERNS = ["data/raw/code_samples/*.txt"]
REASONING_JSONL = [
    ("data/raw/reasoning/reasoning.jsonl", "question", "reasoning", "answer"),
    ("data/raw/instructions/instructions.jsonl", "instruction", "response"),
    ("data/raw/qa_pairs/qa_pairs.jsonl", "question", "answer"),
    ("data/raw/conversations/conversations.jsonl", "conversation"),
]
DOC_PATTERNS = [
    "data/raw/wiki_style/*.txt",
    "data/raw/general_text/*.txt",
]


def _iter_txt(patterns: list[str]) -> Iterable[str]:
    for pattern in patterns:
        for path in sorted(REPO_ROOT.glob(pattern)):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text:
                yield text


def _iter_reasoning_records() -> Iterable[str]:
    for spec in REASONING_JSONL:
        path = REPO_ROOT / spec[0]
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if len(spec) == 2 and spec[1] == "conversation":
                    blocks: list[str] = []
                    for turn in row.get("conversation", []):
                        role = str(turn.get("role", "")).strip().title()
                        content = str(turn.get("content", "")).strip()
                        if role and content:
                            blocks.append(f"{role}: {content}")
                    if blocks:
                        yield "\n".join(blocks)
                    continue
                parts = [str(row.get(field, "")).strip() for field in spec[1:] if str(row.get(field, "")).strip()]
                if parts:
                    yield "\n\n".join(parts)


def _normalize_text(text: str, *, preserve_lines: bool) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if preserve_lines:
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = " ".join(text.split())
    return text.strip()


def _fingerprint(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = " ".join(lowered.split())
    return lowered


def _preprocess(records: Iterable[str], *, preserve_lines: bool, min_chars: int = 40) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in records:
        text = _normalize_text(raw, preserve_lines=preserve_lines)
        if len(text) < min_chars:
            continue
        if "table of contents" in text.lower():
            continue
        fp = _fingerprint(text)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        output.append(text)
    return output


def _hash_bucket(text: str, *, validation_ratio: float = 0.2) -> str:
    h = hashlib.sha256(_fingerprint(text).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "validation" if bucket < int(validation_ratio * 100) else "train"


def _split(records: list[str]) -> tuple[list[str], list[str]]:
    train: list[str] = []
    validation: list[str] = []
    for item in records:
        if _hash_bucket(item) == "validation":
            validation.append(item)
        else:
            train.append(item)

    if records and not validation:
        validation.append(train.pop())
    if records and not train:
        train.append(validation.pop())
    return train, validation


def _write_jsonl(path: Path, records: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in records:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _emit_manifest() -> None:
    manifest = {
        "version": "1",
        "name": "t_dev_6l_gpu_corpus_v1_first_subset",
        "description": "First approved and materialized subset for T-Dev-6L GPU corpus v1.",
        "manifest_type": "training_run",
        "status": "materialized",
        "sources": [
            {
                "name": "code-train",
                "backend": "jsonl",
                "path": "data/processed/t_dev_6l_gpu_corpus_v1/train/code.jsonl",
                "split": "train",
                "text_field": "text",
                "weight": 0.55,
            },
            {
                "name": "code-validation",
                "backend": "jsonl",
                "path": "data/processed/t_dev_6l_gpu_corpus_v1/validation/code.jsonl",
                "split": "validation",
                "text_field": "text",
            },
            {
                "name": "reasoning-train",
                "backend": "jsonl",
                "path": "data/processed/t_dev_6l_gpu_corpus_v1/train/reasoning_instructions.jsonl",
                "split": "train",
                "text_field": "text",
                "weight": 0.25,
            },
            {
                "name": "reasoning-validation",
                "backend": "jsonl",
                "path": "data/processed/t_dev_6l_gpu_corpus_v1/validation/reasoning_instructions.jsonl",
                "split": "validation",
                "text_field": "text",
            },
            {
                "name": "docs-train",
                "backend": "jsonl",
                "path": "data/processed/t_dev_6l_gpu_corpus_v1/train/technical_docs.jsonl",
                "split": "train",
                "text_field": "text",
                "weight": 0.15,
            },
            {
                "name": "docs-validation",
                "backend": "jsonl",
                "path": "data/processed/t_dev_6l_gpu_corpus_v1/validation/technical_docs.jsonl",
                "split": "validation",
                "text_field": "text",
            },
            {
                "name": "synthetic-bridge-train",
                "backend": "jsonl",
                "path": "data/processed/t_dev_6l_gpu_corpus_v1/train/synthetic_bridge.jsonl",
                "split": "train",
                "text_field": "text",
                "weight": 0.05,
            },
        ],
    }
    MANIFEST_PATH.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def materialize() -> None:
    code_records = _preprocess(_iter_txt(CODE_PATTERNS), preserve_lines=True, min_chars=60)
    reasoning_records = _preprocess(_iter_reasoning_records(), preserve_lines=False, min_chars=60)
    docs_records = _preprocess(_iter_txt(DOC_PATTERNS), preserve_lines=False, min_chars=120)
    synthetic_records = [
        "Task: Produce a patch-style summary. Input: rename variable old_count to sample_count in training config. Output: apply a minimal edit and keep behavior unchanged.",
        "Task: Format a deterministic tool-call shell. Input: run validation split only. Output: python -m supreme_modeltx.model_core.training.trainer --config configs/real_training/t_dev_6l_expanded_run.json --preflight",
        "Task: Create an instruction scaffold. Input: explain manifest source provenance. Output: include source, license, and preprocessing rationale in concise bullet points.",
    ]
    synthetic_records = _preprocess(synthetic_records, preserve_lines=False, min_chars=60)

    code_train, code_val = _split(code_records)
    reasoning_train, reasoning_val = _split(reasoning_records)
    docs_train, docs_val = _split(docs_records)

    _write_jsonl(TRAIN_DIR / "code.jsonl", code_train)
    _write_jsonl(VAL_DIR / "code.jsonl", code_val)
    _write_jsonl(TRAIN_DIR / "reasoning_instructions.jsonl", reasoning_train)
    _write_jsonl(VAL_DIR / "reasoning_instructions.jsonl", reasoning_val)
    _write_jsonl(TRAIN_DIR / "technical_docs.jsonl", docs_train)
    _write_jsonl(VAL_DIR / "technical_docs.jsonl", docs_val)
    _write_jsonl(TRAIN_DIR / "synthetic_bridge.jsonl", synthetic_records)
    _emit_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the first approved T-Dev-6L GPU corpus subset.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    materialize()


if __name__ == "__main__":
    main()
