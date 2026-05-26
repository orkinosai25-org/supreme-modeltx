"""
dataset_pipeline.py — SMTX Dataset Ingestion Pipeline

Loads, cleans, and tokenizes raw text corpora for T‑101 pre‑training.

Usage:
    python training/dataset_pipeline.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --tokenizer_path tmodels/t101
"""

import argparse
import json
import os
from pathlib import Path
from typing import Generator, List

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMTX dataset ingestion pipeline.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw text files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write processed JSONL files.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tmodels/t101",
        help="Path to the tokenizer.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--min_chars", type=int, default=50, help="Minimum characters per document.")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def iter_text_files(input_dir: str) -> Generator[str, None, None]:
    """Yield text content from all .txt and .jsonl files in input_dir."""
    for path in Path(input_dir).rglob("*"):
        if path.suffix == ".txt":
            yield path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content") or ""
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue


def clean_text(text: str) -> str:
    """Basic text cleaning: strip excess whitespace."""
    return " ".join(text.split())


def chunk_tokens(token_ids: List[int], chunk_size: int) -> Generator[List[int], None, None]:
    """Split a long token sequence into fixed‑size chunks."""
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i : i + chunk_size]
        if len(chunk) == chunk_size:
            yield chunk


def process(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = Path(args.output_dir) / "train.jsonl"
    val_path = Path(args.output_dir) / "val.jsonl"

    train_count = 0
    val_count = 0
    val_fraction = 0.01  # 1% goes to validation

    with open(train_path, "w", encoding="utf-8") as train_f, \
         open(val_path, "w", encoding="utf-8") as val_f:

        for raw_text in iter_text_files(args.input_dir):
            text = clean_text(raw_text)
            if len(text) < args.min_chars:
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            for chunk in chunk_tokens(token_ids, args.max_seq_length):
                record = json.dumps({"input_ids": chunk, "text": tokenizer.decode(chunk)})
                # Divert 1% to validation using deterministic modulo
                if (train_count + val_count) % 100 < val_fraction * 100:
                    val_f.write(record + "\n")
                    val_count += 1
                else:
                    train_f.write(record + "\n")
                    train_count += 1

    print(f"Pipeline complete. Train: {train_count} samples, Val: {val_count} samples.")
    print(f"Output written to {args.output_dir}")


def main() -> None:
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
