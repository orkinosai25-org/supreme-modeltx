"""CLI entrypoint for training versioned tokenizer artifacts."""

from __future__ import annotations

import argparse

from supreme_modeltx.model_core.tokenizer.workflow import train_versioned_sentencepiece


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a versioned SentencePiece tokenizer.")
    parser.add_argument(
        "--input-path",
        action="append",
        default=[],
        help="Local text file or directory (repeatable).",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional data manifest (JSON/YAML) for source-based corpus loading.",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default="artifacts/tokenizers",
        help="Root directory for tokenizer artifacts.",
    )
    parser.add_argument("--model-variant", type=str, default="t-dev-6l")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "word", "char"],
    )
    args = parser.parse_args()

    artifacts = train_versioned_sentencepiece(
        input_paths=args.input_path,
        manifest_path=args.manifest_path,
        artifact_root=args.artifact_root,
        model_variant=args.model_variant,
        version=args.version,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
    )
    print(f"Tokenizer trained: {artifacts.model_path}")
    print(f"Vocab written: {artifacts.vocab_path}")
    print(f"Metadata written: {artifacts.metadata_path}")


if __name__ == "__main__":
    main()
