from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){8,14}\b"),
}

SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} in {path} must contain a JSON object.")
            records.append(payload)
    return records


def validate_jsonl_dataset(
    path: str | Path,
    *,
    required_fields: list[str],
    scan_pii: bool = False,
    scan_secrets: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    records = load_jsonl(path)
    issues: list[str] = []
    duplicates: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    fingerprint_to_line: dict[str, int] = {}
    sources = Counter()
    licenses = Counter()

    for index, record in enumerate(records, start=1):
        missing = [field for field in required_fields if not str(record.get(field, "")).strip()]
        if missing:
            issues.append(f"line {index}: missing required fields {missing}")

        if "source" in record:
            sources[str(record["source"])] += 1
        if "license" in record:
            licenses[str(record["license"])] += 1

        fingerprint_parts = [str(record.get(field, "")).strip().lower() for field in required_fields]
        fingerprint = "|".join(fingerprint_parts)
        if fingerprint in fingerprint_to_line:
            duplicates.append({"first_line": fingerprint_to_line[fingerprint], "duplicate_line": index})
        else:
            fingerprint_to_line[fingerprint] = index

        if not (scan_pii or scan_secrets):
            continue

        record_text = json.dumps(record, sort_keys=True)
        if scan_pii:
            for label, pattern in PII_PATTERNS.items():
                if pattern.search(record_text):
                    findings.append({"line": index, "type": "pii", "label": label})
        if scan_secrets:
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(record_text):
                    findings.append({"line": index, "type": "secret", "label": label})

    return {
        "path": str(path),
        "record_count": len(records),
        "valid": not issues and not duplicates and not findings,
        "issues": issues,
        "duplicates": duplicates,
        "findings": findings,
        "metadata_summary": {
            "sources": dict(sources),
            "licenses": dict(licenses),
        },
    }


def report_summary_for_console(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": report["path"],
        "record_count": report["record_count"],
        "valid": report["valid"],
        "issue_count": len(report["issues"]),
        "duplicate_count": len(report["duplicates"]),
        "finding_count": len(report["findings"]),
    }


def split_jsonl_dataset(
    path: str | Path,
    *,
    output_dir: str | Path,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int = 7,
) -> dict[str, Any]:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    records = load_jsonl(path)
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = len(shuffled)
    train_end = int(total_count * train_ratio)
    validation_end = train_end + int(total_count * validation_ratio)

    partitions = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }

    outputs: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split_name, split_records in partitions.items():
        split_path = output_dir / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for record in split_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[split_name] = str(split_path)
        counts[split_name] = len(split_records)

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_path": str(Path(path)),
                "seed": seed,
                "ratios": {
                    "train": train_ratio,
                    "validation": validation_ratio,
                    "test": test_ratio,
                },
                "counts": counts,
                "outputs": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "manifest_path": str(manifest_path),
        "counts": counts,
        "outputs": outputs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset governance utilities for Supreme Model T-X.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a JSONL dataset.")
    validate_parser.add_argument("path")
    validate_parser.add_argument(
        "--required-field",
        action="append",
        default=["prompt", "response", "source", "license"],
    )
    validate_parser.add_argument("--scan-pii", action="store_true")
    validate_parser.add_argument("--scan-secrets", action="store_true")

    split_parser = subparsers.add_parser("split", help="Create train/validation/test JSONL splits.")
    split_parser.add_argument("path")
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--validation-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=7)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        report = validate_jsonl_dataset(
            args.path,
            required_fields=args.required_field,
            scan_pii=args.scan_pii,
            scan_secrets=args.scan_secrets,
        )
        print(json.dumps(report_summary_for_console(report), indent=2))
        return

    report = split_jsonl_dataset(
        args.path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
