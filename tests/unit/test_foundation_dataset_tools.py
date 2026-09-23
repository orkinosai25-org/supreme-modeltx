import json

from supreme_modeltx.foundation.dataset_tools import split_jsonl_dataset, validate_jsonl_dataset


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_validate_jsonl_dataset_detects_required_fields_duplicates_and_findings(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    _write_rows(
        dataset_path,
        [
            {"prompt": "hello", "response": "world", "source": "synthetic", "license": "internal"},
            {"prompt": "hello", "response": "world", "source": "synthetic", "license": "internal"},
            {"prompt": "contact me", "response": "email me at test@example.com", "source": "", "license": "internal"},
        ],
    )

    report = validate_jsonl_dataset(
        dataset_path,
        required_fields=["prompt", "response", "source", "license"],
        scan_pii=True,
        scan_secrets=True,
    )

    assert report["record_count"] == 3
    assert report["duplicates"] == [{"first_line": 1, "duplicate_line": 2}]
    assert any("missing required fields" in issue for issue in report["issues"])
    assert any(finding["label"] == "email" for finding in report["findings"])


def test_split_jsonl_dataset_writes_manifest_and_splits(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"prompt": f"p{i}", "response": f"r{i}", "source": "synthetic", "license": "internal"}
        for i in range(10)
    ]
    _write_rows(dataset_path, rows)

    report = split_jsonl_dataset(
        dataset_path,
        output_dir=tmp_path / "splits",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        seed=11,
    )

    assert report["counts"] == {"train": 6, "validation": 2, "test": 2}
    assert (tmp_path / "splits" / "split_manifest.json").exists()
