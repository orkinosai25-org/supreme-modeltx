from __future__ import annotations

from pathlib import Path

from supreme_modeltx.model_core.data.manifest import DataManifest


def test_gpu_corpus_plan_manifest_validates_cleanly():
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "data" / "manifests" / "t_dev_6l_gpu_corpus_v1.yaml"

    manifest = DataManifest.from_file(manifest_path)

    assert manifest.manifest_type == "corpus_plan"
    assert manifest.status == "planned"
    assert manifest.validate_manifest(base_dir=repo_root) == []


def test_validate_manifest_reports_missing_corpus_plan_metadata(tmp_path):
    manifest = DataManifest.model_validate(
        {
            "version": "2",
            "name": "broken-plan",
            "manifest_type": "corpus_plan",
            "status": "planned",
            "corpus_version": "broken-plan-v1",
            "previous_baseline": "data/manifests/t_dev_6l_expanded_run.yaml",
            "split_rules": {"train": "train"},
            "preprocessing_expectations": ["deduplicate"],
            "benchmark_alignment": ["code tasks"],
            "changes_from_previous": ["more data"],
            "sources": [
                {
                    "name": "code-train",
                    "backend": "jsonl",
                    "path": "data/processed/code.jsonl",
                    "split": "train",
                }
            ],
        }
    )

    errors = manifest.validate_manifest(base_dir=tmp_path)

    assert "Corpus-plan manifest must define train and validation split_rules." in errors
    assert "Corpus-plan manifest must include train and validation sources." in errors
    assert "Corpus-plan source 'code-train' must define category." in errors
    assert "Corpus-plan source 'code-train' must define provenance." in errors
    assert "Corpus-plan source 'code-train' must define license." in errors
    assert "Corpus-plan source 'code-train' must define preprocessing." in errors


def test_validate_manifest_requires_existing_files_for_materialized_sources(tmp_path):
    manifest = DataManifest.model_validate(
        {
            "version": "1",
            "sources": [
                {
                    "name": "train-a",
                    "backend": "jsonl",
                    "path": "missing/train.jsonl",
                    "split": "train",
                }
            ],
        }
    )

    errors = manifest.validate_manifest(base_dir=tmp_path)

    assert len(errors) == 1
    assert "path does not exist" in errors[0]
    assert str(tmp_path / "missing" / "train.jsonl") in errors[0]
