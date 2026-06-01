from __future__ import annotations

import json

import torch

from supreme_modeltx.model_core.config.schema import SMTXConfig
from supreme_modeltx.model_core.training import trainer as trainer_module
from supreme_modeltx.model_core.training.trainer import preflight_validate


def _write_manifest(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"text": "hello gpu"}) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                'version: "1"',
                "sources:",
                "  - name: train-a",
                "    backend: jsonl",
                f"    path: {train_file.as_posix()}",
                "    split: train",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


def _base_cfg(tmp_path, *, precision_dtype: str = "float32") -> SMTXConfig:
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_text("dummy-tokenizer", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path)
    return SMTXConfig.model_validate(
        {
            "tokenizer": {"model_path": str(tokenizer_path)},
            "data": {
                "manifest_path": str(manifest_path),
                "tokenizer_path": str(tokenizer_path),
            },
            "training": {
                "checkpoint": {"save_dir": str(tmp_path / "run" / "checkpoints")},
                "precision": {"enabled": precision_dtype != "float32", "dtype": precision_dtype},
                "max_steps": 2,
            },
        }
    )


def test_preflight_validate_success_reports_artifact_contract(tmp_path):
    cfg = _base_cfg(tmp_path)

    report = preflight_validate(cfg)

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["artifact_contract_paths"]["config_used"].endswith("run_artifacts/config_used.json")
    assert report["artifact_contract_paths"]["samples_dir"].endswith("run_artifacts/samples")


def test_preflight_validate_fails_for_missing_tokenizer(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    cfg = SMTXConfig.model_validate(
        {
            "tokenizer": {"model_path": str(tmp_path / "missing_tokenizer.model")},
            "data": {
                "manifest_path": str(manifest_path),
                "tokenizer_path": str(tmp_path / "missing_tokenizer.model"),
            },
            "training": {"checkpoint": {"save_dir": str(tmp_path / "run" / "checkpoints")}},
        }
    )

    report = preflight_validate(cfg)

    assert report["ok"] is False
    assert any("Tokenizer path does not exist" in message for message in report["errors"])


def test_preflight_validate_rejects_cpu_float16(tmp_path, monkeypatch):
    cfg = _base_cfg(tmp_path, precision_dtype="float16")
    monkeypatch.setattr(trainer_module, "get_device", lambda: torch.device("cpu"))

    report = preflight_validate(cfg)

    assert report["ok"] is False
    assert any("float16 precision is not supported on CPU" in message for message in report["errors"])
