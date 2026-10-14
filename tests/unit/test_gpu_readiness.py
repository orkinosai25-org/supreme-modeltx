from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from supreme_modeltx.foundation import gpu_diagnostics
from supreme_modeltx.model_core.config.schema import SMTXConfig
from supreme_modeltx.model_core.training import trainer


def _cfg(tmp_path):
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"approved-tokenizer")
    source = tmp_path / "train.jsonl"
    source.write_text('{"text": "training sample"}\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": [{"name": "train", "path": str(source)}]}))
    return SMTXConfig.model_validate({
        "model": {"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 1,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64},
        "tokenizer": {"model_path": str(tokenizer)},
        "data": {"manifest_path": str(manifest), "max_seq_len": 64},
        "training": {"batch_size": 1, "max_steps": 2,
                     "precision": {"enabled": False, "dtype": "float16"},
                     "checkpoint": {"save_dir": str(tmp_path / "run/checkpoints")}},
    })


@pytest.mark.parametrize("env", [{"SMTX_REQUIRE_GPU": "1"}, {"SMTX_DEVICE": "cuda"},
                                 {"SUPREME_MODELTX_DEVICE": "cuda"}])
def test_gpu_designation_never_falls_back(tmp_path, monkeypatch, env):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(trainer, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    report = trainer.preflight_validate(cfg)
    assert not report["ok"]
    assert any("CPU fallback is forbidden" in error for error in report["errors"])
    with pytest.raises(ValueError, match="CPU fallback is forbidden"):
        trainer.train(cfg, dry_run=True)


@pytest.mark.parametrize("fsdp,world_size", [(True, "1"), (False, "2")])
def test_unvalidated_distributed_modes_rejected(tmp_path, monkeypatch, fsdp, world_size):
    cfg = _cfg(tmp_path)
    cfg.training.distributed.fsdp_enabled = fsdp
    monkeypatch.setenv("WORLD_SIZE", world_size)
    assert not trainer.preflight_validate(cfg)["ok"]
    with pytest.raises(ValueError, match="unsupported until validated"):
        trainer.train(cfg, dry_run=True)


def test_disabled_precision_execution_and_reporting(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(trainer, "get_device", lambda: torch.device("cpu"))
    observed_autocast, observed_scaler = [], []
    autocast = trainer.get_autocast_context

    def record_autocast(dtype, device_type):
        observed_autocast.append(dtype)
        return autocast(dtype, device_type)

    def record_scaler(dtype, device_type):
        observed_scaler.append(dtype)
        return None

    monkeypatch.setattr(trainer, "get_autocast_context", record_autocast)
    monkeypatch.setattr(trainer, "get_grad_scaler", record_scaler)
    cfg.data.manifest_path = None
    trainer.train(cfg, dry_run=True)
    assert observed_autocast == ["float32", "float32"]
    assert observed_scaler == ["float32"]
    precision = trainer.preflight_validate(cfg)["precision"]
    assert precision["dtype"] == "float32"
    assert precision["configured_dtype"] == "float16"
    trainer._write_run_summary(
        cfg, trainer._resolve_run_artifact_dir(cfg), device=torch.device("cpu"),
        started_at_utc="start", ended_at_utc="end", validation_history=[],
        checkpoint_paths=[], sample_artifact_paths=[],
    )
    summary = json.loads((tmp_path / "run/run_artifacts/training_summary.json").read_text())
    assert summary["precision"]["dtype"] == "float32"
    assert summary["precision"]["enabled"] is False
    assert summary["artifact_identity"]["tokenizer_sha256"]


def test_content_identity_detects_tokenizer_and_dataset_changes(tmp_path):
    cfg = _cfg(tmp_path)
    original = trainer._collect_artifact_identity(cfg)
    tokenizer = cfg.tokenizer.model_path
    Path(tokenizer).write_bytes(b"retrained-at-same-path")
    changed = trainer._collect_artifact_identity(cfg)
    assert changed["tokenizer_sha256"] != original["tokenizer_sha256"]
    (tmp_path / "train.jsonl").write_text('{"text": "changed"}\n')
    assert trainer._collect_artifact_identity(cfg)["dataset_sha256"] != original["dataset_sha256"]


def test_identity_normalizes_output_paths_not_settings(tmp_path):
    cfg = _cfg(tmp_path)
    original = trainer._collect_artifact_identity(cfg)
    cfg.training.checkpoint.save_dir = str(tmp_path / "other/checkpoints")
    moved = trainer._collect_artifact_identity(cfg)
    assert moved["settings_sha256"] == original["settings_sha256"]
    assert moved["config_sha256"] != original["config_sha256"]
    cfg.training.batch_size += 1
    assert trainer._collect_artifact_identity(cfg)["settings_sha256"] != original["settings_sha256"]


def test_synthetic_device_rng_is_not_an_identical_dataset_artifact(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.data.manifest_path = None
    identity = trainer._collect_artifact_identity(cfg)
    assert identity["dataset_sha256"] is None
    summary = {"artifact_identity": identity}
    assert not trainer.compare_gpu_run(summary, summary, {}, {})["comparable"]


def test_missing_manifest_identity_remains_unknown_without_breaking_summary(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.data.manifest_path = str(tmp_path / "unavailable-manifest.json")
    identity = trainer._collect_artifact_identity(cfg)
    assert identity["dataset_sha256"] is None
    assert identity["manifest_sha256"] is None


def _summary_and_benchmark(tmp_path):
    cfg = _cfg(tmp_path)
    summary = {"artifact_identity": trainer._collect_artifact_identity(cfg),
               "best_checkpoint_path": "checkpoint.pt",
               "validation_history": [{"val_loss": 2.0, "perplexity": 7.0}]}
    benchmark = {"local_checkpoints": [{"checkpoint_path": "checkpoint.pt",
                                      "metrics": {"overall_score": 0.5, "code_score": 0.3}}]}
    return summary, benchmark


def test_comparison_reads_actual_schemas_and_missing_metrics(tmp_path):
    summary, benchmark = _summary_and_benchmark(tmp_path)
    cpu = copy.deepcopy(summary)
    cpu["validation_history"][0]["val_loss"] = 3.0
    result = trainer.compare_gpu_run(summary, cpu, benchmark, benchmark)
    assert result["comparable"]
    assert result["gpu_vs_cpu"]["val_loss_delta"] == -1.0
    assert result["gpu_vs_cpu"]["benchmark_overall_delta"] == 0.0
    assert result["gpu_vs_cpu"]["benchmark_reasoning_delta"] is None
    assert result["gpu_best_checkpoint"]["checkpoint"] == "checkpoint.pt"


@pytest.mark.parametrize("key", ["tokenizer_sha256", "dataset_sha256", "model_sha256",
                                "settings_sha256", "benchmark_eval_sha256", "code_sha256"])
def test_nonidentical_artifacts_have_no_hardware_deltas(tmp_path, key):
    summary, benchmark = _summary_and_benchmark(tmp_path)
    cpu = copy.deepcopy(summary)
    cpu["artifact_identity"][key] = "different"
    result = trainer.compare_gpu_run(summary, cpu, benchmark, benchmark)
    assert not result["comparable"]
    assert all(value is None for value in result["gpu_vs_cpu"].values())


def test_historical_missing_identity_and_nonfinite_metrics(tmp_path):
    summary, benchmark = _summary_and_benchmark(tmp_path)
    result = trainer.compare_gpu_run(summary, {}, benchmark, {})
    assert not result["comparable"]
    assert all(value is None for value in result["gpu_vs_cpu"].values())
    summary["validation_history"][0]["val_loss"] = float("nan")
    benchmark["local_checkpoints"][0]["metrics"]["overall_score"] = float("inf")
    result = trainer.compare_gpu_run(summary, summary, benchmark, benchmark)
    assert result["gpu_vs_cpu"]["val_loss_delta"] is None
    assert result["gpu_vs_cpu"]["benchmark_overall_delta"] is None


def test_cpu_diagnostics_skip_cuda_operations(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gpu_diagnostics, "_numerical_check",
                        lambda dtype: pytest.fail("CPU diagnostics must not touch CUDA"))
    report = gpu_diagnostics.collect_gpu_diagnostics()
    assert report["cuda_available"] is False
    assert report["numerical_checks"] == {"cuda_operation": None, "fp16": None, "bf16": None}
    assert report["compute_capability"] is None
    assert report["readiness_status"] == "candidate_not_gpu_validated"


def test_cuda_metadata_and_numerical_checks(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(name="mock GPU", total_memory=24 * 1024**3,
                                                  major=8, minor=0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(gpu_diagnostics, "_driver_version", lambda: "mock-driver")
    dtypes = []

    def check(dtype):
        dtypes.append(dtype)
        return {"passed": True, "max_abs_error": 0.0}

    monkeypatch.setattr(gpu_diagnostics, "_numerical_check", check)
    report = gpu_diagnostics.collect_gpu_diagnostics()
    assert dtypes == [torch.float32, torch.float16, torch.bfloat16]
    assert report["driver_version"] == "mock-driver"
    assert report["compute_capability"] == "8.0"
    dtypes.clear()
    assert gpu_diagnostics.collect_gpu_diagnostics(run_cuda_checks=False)["numerical_checks"]["cuda_operation"] is None
    assert dtypes == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_numerical_probe_synchronizes_and_checks_values_on_cpu_standin(monkeypatch, dtype):
    # Exercise the probe mechanics without requiring or claiming GPU validation.
    original_to = torch.Tensor.to
    synchronized = []

    def cpu_standin(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_standin)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(True))
    result = gpu_diagnostics._numerical_check(dtype)
    assert synchronized == [True]
    assert result["passed"]
    assert result["max_abs_error"] >= 0.0


def test_numerical_probe_failure_is_reported(monkeypatch):
    def failing_to(*args, **kwargs):
        raise RuntimeError("mock CUDA failure")

    monkeypatch.setattr(torch.Tensor, "to", failing_to)
    result = gpu_diagnostics._numerical_check(torch.float16)
    assert result == {"passed": False, "error": "mock CUDA failure"}


def test_require_cuda_cli_fails_safely_without_gpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["gpu_diagnostics", "--require-cuda"])
    with pytest.raises(SystemExit) as exc:
        gpu_diagnostics.main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["cuda_available"] is False
