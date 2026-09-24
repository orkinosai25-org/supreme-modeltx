from __future__ import annotations

import json
import subprocess

import pytest

from supreme_modeltx.foundation import gpu_diagnostics as diagnostics_module


class _Props:
    def __init__(self, name: str, total_memory: int, major: int = 8, minor: int = 0) -> None:
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor


def test_collect_gpu_diagnostics_reports_cpu_only_runtime(monkeypatch):
    monkeypatch.setattr(diagnostics_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diagnostics_module, "_collect_nvidia_runtime", lambda: {"available": False})

    report = diagnostics_module.collect_gpu_diagnostics()

    assert report["cuda_available"] is False
    assert report["cuda_device_count"] == 0
    assert report["devices"] == []
    assert report["mixed_precision"]["fp16_supported"] is False


def test_collect_gpu_diagnostics_reports_cuda_details(monkeypatch):
    monkeypatch.setattr(diagnostics_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diagnostics_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        diagnostics_module.torch.cuda,
        "get_device_properties",
        lambda index: _Props("Mock GPU", 24 * 1024**3),
    )
    monkeypatch.setattr(diagnostics_module.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        diagnostics_module,
        "_collect_nvidia_runtime",
        lambda: {"available": True, "path": "/usr/bin/nvidia-smi", "gpus": [{"index": 0}]},
    )

    report = diagnostics_module.collect_gpu_diagnostics()

    assert report["cuda_available"] is True
    assert report["cuda_device_count"] == 1
    assert report["devices"][0]["name"] == "Mock GPU"
    assert report["devices"][0]["bf16_supported"] is True
    assert report["runtime"]["nvidia_smi"]["available"] is True


def test_collect_nvidia_runtime_handles_commas_in_gpu_names(monkeypatch):
    monkeypatch.setattr(diagnostics_module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        diagnostics_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='0,"Mock, GPU",550.54.15,24564\n',
            stderr="",
        ),
    )

    report = diagnostics_module._collect_nvidia_runtime()

    assert report["available"] is True
    assert report["gpus"][0]["name"] == "Mock, GPU"


def test_gpu_diagnostics_main_requires_cuda(monkeypatch, capsys):
    monkeypatch.setattr(
        diagnostics_module,
        "collect_gpu_diagnostics",
        lambda: {
            "cuda_available": False,
            "cuda_device_count": 0,
            "devices": [],
            "torch_version": "2.x",
            "torch_compiled_cuda_version": None,
            "runtime": {"nvidia_smi": {"available": False}},
            "mixed_precision": {"fp16_supported": False, "bf16_supported": False},
        },
    )

    with pytest.raises(SystemExit, match="1"):
        diagnostics_module.main(["--require-cuda"])

    report = json.loads(capsys.readouterr().out)
    assert "strict_errors" in report
    assert "CUDA was required but is not available" in report["strict_errors"][0]


def test_gpu_diagnostics_main_surfaces_preflight_failures(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "gpu.yaml"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                "  device: cuda",
                "  mixed_precision: bf16",
                f"  output_dir: {tmp_path / 'run'}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        diagnostics_module,
        "collect_gpu_diagnostics",
        lambda: {
            "cuda_available": True,
            "cuda_device_count": 1,
            "devices": [{"index": 0, "name": "Mock GPU"}],
            "torch_version": "2.x",
            "torch_compiled_cuda_version": "12.4",
            "runtime": {"nvidia_smi": {"available": True}},
            "mixed_precision": {"fp16_supported": True, "bf16_supported": False},
        },
    )
    monkeypatch.setattr(
        diagnostics_module,
        "build_training_preflight",
        lambda cfg: {"ok": False, "errors": ["mock preflight failure"]},
    )

    with pytest.raises(SystemExit, match="1"):
        diagnostics_module.main(["--config", str(config_path)])

    report = json.loads(capsys.readouterr().out)
    assert report["preflight"]["ok"] is False
    assert report["strict_errors"] == ["mock preflight failure"]


def test_gpu_diagnostics_main_reports_invalid_config(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("training:\n  cuda_device_index: nope\n", encoding="utf-8")
    monkeypatch.setattr(
        diagnostics_module,
        "collect_gpu_diagnostics",
        lambda: {
            "cuda_available": True,
            "cuda_device_count": 1,
            "devices": [{"index": 0, "name": "Mock GPU"}],
            "torch_version": "2.x",
            "torch_compiled_cuda_version": "12.4",
            "runtime": {"nvidia_smi": {"available": True}},
            "mixed_precision": {"fp16_supported": True, "bf16_supported": True},
        },
    )

    with pytest.raises(SystemExit, match="1"):
        diagnostics_module.main(["--config", str(config_path)])

    report = json.loads(capsys.readouterr().out)
    assert report["preflight"]["ok"] is False
    assert "Failed to load config" in report["strict_errors"][0]
