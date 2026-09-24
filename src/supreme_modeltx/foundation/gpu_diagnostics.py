from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from typing import Any

import torch

from supreme_modeltx.foundation.config import TrainingRunConfig
from supreme_modeltx.foundation.training import build_training_preflight


def _collect_nvidia_runtime() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False}
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}

    gpus = []
    for parts in csv.reader(result.stdout.splitlines()):
        parts = [part.strip() for part in parts]
        if len(parts) != 4:
            continue
        index, name, driver_version, memory_total_mb = parts
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver_version,
                "memory_total_mb": int(memory_total_mb),
            }
        )
    return {
        "available": True,
        "path": nvidia_smi,
        "gpus": gpus,
    }


def collect_gpu_diagnostics() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if cuda_available else False
    devices = []
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_vram_gb": round(props.total_memory / (1024**3), 2),
                "capability": f"{props.major}.{props.minor}",
                "fp16_supported": True,
                "bf16_supported": bf16_supported,
            }
        )

    return {
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "devices": devices,
        "torch_version": torch.__version__,
        "torch_compiled_cuda_version": torch.version.cuda,
        "runtime": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cudnn_enabled": bool(getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.enabled),
            "cudnn_version": torch.backends.cudnn.version() if getattr(torch.backends, "cudnn", None) else None,
            "nvidia_smi": _collect_nvidia_runtime(),
        },
        "mixed_precision": {
            "fp16_supported": cuda_available,
            "bf16_supported": bf16_supported,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report CUDA/GPU diagnostics for Supreme Model T-X foundation runs.")
    parser.add_argument("--config", help="Optional training config to validate alongside diagnostics.")
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero unless CUDA is available.")
    args = parser.parse_args(argv)

    diagnostics = collect_gpu_diagnostics()
    strict_errors: list[str] = []
    if args.require_cuda and not diagnostics["cuda_available"]:
        strict_errors.append(
            "CUDA was required but is not available. Install a CUDA-enabled PyTorch wheel for this host, "
            "confirm the NVIDIA driver is present, then re-run this command."
        )
    if args.config:
        diagnostics["preflight"] = build_training_preflight(TrainingRunConfig.from_file(args.config))
        if not diagnostics["preflight"]["ok"]:
            strict_errors.extend(diagnostics["preflight"]["errors"])
    if strict_errors:
        diagnostics["strict_errors"] = strict_errors

    print(json.dumps(diagnostics, indent=2))
    raise SystemExit(1 if strict_errors else 0)


if __name__ == "__main__":
    main()
