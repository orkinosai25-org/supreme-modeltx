from __future__ import annotations

import argparse
import json
import subprocess

import torch


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _numerical_check(dtype: torch.dtype) -> dict[str, object]:
    try:
        reference = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
        expected = reference @ reference.T
        inputs = reference.to(device="cuda", dtype=dtype).requires_grad_()
        actual = inputs @ inputs.T
        actual.sum().backward()
        torch.cuda.synchronize()
        result = actual.detach().float().cpu()
        tolerance = 0.05 if dtype == torch.bfloat16 else 0.01
        passed = bool(
            torch.isfinite(result).all()
            and torch.isfinite(inputs.grad).all()
            and torch.allclose(result, expected, atol=tolerance, rtol=tolerance)
        )
        return {"passed": passed, "max_abs_error": (result - expected).abs().max().item()}
    except (RuntimeError, AssertionError) as exc:
        return {"passed": False, "error": str(exc)}


def collect_gpu_diagnostics(*, run_cuda_checks: bool = True) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        total_vram_gb = round(props.total_memory / (1024**3), 2)
        mixed_precision = {
            "fp16": True,
            "bf16": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        }
        compute_capability = f"{props.major}.{props.minor}"
    else:
        gpu_name = None
        total_vram_gb = 0.0
        mixed_precision = {
            "fp16": False,
            "bf16": False,
        }
        compute_capability = None

    checks: dict[str, object] = {"cuda_operation": None, "fp16": None, "bf16": None}
    if cuda_available and run_cuda_checks:
        checks["cuda_operation"] = _numerical_check(torch.float32)
        for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            if mixed_precision[name]:
                checks[name] = _numerical_check(dtype)

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": _driver_version() if cuda_available else None,
        "compute_capability": compute_capability,
        "cudnn_version": torch.backends.cudnn.version(),
        "mixed_precision": mixed_precision,
        "numerical_checks": checks,
        "readiness_status": "candidate_not_gpu_validated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-safe CUDA diagnostics (no training experiment).")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA or numerical checks fail.")
    parser.add_argument("--skip-cuda-checks", action="store_true", help="Report metadata without CUDA operations.")
    args = parser.parse_args()
    report = collect_gpu_diagnostics(run_cuda_checks=not args.skip_cuda_checks)
    print(json.dumps(report, indent=2))
    if args.require_cuda:
        checks = report["numerical_checks"]
        passed = report["cuda_available"] and not args.skip_cuda_checks and all(
            check["passed"] for check in checks.values() if check is not None
        )
        raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
