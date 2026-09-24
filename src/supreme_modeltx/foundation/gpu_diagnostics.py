from __future__ import annotations

import json

import torch


def collect_gpu_diagnostics() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        total_vram_gb = round(props.total_memory / (1024**3), 2)
        mixed_precision = {
            "fp16": True,
            "bf16": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        }
    else:
        gpu_name = None
        total_vram_gb = 0.0
        mixed_precision = {
            "fp16": False,
            "bf16": False,
        }

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mixed_precision": mixed_precision,
    }


def main() -> None:
    print(json.dumps(collect_gpu_diagnostics(), indent=2))


if __name__ == "__main__":
    main()
