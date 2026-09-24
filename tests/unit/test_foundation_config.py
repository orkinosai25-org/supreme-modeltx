from pathlib import Path

import pytest

from supreme_modeltx.foundation.config import EvaluationRunConfig, InferenceRunConfig, TrainingRunConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_training_config_loads():
    config = TrainingRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "training-smoke.yaml")
    assert config.training.batch_size == 4
    assert config.training.cuda_device_index == 0
    assert config.training.gradient_accumulation_steps == 2
    assert config.training.output_dir.endswith("artifacts/private_foundation/smoke_run")


def test_inference_config_loads():
    config = InferenceRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "inference.yaml")
    assert config.inference.backend == "stub"
    assert config.inference.response_prefix == "Draft pilot response:"


def test_evaluation_config_loads():
    config = EvaluationRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "evaluation.yaml")
    assert config.evaluation.suite_path.endswith("configs/foundation/eval_cases/smoke-v1.yaml")


def test_gpu_training_config_loads():
    config = TrainingRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "training-single-gpu.yaml")
    assert config.training.device == "cuda"
    assert config.training.cuda_device_index == 0
    assert config.training.mixed_precision == "bf16"


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_cpu_mixed_precision_config_is_rejected(precision):
    with pytest.raises(ValueError, match=f"{precision} mixed precision requires device='cuda'"):
        TrainingRunConfig.model_validate({"training": {"device": "cpu", "mixed_precision": precision}})
