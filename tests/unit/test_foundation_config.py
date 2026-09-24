from pathlib import Path

from supreme_modeltx.foundation.config import EvaluationRunConfig, InferenceRunConfig, TrainingRunConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_training_config_loads():
    config = TrainingRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "training-smoke.yaml")
    assert config.training.batch_size == 4
    assert config.training.gradient_accumulation_steps == 2
    assert config.training.output_dir.endswith("artifacts/private_foundation/smoke_run")


def test_inference_config_loads():
    config = InferenceRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "inference.yaml")
    assert config.inference.backend == "stub"
    assert config.inference.response_prefix == "Draft pilot response:"


def test_evaluation_config_loads():
    config = EvaluationRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "evaluation.yaml")
    assert config.evaluation.suite_path.endswith("configs/foundation/eval_cases/smoke-v1.yaml")
