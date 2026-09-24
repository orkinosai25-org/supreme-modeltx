from pathlib import Path

from supreme_modeltx.foundation.config import EvaluationRunConfig, TrainingRunConfig
from supreme_modeltx.foundation.evaluation import run_evaluation
from supreme_modeltx.foundation.training import train


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_private_foundation_cpu_smoke(tmp_path):
    training_config = TrainingRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "training-smoke.yaml")
    training_config.training.output_dir = str(tmp_path / "smoke_run")
    summary = train(training_config)
    assert summary["status"] == "completed"

    evaluation_config = EvaluationRunConfig.from_file(REPO_ROOT / "configs" / "foundation" / "evaluation.yaml")
    evaluation_config.inference.backend = "checkpoint"
    evaluation_config.inference.checkpoint_path = summary["latest_checkpoint"]
    evaluation_config.evaluation.output_path = str(tmp_path / "evaluation.json")
    results = run_evaluation(evaluation_config)
    assert results["total_cases"] >= 2
