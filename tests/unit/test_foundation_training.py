import json

from supreme_modeltx.foundation.config import TrainingRunConfig
from supreme_modeltx.foundation.training import checkpoint_path, train


def test_checkpoint_path_uses_expected_pattern(tmp_path):
    path = checkpoint_path(tmp_path / "run", 7)
    assert path.name == "checkpoint_step_00000007.pt"
    assert path.parent == tmp_path / "run" / "checkpoints"


def test_training_smoke_writes_checkpoint_and_summary(tmp_path):
    config = TrainingRunConfig.model_validate(
        {
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 2,
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "keep_last_n_checkpoints": 2,
                "output_dir": str(tmp_path / "run"),
                "auto_resume_latest": False,
                "train_samples": 8,
                "validation_samples": 4,
            }
        }
    )

    summary = train(config)

    assert summary["status"] == "completed"
    assert (tmp_path / "run" / "training_summary.json").exists()
    assert checkpoint_path(tmp_path / "run", 2).exists()
    metrics_lines = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2
    latest_summary = json.loads((tmp_path / "run" / "training_summary.json").read_text(encoding="utf-8"))
    assert latest_summary["steps_completed"] == 2
