import json

from supreme_modeltx.foundation.config import InferenceRunConfig, TrainingRunConfig
from supreme_modeltx.foundation.inference import FoundationResponder
from supreme_modeltx.foundation import training as training_module
from supreme_modeltx.foundation.training import checkpoint_path, train, training_state_path


def test_checkpoint_path_uses_expected_pattern(tmp_path):
    path = checkpoint_path(tmp_path / "run", 7)
    assert path.name == "checkpoint_step_00000007.pt"
    assert path.parent == tmp_path / "run" / "checkpoints"
    assert training_state_path(tmp_path / "run", 7).name == "checkpoint_step_00000007.state.pt"


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
    assert training_state_path(tmp_path / "run", 2).exists()
    metrics_lines = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2
    latest_summary = json.loads((tmp_path / "run" / "training_summary.json").read_text(encoding="utf-8"))
    assert latest_summary["steps_completed"] == 2


def test_training_resume_and_checkpoint_inference(tmp_path):
    initial_config = TrainingRunConfig.model_validate(
        {
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 1,
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "keep_last_n_checkpoints": 2,
                "output_dir": str(tmp_path / "resume_run"),
                "auto_resume_latest": True,
                "train_samples": 8,
                "validation_samples": 4,
            }
        }
    )
    train(initial_config)

    resumed_config = TrainingRunConfig.model_validate(
        {
            **initial_config.model_dump(mode="json"),
            "training": {
                **initial_config.training.model_dump(mode="json"),
                "max_steps": 2,
            },
        }
    )
    resumed_summary = train(resumed_config)
    assert resumed_summary["steps_completed"] == 2
    assert resumed_summary["latest_training_state"].endswith(".state.pt")

    responder = FoundationResponder(
        InferenceRunConfig.model_validate(
        {
            "model": resumed_config.model.model_dump(mode="json"),
            "inference": {
                "backend": "checkpoint",
                "device": "cpu",
                "checkpoint_path": resumed_summary["latest_checkpoint"],
                "response_prefix": "Draft pilot response:",
                "max_new_tokens": 8,
            },
        }
        )
    )
    assert responder.generate("hello")


def test_training_interrupt_writes_checkpoint_and_summary(tmp_path, monkeypatch):
    config = TrainingRunConfig.model_validate(
        {
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 3,
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "keep_last_n_checkpoints": 2,
                "output_dir": str(tmp_path / "interrupt_run"),
                "auto_resume_latest": False,
                "train_samples": 8,
                "validation_samples": 4,
            }
        }
    )

    original_make_batch = training_module._make_batch
    call_count = {"value": 0}

    def interrupting_make_batch(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise KeyboardInterrupt()
        return original_make_batch(*args, **kwargs)

    monkeypatch.setattr(training_module, "_make_batch", interrupting_make_batch)

    summary = train(config)

    assert summary["status"] == "interrupted"
    assert checkpoint_path(tmp_path / "interrupt_run", 1).exists()
    assert training_state_path(tmp_path / "interrupt_run", 1).exists()
    assert (tmp_path / "interrupt_run" / "training_summary.json").exists()
