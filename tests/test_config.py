"""Unit tests for model_core.config schema."""

import pytest
from pydantic import ValidationError

from supreme_modeltx.model_core.config import ModelConfig, TrainingConfig, T_SERIES_CONFIGS


class TestModelConfig:
    def test_default_construction(self):
        cfg = ModelConfig(model_id="test-model")
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        assert cfg.num_attention_heads == 12
        assert cfg.vocab_size == 32000
        assert cfg.hidden_act == "silu"

    def test_custom_construction(self):
        cfg = ModelConfig(
            model_id="tiny",
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
        )
        assert cfg.hidden_size == 128
        assert cfg.num_hidden_layers == 2

    def test_heads_must_divide_hidden(self):
        with pytest.raises(ValidationError, match="divisible"):
            ModelConfig(model_id="bad", hidden_size=100, num_attention_heads=3)

    def test_invalid_hidden_size(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_id="bad", hidden_size=0)

    def test_invalid_vocab_size(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_id="bad", vocab_size=10)

    def test_valid_hidden_acts(self):
        for act in ("gelu", "relu", "silu"):
            cfg = ModelConfig(model_id="test", hidden_act=act)
            assert cfg.hidden_act == act

    def test_invalid_hidden_act(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_id="test", hidden_act="tanh")


class TestTrainingConfig:
    def test_default_construction(self):
        cfg = TrainingConfig(
            run_name="run-001",
            model_id="t101",
            output_dir="/tmp/checkpoints",
            dataset_path="/tmp/data",
        )
        assert cfg.max_steps == 10_000
        assert cfg.learning_rate == pytest.approx(3e-4)
        assert cfg.bf16 is True

    def test_invalid_max_steps(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                run_name="x",
                model_id="t101",
                output_dir="/tmp",
                dataset_path="/tmp",
                max_steps=0,
            )


class TestTSeriesPresets:
    def test_all_presets_present(self):
        for name in ("t101", "t201", "t301", "t501"):
            assert name in T_SERIES_CONFIGS

    def test_presets_valid(self):
        for name, cfg in T_SERIES_CONFIGS.items():
            assert cfg.model_id == name
            assert cfg.hidden_size % cfg.num_attention_heads == 0

    def test_t101_is_smallest(self):
        t101 = T_SERIES_CONFIGS["t101"]
        t501 = T_SERIES_CONFIGS["t501"]
        assert t101.hidden_size < t501.hidden_size
        assert t101.num_hidden_layers < t501.num_hidden_layers
