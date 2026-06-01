"""
tests/unit/test_config.py — Unit tests for the configuration schema.
"""
import json
import tempfile
from pathlib import Path

import pytest

from supreme_modeltx.model_core.config.schema import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SMTXConfig,
    TokenizerConfig,
    TrainingConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.vocab_size == 32_000
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 6
        assert cfg.num_attention_heads == 8
        assert cfg.num_key_value_heads == 8

    def test_custom(self):
        cfg = ModelConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
        assert cfg.hidden_size == 256

    def test_kv_heads_validation(self):
        with pytest.raises(Exception):
            # 3 kv heads does not divide 8 query heads
            ModelConfig(num_attention_heads=8, num_key_value_heads=3)

    def test_gqa(self):
        # GQA: 8 query heads, 2 kv heads
        cfg = ModelConfig(num_attention_heads=8, num_key_value_heads=2)
        assert cfg.num_key_value_heads == 2


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.max_steps == 10_000
        assert cfg.batch_size == 8
        assert cfg.precision.dtype == "bfloat16"
        assert cfg.optimizer.name == "adamw"
        assert cfg.scheduler.name == "cosine"

    def test_checkpoint_config(self):
        cfg = TrainingConfig()
        assert cfg.checkpoint.keep_last_n == 3
        assert cfg.checkpoint.save_every_n_steps == 500


class TestSMTXConfig:
    def test_defaults(self):
        cfg = SMTXConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_from_json_file(self):
        data = {
            "model": {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4},
            "training": {"max_steps": 100},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            fpath = f.name
        cfg = SMTXConfig.from_file(fpath)
        assert cfg.model.hidden_size == 128
        assert cfg.training.max_steps == 100

    def test_roundtrip_json(self):
        cfg = SMTXConfig()
        json_str = cfg.to_json()
        loaded = SMTXConfig.model_validate_json(json_str)
        assert loaded.model.vocab_size == cfg.model.vocab_size

    def test_save_and_load(self, tmp_path):
        cfg = SMTXConfig()
        cfg.model.hidden_size = 256
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = SMTXConfig.from_file(path)
        assert loaded.model.hidden_size == 256

    def test_gpu_first_run_config_loads(self):
        repo_root = Path(__file__).resolve().parents[2]
        cfg_path = repo_root / "configs" / "real_training" / "t_dev_6l_first_gpu_run.json"
        cfg = SMTXConfig.from_file(cfg_path)
        assert cfg.training.precision.enabled is True
        assert cfg.training.precision.dtype == "bfloat16"
        assert cfg.training.batch_size == 8
        assert cfg.data.max_seq_len == 512


class TestOptimizerConfig:
    def test_weight_decay_default(self):
        cfg = OptimizerConfig()
        assert cfg.weight_decay == 0.1
        assert cfg.grad_clip == 1.0
