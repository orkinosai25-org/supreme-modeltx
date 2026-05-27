"""Smoke tests for T-Series model instantiation and forward pass."""

import torch
import pytest

from supreme_modeltx.model_core.config import ModelConfig, T_SERIES_CONFIGS
from supreme_modeltx.model_core.models.t_series import TSeriesModel


def _tiny_config() -> ModelConfig:
    """Return the smallest possible config suitable for CPU smoke tests."""
    return ModelConfig(
        model_id="smoke-tiny",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=64,
    )


class TestTSeriesInstantiation:
    def test_instantiate_tiny(self):
        cfg = _tiny_config()
        model = TSeriesModel(cfg)
        assert model is not None

    def test_num_parameters_positive(self):
        cfg = _tiny_config()
        model = TSeriesModel(cfg)
        assert model.num_parameters > 0

    def test_instantiate_t101_preset(self):
        cfg = T_SERIES_CONFIGS["t101"]
        model = TSeriesModel(cfg)
        assert model.num_parameters > 0


class TestTSeriesForwardPass:
    def setup_method(self):
        self.cfg = _tiny_config()
        self.model = TSeriesModel(self.cfg)
        self.model.eval()

    def test_forward_returns_logits(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = self.model(input_ids=ids)
        assert "logits" in out
        assert out["logits"].shape == (1, 8, self.cfg.vocab_size)

    def test_forward_with_labels_returns_loss(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        labels = ids.clone()
        with torch.no_grad():
            out = self.model(input_ids=ids, labels=labels)
        assert "loss" in out
        assert out["loss"].item() > 0

    def test_batch_forward(self):
        batch_size = 4
        seq_len = 16
        ids = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            out = self.model(input_ids=ids)
        assert out["logits"].shape == (batch_size, seq_len, self.cfg.vocab_size)

    def test_logits_are_finite(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = self.model(input_ids=ids)
        assert torch.isfinite(out["logits"]).all()


class TestTSeriesGeneration:
    def test_generate_returns_tokens(self):
        from supreme_modeltx.model_core.inference.engine import InferenceEngine

        cfg = _tiny_config()
        model = TSeriesModel(cfg)
        engine = InferenceEngine(model, device="cpu")

        input_ids = [1, 2, 3, 4]
        generated = engine.generate(input_ids, max_new_tokens=5, top_p=1.0)
        assert isinstance(generated, list)
        assert len(generated) == 5
        assert all(0 <= tok < cfg.vocab_size for tok in generated)
