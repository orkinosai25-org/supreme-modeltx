"""
tests/smoke/test_model_smoke.py — Smoke tests for model instantiation and forward pass.

These tests run on CPU and require no GPU or pre-trained weights.
They verify:
  1. Model can be instantiated from config
  2. A forward pass produces output of expected shape
  3. Loss is computed when labels are provided
  4. The training loop runs for 2 steps without error (dry run)
  5. Sampling utilities work on random logits
"""
import torch
import pytest

from supreme_modeltx.model_core.config.schema import ModelConfig, SMTXConfig
from supreme_modeltx.model_core.models.t_series.baseline import TSeriesBaseline
from supreme_modeltx.model_core.models.common.attention import (
    GroupedQueryAttention,
    precompute_freqs_cis,
)
from supreme_modeltx.model_core.inference.sampling import sample_tokens
from supreme_modeltx.model_core.training.trainer import train


# ── Tiny config for fast CPU smoke runs ───────────────────────────────────────

def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
    )


class TestModelInstantiation:
    def test_dev_model_creates(self):
        model = TSeriesBaseline.dev_model()
        assert model is not None

    def test_param_count_positive(self):
        model = TSeriesBaseline.dev_model()
        assert model.num_parameters() > 0

    def test_tiny_model_creates(self):
        cfg = _tiny_config()
        model = TSeriesBaseline.from_config(cfg)
        assert model is not None

    def test_from_config(self):
        cfg = _tiny_config()
        model = TSeriesBaseline(cfg)
        assert model.config.vocab_size == 256


class TestForwardPass:
    def test_output_shape(self):
        cfg = _tiny_config()
        model = TSeriesBaseline.from_config(cfg)
        model.eval()
        B, T = 2, 16
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert "logits" in out
        assert out["logits"].shape == (B, T, cfg.vocab_size)

    def test_loss_computed_with_labels(self):
        cfg = _tiny_config()
        model = TSeriesBaseline.from_config(cfg)
        model.eval()
        B, T = 2, 16
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
        assert "loss" in out
        assert out["loss"].item() > 0.0
        assert torch.isfinite(out["loss"])

    def test_backward_pass(self):
        """Gradient flow check — loss.backward() should not raise."""
        cfg = _tiny_config()
        model = TSeriesBaseline.from_config(cfg)
        model.train()
        B, T = 1, 8
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = model(input_ids=input_ids, labels=input_ids)
        out["loss"].backward()
        # Check at least one gradient is non-None
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_gqa_model(self):
        """GQA variant: 4 query heads, 2 kv heads."""
        cfg = ModelConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            max_position_embeddings=64,
        )
        # Only instantiate if GQA split is valid
        cfg2 = _tiny_config()
        cfg2.num_key_value_heads = 2
        cfg2.num_attention_heads = 4
        model = TSeriesBaseline.from_config(cfg2)
        model.eval()
        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out = model(input_ids=ids)
        assert out["logits"].shape[-1] == 256


class TestSamplingUtilities:
    def test_greedy(self):
        logits = torch.randn(1, 256)
        token = sample_tokens(logits, temperature=0.0)
        assert token.item() == torch.argmax(logits).item()

    def test_temperature_sampling(self):
        torch.manual_seed(42)
        logits = torch.randn(256)
        token = sample_tokens(logits, temperature=1.0)
        assert 0 <= token.item() < 256

    def test_top_k(self):
        logits = torch.randn(256)
        token = sample_tokens(logits, temperature=1.0, top_k=10)
        assert 0 <= token.item() < 256

    def test_top_p(self):
        logits = torch.randn(256)
        token = sample_tokens(logits, temperature=1.0, top_p=0.9)
        assert 0 <= token.item() < 256


class TestAttentionPrimitives:
    def test_rope_freqs(self):
        freqs = precompute_freqs_cis(head_dim=32, seq_len=16)
        assert freqs.shape == (16, 16)  # (seq_len, head_dim//2)

    def test_gqa_attention_forward(self):
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=32,
        )
        x = torch.randn(1, 16, 64)
        out = attn(x)
        assert out.shape == (1, 16, 64)


class TestTrainingDryRun:
    """Minimal dry-run training smoke test (CPU, 2 steps, tiny model)."""

    def test_dry_run(self):
        cfg = SMTXConfig()
        cfg.model = _tiny_config()
        cfg.training.batch_size = 2
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.log_every_n_steps = 1
        # dry_run=True runs exactly 2 steps
        train(cfg, dry_run=True)
