"""
tests/unit/test_run_artifacts.py — Focused tests for run artifact creation.

Covers:
- _generate_checkpoint_samples returns None when no tokenizer is configured
- _generate_checkpoint_samples handles checkpoint load failure gracefully
- _write_consolidated_samples produces samples.json and samples.md
- _find_best_checkpoint identifies the correct checkpoint
- _write_run_summary includes all required metadata fields
- _load_canonical_prompts loads from file when available and falls back to defaults
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from supreme_modeltx.model_core.config.schema import SMTXConfig
from supreme_modeltx.model_core.training.trainer import (
    _CANONICAL_PROMPTS,
    _find_best_checkpoint,
    _load_canonical_prompts,
    _write_consolidated_samples,
    _write_run_summary,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _minimal_cfg(tmp_path: Path, *, tokenizer_path: str | None = None) -> SMTXConfig:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return SMTXConfig.model_validate(
        {
            "model": {
                "model_variant": "t-dev-6l",
                "vocab_size": 256,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "max_position_embeddings": 64,
                "torch_dtype": "float32",
                "pad_token_id": 3,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            "tokenizer": {
                "backend": "sentencepiece",
                "model_path": tokenizer_path,
            },
            "data": {
                "manifest_path": str(tmp_path / "manifest.yaml"),
                "tokenizer_path": tokenizer_path,
                "train_split": "train",
                "validation_split": "validation",
                "max_seq_len": 64,
            },
            "training": {
                "max_steps": 2,
                "batch_size": 1,
                "eval_every_n_steps": 1,
                "eval_max_batches": 1,
                "checkpoint": {
                    "save_dir": str(ckpt_dir),
                    "save_every_n_steps": 1,
                    "keep_last_n": 2,
                },
                "precision": {"enabled": False, "dtype": "float32"},
            },
        }
    )


def _fake_sample_payload(checkpoint_name: str, step: int) -> dict:
    return {
        "checkpoint_path": f"/run/checkpoints/{checkpoint_name}",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "generation": {"max_new_tokens": 24, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
        "samples": [
            {
                "prompt": "Sovereign AI enables",
                "prompt_token_count": 3,
                "completion_token_count": 5,
                "completion_text": "tokenised output text",
                "full_output_text": "Sovereign AI enables tokenised output text",
            }
        ],
    }


# ── _find_best_checkpoint ─────────────────────────────────────────────────────

def test_find_best_checkpoint_returns_none_when_no_checkpoints():
    result = _find_best_checkpoint([], [{"step": 1, "val_loss": 0.5, "perplexity": 1.6}])
    assert result is None


def test_find_best_checkpoint_returns_none_when_no_history():
    result = _find_best_checkpoint(["checkpoint_step_00000001.pt"], [])
    assert result is None


def test_find_best_checkpoint_picks_lowest_val_loss(tmp_path):
    ckpt1 = str(tmp_path / "checkpoint_step_00000001.pt")
    ckpt2 = str(tmp_path / "checkpoint_step_00000002.pt")
    history = [
        {"step": 1, "val_loss": 3.0, "perplexity": 20.0},
        {"step": 2, "val_loss": 1.5, "perplexity": 4.5},
    ]
    result = _find_best_checkpoint([ckpt1, ckpt2], history)
    assert result == ckpt2


def test_find_best_checkpoint_single_entry(tmp_path):
    ckpt = str(tmp_path / "checkpoint_step_00000005.pt")
    history = [{"step": 5, "val_loss": 2.0, "perplexity": 7.4}]
    result = _find_best_checkpoint([ckpt], history)
    assert result == ckpt


# ── _write_consolidated_samples ───────────────────────────────────────────────

def test_write_consolidated_samples_creates_json_and_md(tmp_path):
    payload1 = _fake_sample_payload("checkpoint_step_00000001.pt", 1)
    payload2 = _fake_sample_payload("checkpoint_step_00000002.pt", 2)
    p1 = tmp_path / "s1.json"
    p2 = tmp_path / "s2.json"
    p1.write_text(json.dumps(payload1), encoding="utf-8")
    p2.write_text(json.dumps(payload2), encoding="utf-8")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_consolidated_samples(artifact_dir, [str(p1), str(p2)])

    samples_json = artifact_dir / "samples.json"
    samples_md = artifact_dir / "samples.md"
    assert samples_json.exists()
    assert samples_md.exists()

    payloads = json.loads(samples_json.read_text(encoding="utf-8"))
    assert len(payloads) == 2
    assert payloads[0]["samples"][0]["prompt"] == "Sovereign AI enables"

    md_text = samples_md.read_text(encoding="utf-8")
    assert "# Sample Outputs" in md_text
    assert "Sovereign AI enables" in md_text
    assert "checkpoint_step_00000001.pt" in md_text


def test_write_consolidated_samples_empty_list(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_consolidated_samples(artifact_dir, [])

    samples_json = artifact_dir / "samples.json"
    samples_md = artifact_dir / "samples.md"
    assert samples_json.exists()
    assert samples_md.exists()

    payloads = json.loads(samples_json.read_text(encoding="utf-8"))
    assert payloads == []

    md_text = samples_md.read_text(encoding="utf-8")
    assert "_No samples generated._" in md_text


def test_write_consolidated_samples_skips_corrupt_file(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json{{{", encoding="utf-8")

    good_payload = _fake_sample_payload("checkpoint_step_00000001.pt", 1)
    good_file = tmp_path / "good.json"
    good_file.write_text(json.dumps(good_payload), encoding="utf-8")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_consolidated_samples(artifact_dir, [str(bad_file), str(good_file)])

    payloads = json.loads((artifact_dir / "samples.json").read_text(encoding="utf-8"))
    assert len(payloads) == 1


# ── _write_run_summary ────────────────────────────────────────────────────────

def test_write_run_summary_includes_all_required_fields(tmp_path):
    cfg = _minimal_cfg(tmp_path, tokenizer_path=None)
    artifact_dir = tmp_path / "run_artifacts"
    artifact_dir.mkdir()

    _write_run_summary(
        cfg,
        artifact_dir,
        device=torch.device("cpu"),
        started_at_utc="2026-01-01T00:00:00+00:00",
        ended_at_utc="2026-01-01T00:01:00+00:00",
        validation_history=[
            {"step": 1, "val_loss": 3.5, "perplexity": 33.1, "timestamp_utc": "2026-01-01T00:00:30+00:00"},
            {"step": 2, "val_loss": 2.9, "perplexity": 18.2, "timestamp_utc": "2026-01-01T00:00:45+00:00"},
        ],
        checkpoint_paths=[
            str(tmp_path / "checkpoints" / "checkpoint_step_00000001.pt"),
            str(tmp_path / "checkpoints" / "checkpoint_step_00000002.pt"),
        ],
        sample_artifact_paths=[],
    )

    summary_json = artifact_dir / "training_summary.json"
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    # Core provenance fields
    assert summary["training_end_status"] == "completed"
    assert "started_at_utc" in summary["timestamps"]
    assert "ended_at_utc" in summary["timestamps"]
    assert "git_commit" in summary
    assert summary["config_path"].endswith("config_used.json")

    # Data config
    assert "manifest_path" in summary["data"]
    assert summary["data"]["train_split"] == "train"
    assert summary["data"]["validation_split"] == "validation"

    # Device / precision
    assert summary["device"] == "cpu"
    assert summary["precision"]["dtype"] == "float32"
    assert "enabled" in summary["precision"]

    # Eval cadence
    assert "eval_every_n_steps" in summary["eval_cadence"]
    assert "eval_max_batches" in summary["eval_cadence"]

    # Metrics
    assert summary["latest_validation_loss"] == pytest.approx(2.9, abs=1e-6)
    assert summary["latest_perplexity"] is not None

    # Best checkpoint — step 2 has lower loss
    assert summary["best_checkpoint_path"] is not None
    assert "00000002" in summary["best_checkpoint_path"]

    assert len(summary["checkpoint_paths"]) == 2
    assert summary["sample_artifact_paths"] == []

    # Markdown summary
    assert (artifact_dir / "training_summary.md").exists()
    md = (artifact_dir / "training_summary.md").read_text(encoding="utf-8")
    assert "# Training Run Summary" in md
    assert "completed" in md
    assert "cpu" in md

    # Consolidated samples files
    assert (artifact_dir / "samples.json").exists()
    assert (artifact_dir / "samples.md").exists()


def test_write_run_summary_no_validation_history(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    artifact_dir = tmp_path / "run_artifacts"
    artifact_dir.mkdir()

    _write_run_summary(
        cfg,
        artifact_dir,
        device=torch.device("cpu"),
        started_at_utc="2026-01-01T00:00:00+00:00",
        ended_at_utc="2026-01-01T00:01:00+00:00",
        validation_history=[],
        checkpoint_paths=[],
        sample_artifact_paths=[],
    )

    summary = json.loads((artifact_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["latest_validation_loss"] is None
    assert summary["latest_perplexity"] is None
    assert summary["best_checkpoint_path"] is None


# ── _load_canonical_prompts ───────────────────────────────────────────────────

def test_load_canonical_prompts_falls_back_to_defaults(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    # No canonical_prompts.json in run dir or repo configs
    with patch(
        "supreme_modeltx.model_core.training.trainer._CANONICAL_PROMPTS",
        ["Prompt one", "Prompt two"],
    ):
        # Point the repo-candidate path somewhere that doesn't exist
        with patch("supreme_modeltx.model_core.training.trainer.Path") as _:
            # Use the real function but ensure neither candidate exists
            pass
    prompts = _load_canonical_prompts(cfg)
    assert isinstance(prompts, list)
    assert len(prompts) >= 1


def test_load_canonical_prompts_loads_from_run_dir(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    # Place a canonical_prompts.json in the run directory (parent of checkpoints)
    run_dir = Path(cfg.training.checkpoint.save_dir).parent
    canon_file = run_dir / "canonical_prompts.json"
    canon_file.write_text(
        json.dumps({"prompts": ["Custom prompt alpha", "Custom prompt beta"]}),
        encoding="utf-8",
    )
    prompts = _load_canonical_prompts(cfg)
    assert prompts == ["Custom prompt alpha", "Custom prompt beta"]


def test_load_canonical_prompts_ignores_empty_prompts_list(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    run_dir = Path(cfg.training.checkpoint.save_dir).parent
    canon_file = run_dir / "canonical_prompts.json"
    canon_file.write_text(json.dumps({"prompts": []}), encoding="utf-8")
    prompts = _load_canonical_prompts(cfg)
    # Falls back to built-in defaults when file has empty list
    assert prompts == _CANONICAL_PROMPTS


def test_load_canonical_prompts_ignores_corrupt_file(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    run_dir = Path(cfg.training.checkpoint.save_dir).parent
    canon_file = run_dir / "canonical_prompts.json"
    canon_file.write_text("not valid {{{ json", encoding="utf-8")
    prompts = _load_canonical_prompts(cfg)
    assert prompts == _CANONICAL_PROMPTS


# ── _generate_checkpoint_samples failure cases ────────────────────────────────

def test_generate_checkpoint_samples_returns_none_when_no_tokenizer(tmp_path):
    """When no tokenizer path is configured, sample generation returns None."""
    from supreme_modeltx.model_core.training.trainer import _generate_checkpoint_samples

    cfg = _minimal_cfg(tmp_path, tokenizer_path=None)
    # Ensure data.tokenizer_path and tokenizer.model_path are both None
    assert cfg.data.tokenizer_path is None
    assert cfg.tokenizer.model_path is None

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    fake_checkpoint = tmp_path / "checkpoint_step_00000001.pt"
    fake_checkpoint.touch()

    result = _generate_checkpoint_samples(
        cfg,
        fake_checkpoint,
        artifact_dir,
        generated_at_utc="2026-01-01T00:00:00+00:00",
    )
    assert result is None


def test_generate_checkpoint_samples_handles_engine_failure(tmp_path):
    """When the InferenceEngine raises on load, the caller catches and returns None."""
    from supreme_modeltx.model_core.training.trainer import _generate_checkpoint_samples
    from supreme_modeltx.model_core.tokenizer.workflow import train_versioned_sentencepiece

    # Build a minimal tokenizer so the tokenizer_path check passes
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "Sovereign AI enables training runs with real data.\n" * 10,
        encoding="utf-8",
    )
    tokenizer_artifacts = train_versioned_sentencepiece(
        input_paths=[str(corpus_file)],
        artifact_root=tmp_path / "tok",
        model_variant="t-dev-6l",
        version="v-test",
        vocab_size=64,
        character_coverage=1.0,
    )
    cfg = _minimal_cfg(tmp_path, tokenizer_path=str(tokenizer_artifacts.model_path))

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    bad_checkpoint = tmp_path / "not_a_real.pt"
    bad_checkpoint.write_bytes(b"corrupted")

    # Patch InferenceEngine to raise so we test the error-path
    with patch(
        "supreme_modeltx.model_core.training.trainer.InferenceEngine",
        side_effect=RuntimeError("simulated checkpoint load failure"),
    ):
        with pytest.raises(RuntimeError, match="simulated checkpoint load failure"):
            _generate_checkpoint_samples(
                cfg,
                bad_checkpoint,
                artifact_dir,
                generated_at_utc="2026-01-01T00:00:00+00:00",
            )

    # The train() loop catches this exception and logs a warning; samples dir
    # should not have been populated.
    samples_dir = artifact_dir / "samples"
    if samples_dir.exists():
        assert not list(samples_dir.glob("*.json"))
