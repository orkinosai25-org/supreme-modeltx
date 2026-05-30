from __future__ import annotations

import json
from pathlib import Path

from supreme_modeltx.model_core.config.schema import SMTXConfig
from supreme_modeltx.model_core.tokenizer.workflow import train_versioned_sentencepiece
from supreme_modeltx.model_core.training.trainer import train


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_manifest_sentencepiece_training_emits_eval_and_checkpoint(tmp_path, caplog):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    _write_jsonl(
        train_file,
        [
            {"text": "tiny sovereign training sample one"},
            {"text": "tiny sovereign training sample two"},
            {"text": "tiny sovereign training sample three"},
            {"text": "tiny sovereign training sample four"},
        ],
    )
    _write_jsonl(
        val_file,
        [
            {"text": "tiny sovereign validation sample one"},
            {"text": "tiny sovereign validation sample two"},
        ],
    )

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                'version: "1"',
                "description: trainer-real-run-test",
                "sources:",
                "  - name: train-a",
                "    backend: jsonl",
                f"    path: {train_file.as_posix()}",
                "    split: train",
                "  - name: val-a",
                "    backend: jsonl",
                f"    path: {val_file.as_posix()}",
                "    split: validation",
            ]
        ),
        encoding="utf-8",
    )

    tokenizer_artifacts = train_versioned_sentencepiece(
        manifest_path=str(manifest_path),
        artifact_root=tmp_path / "artifacts" / "tokenizers",
        model_variant="t-dev-6l",
        version="v-test",
        vocab_size=64,
        character_coverage=1.0,
    )

    cfg = SMTXConfig.model_validate(
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
                "model_path": str(tokenizer_artifacts.model_path),
            },
            "data": {
                "manifest_path": str(manifest_path),
                "tokenizer_path": str(tokenizer_artifacts.model_path),
                "train_split": "train",
                "validation_split": "validation",
                "max_seq_len": 64,
                "pack_sequences": True,
            },
            "training": {
                "max_steps": 2,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "log_every_n_steps": 1,
                "eval_every_n_steps": 1,
                "eval_max_batches": 1,
                "checkpoint": {
                    "save_dir": str(tmp_path / "checkpoints"),
                    "save_every_n_steps": 1,
                    "keep_last_n": 2,
                },
                "precision": {"enabled": False, "dtype": "float32"},
                "distributed": {"backend": "auto"},
            },
        }
    )

    train(cfg)

    assert (tmp_path / "checkpoints" / "checkpoint_step_00000001.pt").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_step_00000002.pt").exists()

    assert "eval step=1/2" in caplog.text
    assert "perplexity=" in caplog.text
