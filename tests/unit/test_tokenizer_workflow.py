import json

from supreme_modeltx.model_core.config.schema import SMTXConfig
from supreme_modeltx.model_core.tokenizer.workflow import (
    TokenizerWorkflow,
    train_versioned_sentencepiece,
)


def test_train_versioned_sentencepiece_creates_artifacts(tmp_path):
    corpus_dir = tmp_path / "raw"
    corpus_dir.mkdir()
    (corpus_dir / "part1.txt").write_text("hello world\nthis is tokenizer training\n", encoding="utf-8")
    (corpus_dir / "part2.txt").write_text("another sovereign local sample\n", encoding="utf-8")

    artifacts = train_versioned_sentencepiece(
        input_paths=[str(corpus_dir)],
        artifact_root=tmp_path / "artifacts",
        model_variant="t-dev-6l",
        version="v-test",
        vocab_size=64,
        character_coverage=1.0,
    )

    assert artifacts.artifact_dir == tmp_path / "artifacts" / "t-dev-6l" / "v-test"
    assert artifacts.model_path.exists()
    assert artifacts.vocab_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.corpus_path.exists()

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["backend"] == "sentencepiece"
    assert metadata["version"] == "v-test"
    assert metadata["model_variant"] == "t-dev-6l"


def test_tokenizer_workflow_roundtrip_after_training(tmp_path):
    corpus_file = tmp_path / "tiny.txt"
    corpus_file.write_text("british sovereign ai training corpus\n", encoding="utf-8")

    artifacts = train_versioned_sentencepiece(
        input_paths=[str(corpus_file)],
        artifact_root=tmp_path / "artifacts",
        model_variant="t-dev-6l",
        version="v-roundtrip",
        vocab_size=64,
        character_coverage=1.0,
    )

    tokenizer = TokenizerWorkflow(artifacts.model_path)
    ids = tokenizer.encode("sovereign tokenizer")
    text = tokenizer.decode(ids)

    assert len(ids) > 0
    assert isinstance(text, str)
    assert text


def test_manifest_input_training_and_config_paths(tmp_path):
    text_path = tmp_path / "train.txt"
    text_path.write_text("manifest based tokenizer data\nmore local text\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "1",
                "sources": [{"name": "local", "backend": "text", "path": str(text_path)}],
            }
        ),
        encoding="utf-8",
    )

    artifacts = train_versioned_sentencepiece(
        manifest_path=str(manifest_path),
        artifact_root=tmp_path / "artifacts",
        model_variant="t-dev-6l",
        version="v-manifest",
        vocab_size=64,
        character_coverage=1.0,
    )

    cfg = SMTXConfig()
    cfg.tokenizer.model_path = str(artifacts.model_path)
    resolved = cfg.data.tokenizer_path or cfg.tokenizer.model_path

    assert resolved == str(artifacts.model_path)
