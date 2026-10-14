"""Provider contracts without downloads or pretrained model artifacts."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from supreme_modeltx.platform_api.api import engine
from supreme_modeltx.model_core.inference.providers import (
    LocalTransformersProvider, MistralProvider, PretrainedArtifact,
)


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    files = {
        "config.json": '{"model_type": "qwen2"}',
        "tokenizer_config.json": '{"chat_template": "test"}',
        "model.safetensors": "mock weights, never loaded",
    }
    manifest = {
        "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "a" * 40,
        "license": "Apache-2.0",
        "architecture": "qwen2",
        "conversion_lineage": [],
        "files_sha256": {},
    }
    for name, content in files.items():
        path = root / name
        path.write_text(content)
        manifest["files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "smtx-artifact.json").write_text(json.dumps(manifest))
    return root


def test_artifact_verifies_provenance(snapshot):
    artifact = PretrainedArtifact.verify(snapshot)
    assert artifact.source_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert len(artifact.files_sha256) == 3


def test_artifact_rejects_modified_weights(snapshot):
    (snapshot / "model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        PretrainedArtifact.verify(snapshot)


def test_artifact_rejects_untracked_files(snapshot):
    (snapshot / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="every snapshot file"):
        PretrainedArtifact.verify(snapshot)


def test_artifact_rejects_external_symlink(snapshot, tmp_path):
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.write_text("outside")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(external)
    with pytest.raises(ValueError, match="inside the local snapshot"):
        PretrainedArtifact.verify(snapshot)


def _write_hashed_file(snapshot, name, data):
    path = snapshot / name
    path.write_text(json.dumps(data))
    manifest_path = snapshot / "smtx-artifact.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.fixture
def mistral_snapshot(snapshot):
    _write_hashed_file(snapshot, "config.json", {"model_type": "mistral"})
    _write_hashed_file(snapshot, "tokenizer.json", {})
    manifest_path = snapshot / "smtx-artifact.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(architecture="mistral", source_model="synthetic-mistral-test")
    manifest_path.write_text(json.dumps(manifest))
    return snapshot


@pytest.mark.parametrize("model_type", ["qwen2", "mixtral", None])
def test_mistral_rejects_incompatible_config(mistral_snapshot, model_type):
    _write_hashed_file(mistral_snapshot, "config.json", {"model_type": model_type})
    with pytest.raises(ValueError, match="requires model_type"):
        MistralProvider(str(mistral_snapshot))


def test_mistral_rejects_incompatible_manifest(snapshot):
    _write_hashed_file(snapshot, "config.json", {"model_type": "mistral"})
    with pytest.raises(ValueError, match="manifest architecture"):
        MistralProvider(str(snapshot))


@pytest.mark.parametrize("filename", [
    "smtx-artifact.json", "config.json", "tokenizer_config.json", "model.safetensors",
])
def test_mistral_rejects_missing_snapshot_files(mistral_snapshot, filename):
    (mistral_snapshot / filename).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        MistralProvider(str(mistral_snapshot))


def test_mistral_requires_tokenizer_artifacts(mistral_snapshot):
    (mistral_snapshot / "tokenizer.json").unlink()
    manifest_path = mistral_snapshot / "smtx-artifact.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files_sha256"]["tokenizer.json"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hashed tokenizer"):
        MistralProvider(str(mistral_snapshot))


def test_mistral_rejects_tampered_tokenizer(mistral_snapshot):
    (mistral_snapshot / "tokenizer.json").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch: tokenizer.json"):
        MistralProvider(str(mistral_snapshot))


def test_mistral_rejects_incomplete_shards(mistral_snapshot):
    _write_hashed_file(mistral_snapshot, "model.safetensors.index.json", {
        "weight_map": {"weight": "missing.safetensors"},
    })
    with pytest.raises(ValueError, match="hashed local safetensors"):
        MistralProvider(str(mistral_snapshot))


def test_mistral_requires_loadable_weight_filename(mistral_snapshot):
    weights = mistral_snapshot / "model.safetensors"
    weights.rename(mistral_snapshot / "unindexed.safetensors")
    manifest_path = mistral_snapshot / "smtx-artifact.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["unindexed.safetensors"] = manifest["files_sha256"].pop("model.safetensors")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete shard index"):
        MistralProvider(str(mistral_snapshot))


@pytest.mark.parametrize("shard", [
    "../outside.safetensors", "/tmp/outside.safetensors", "model.bin", "untracked.safetensors",
])
def test_index_cannot_reference_external_unhashed_or_pickle_shards(snapshot, shard):
    _write_hashed_file(snapshot, "model.safetensors.index.json", {"weight_map": {"weight": shard}})
    with pytest.raises(ValueError, match="hashed local safetensors"):
        PretrainedArtifact.verify(snapshot)


def test_valid_index_references_hashed_shards(snapshot):
    _write_hashed_file(snapshot, "model.safetensors.index.json", {
        "weight_map": {"weight": "model.safetensors"},
    })
    assert PretrainedArtifact.verify(snapshot).architecture == "qwen2"


@pytest.mark.parametrize("config", [
    {"tokenizer_file": "/tmp/external.json"},
    {"vocab_file": "../external.model"},
    {"init_inputs": ["/tmp/external.model"]},
    {"fast_tokenizer_files": ["../outside/tokenizer.0.0.json"]},
    {"fast_tokenizer_files": ["/tmp/tokenizer.0.0.json"]},
    {"fast_tokenizer_files": "tokenizer.0.0.json"},
])
def test_tokenizer_cannot_override_external_files(snapshot, config):
    _write_hashed_file(snapshot, "tokenizer_config.json", config)
    with pytest.raises(ValueError, match="Tokenizer"):
        PretrainedArtifact.verify(snapshot)


def test_fast_tokenizer_variants_must_be_inventoried(snapshot):
    _write_hashed_file(snapshot, "tokenizer.0.0.json", {})
    _write_hashed_file(snapshot, "tokenizer_config.json", {
        "fast_tokenizer_files": ["tokenizer.0.0.json"],
    })
    assert PretrainedArtifact.verify(snapshot).architecture == "qwen2"


def test_adapter_cannot_redirect_to_unverified_base_model(snapshot):
    _write_hashed_file(snapshot, "adapter_config.json", {"base_model_name_or_path": "/outside"})
    with pytest.raises(ValueError, match="complete merged snapshot"):
        PretrainedArtifact.verify(snapshot)


@pytest.fixture
def runtime(monkeypatch):
    torch = pytest.importorskip("torch")

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        chat_template = "model-specific template"
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Merhaba"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "templated"

        def __call__(self, prompt, **kwargs):
            assert prompt == "templated"
            assert kwargs["add_special_tokens"] is False
            return Inputs(input_ids=torch.tensor([[1, 4, 5]]))

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [6, 2]
            assert kwargs["skip_special_tokens"]
            return "Merhaba!"

    class Model:
        config = SimpleNamespace(max_position_embeddings=32, model_type="qwen2")

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            self.last_generation = kwargs
            return torch.tensor([[1, 4, 5, 6, 2]])

    model = Model()
    calls = {}

    def tokenizer_loader(path, **kwargs):
        calls["tokenizer"] = kwargs
        return Tokenizer()

    def model_loader(path, **kwargs):
        calls["model"] = kwargs
        return model

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
    ))
    return model, calls


def test_local_generation_contract(snapshot, runtime):
    model, calls = runtime
    provider = LocalTransformersProvider(str(snapshot))
    assert provider.is_ready
    assert provider.metadata["artifact"]["revision"] == "a" * 40
    assert provider.metadata["checkpoint_format"] == "safetensors"
    messages = [SimpleNamespace(role="user", content="Merhaba")]
    assert provider.generate_from_messages(messages, max_new_tokens=4, temperature=0) == (
        "Merhaba!", 3, 2,
    )
    assert model.last_generation["do_sample"] is False
    assert "temperature" not in model.last_generation
    assert model.last_generation["pad_token_id"] == 0
    for name in ("model", "tokenizer"):
        assert calls[name]["local_files_only"] is True
        assert calls[name]["trust_remote_code"] is False
    assert calls["model"]["use_safetensors"] is True


def test_local_tokenizer_file_override_is_resolved_inside_snapshot(snapshot, runtime):
    _, calls = runtime
    _write_hashed_file(snapshot, "tokenizer.json", {})
    _write_hashed_file(snapshot, "tokenizer_config.json", {
        "tokenizer_file": "tokenizer.json", "chat_template": "test",
    })
    LocalTransformersProvider(str(snapshot))
    assert calls["tokenizer"]["tokenizer_file"] == str(snapshot / "tokenizer.json")


def test_context_limit_is_not_silently_truncated(snapshot, runtime):
    provider = LocalTransformersProvider(str(snapshot))
    with pytest.raises(ValueError, match="context window"):
        provider.generate_from_messages([SimpleNamespace(role="user", content="Merhaba")], 32)


def test_zero_top_p_matches_existing_api_contract(snapshot, runtime):
    model, _ = runtime
    provider = LocalTransformersProvider(str(snapshot))
    provider.generate_from_messages(
        [SimpleNamespace(role="user", content="Merhaba")], max_new_tokens=4, top_p=0,
    )
    assert model.last_generation["top_p"] == 0


def test_unsupported_device_fails(snapshot, runtime):
    with pytest.raises(ValueError, match="explicit cpu or cuda"):
        LocalTransformersProvider(str(snapshot), device="auto")


def test_requested_cuda_does_not_fall_back(snapshot, runtime, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested"):
        LocalTransformersProvider(str(snapshot), device="cuda")


def test_cpu_requires_float32(snapshot, runtime):
    with pytest.raises(ValueError, match="float32 on CPU"):
        LocalTransformersProvider(str(snapshot), dtype="float16")


def test_architecture_must_match_manifest(snapshot, runtime):
    model, _ = runtime
    model.config = SimpleNamespace(max_position_embeddings=32, model_type="other")
    with pytest.raises(ValueError, match="architecture"):
        LocalTransformersProvider(str(snapshot))


def test_missing_template_is_rejected(snapshot, runtime, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(chat_template=None))
    with pytest.raises(ValueError, match="chat template"):
        LocalTransformersProvider(str(snapshot))


def test_unsupported_backend_fails_closed(monkeypatch):
    monkeypatch.setattr(engine, "_engine_backend", None)
    monkeypatch.setenv("SMTX_INFERENCE_BACKEND", "tensorrt")
    engine.initialize_engine()
    assert engine.get_engine() is None


@pytest.mark.parametrize("backend", ["transformers", "mistral"])
def test_pretrained_initialization_failure_stays_unloaded(snapshot, monkeypatch, backend):
    monkeypatch.setattr(engine, "_engine_backend", None)
    monkeypatch.setenv("SMTX_INFERENCE_BACKEND", backend)
    monkeypatch.setenv("SMTX_PRETRAINED_MODEL_PATH", str(snapshot))

    def failed_provider(*args, **kwargs):
        raise ValueError("unapproved snapshot")

    monkeypatch.setattr(engine, "MistralProvider" if backend == "mistral" else "LocalTransformersProvider", failed_provider)
    engine.initialize_engine()
    assert engine.get_engine() is None


def test_invalid_mistral_startup_does_not_fall_back(mistral_snapshot, monkeypatch, caplog):
    (mistral_snapshot / "tokenizer.json").write_text("tampered")
    monkeypatch.setattr(engine, "_engine_backend", None)
    monkeypatch.setenv("SMTX_INFERENCE_BACKEND", "mistral")
    monkeypatch.setenv("SMTX_PRETRAINED_MODEL_PATH", str(mistral_snapshot))
    engine.initialize_engine()
    assert engine.get_engine() is None
    assert "hash mismatch: tokenizer.json" in caplog.text


def test_pretrained_registry_failure_stays_unloaded(snapshot, runtime, monkeypatch):
    monkeypatch.setattr(engine, "_engine_backend", None)
    monkeypatch.setenv("SMTX_INFERENCE_BACKEND", "transformers")
    monkeypatch.setenv("SMTX_PRETRAINED_MODEL_PATH", str(snapshot))

    class BrokenRegistry:
        def register(self, entry):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(engine, "ModelRegistry", BrokenRegistry)
    engine.initialize_engine()
    assert engine.get_engine() is None


def test_pretrained_initialization_registers_provenance(snapshot, runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "_engine_backend", None)
    monkeypatch.setenv("SUPREME_MODELTX_PLATFORM_DB_PATH", str(tmp_path / "registry.sqlite3"))
    monkeypatch.setenv("SMTX_INFERENCE_BACKEND", "transformers")
    monkeypatch.setenv("SMTX_PRETRAINED_MODEL_PATH", str(snapshot))
    engine.initialize_engine()
    assert engine.get_engine().is_ready
    entry = engine.ModelRegistry().get_model("local-pretrained")
    assert entry.stage == "development"
    assert json.loads(entry.description)["artifact"]["license"] == "Apache-2.0"


def test_mistral_serves_existing_chat_boundary(mistral_snapshot, runtime, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from supreme_modeltx.platform_api.api.app import create_app
    from supreme_modeltx.platform_api.api.routers.auth import require_api_key

    model, calls = runtime
    model.config.model_type = "mistral"
    monkeypatch.setattr(engine, "_engine_backend", None)
    monkeypatch.setenv("SUPREME_MODELTX_PLATFORM_DB_PATH", str(tmp_path / "registry.sqlite3"))
    monkeypatch.setenv("SMTX_INFERENCE_BACKEND", "mistral")
    monkeypatch.setenv("SMTX_PRETRAINED_MODEL_PATH", str(mistral_snapshot))
    monkeypatch.setenv("SMTX_MODEL_ID", "ebru-mistral-demo")
    monkeypatch.setenv("SUPREME_MODELTX_DEVICE", "cpu")
    engine.initialize_engine()
    assert isinstance(engine.get_engine(), MistralProvider)
    entry = engine.ModelRegistry().get_model("ebru-mistral-demo")
    assert entry.stage == "development"
    assert json.loads(entry.description)["backend"] == "mistral"
    app = create_app()
    app.dependency_overrides[require_api_key] = lambda: "test-project"
    response = TestClient(app).post("/v1/chat/completions", json={
        "model": entry.id, "messages": [{"role": "user", "content": "Merhaba"}],
        "max_tokens": 4, "temperature": 0,
    })
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "Merhaba!"
    assert response.json()["usage"]["prompt_tokens"] == 3
    for name in ("model", "tokenizer"):
        assert calls[name]["local_files_only"] is True
        assert calls[name]["trust_remote_code"] is False
    assert calls["model"]["use_safetensors"] is True


@pytest.mark.parametrize("backend", ["transformers", "mistral"])
def test_existing_evaluation_harness_records_pretrained_metrics(mistral_snapshot, runtime, tmp_path, backend):
    from supreme_modeltx.foundation.config import EvaluationRunConfig
    from supreme_modeltx.foundation.evaluation import run_evaluation

    runtime[0].config.model_type = "mistral"

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({
        "suite_name": "turkish-test", "version": "v1",
        "cases": [{
            "case_id": "greeting", "prompt": "Merhaba",
            "expected": {"exact_match": "Merhaba!"},
        }],
    }))
    cfg = EvaluationRunConfig.model_validate({
        "inference": {
            "backend": backend, "device": "cpu",
            "pretrained_model_path": str(mistral_snapshot), "max_new_tokens": 4,
        },
        "evaluation": {
            "suite_path": str(suite), "output_path": str(tmp_path / "results.json"),
            "compute_cost_per_hour_usd": 1.0,
        },
    })
    result = run_evaluation(cfg)
    assert result["passed_cases"] == 1
    assert result["results"][0]["prompt_tokens"] == 3
    assert result["results"][0]["completion_tokens"] == 2
    assert result["runtime"]["provider"]["backend"] == backend
    assert result["runtime"]["provider"]["artifact"]["source_model"] == "synthetic-mistral-test"
    assert result["runtime"]["suite_sha256"] == hashlib.sha256(suite.read_bytes()).hexdigest()
    assert result["runtime"]["peak_allocated_vram_bytes"] is None
    assert result["runtime"]["estimated_generation_cost_usd"] >= 0


@pytest.mark.parametrize("backend", ["transformers", "mistral"])
def test_pretrained_evaluation_requires_snapshot_path(backend):
    from supreme_modeltx.foundation.config import InferenceRunConfig
    from supreme_modeltx.foundation.inference import FoundationResponder

    with pytest.raises(ValueError, match="pretrained_model_path"):
        FoundationResponder(InferenceRunConfig.model_validate({
            "inference": {"backend": backend, "device": "cpu"},
        }))


@pytest.mark.parametrize("backend", ["transformers", "mistral"])
def test_pretrained_evaluation_requires_explicit_device(backend):
    from supreme_modeltx.foundation.config import InferenceConfig

    with pytest.raises(ValueError, match="explicit device"):
        InferenceConfig(backend=backend, pretrained_model_path="/approved/snapshot")


def test_invalid_provider_request_returns_client_error(monkeypatch):
    from fastapi.testclient import TestClient
    from supreme_modeltx.platform_api.api.app import create_app
    from supreme_modeltx.platform_api.api.routers.auth import require_api_key

    class LimitedProvider:
        def generate_from_messages(self, *args, **kwargs):
            raise ValueError("Prompt and generation budget exceed the model context window.")

    monkeypatch.setattr(engine, "_engine_backend", LimitedProvider())
    app = create_app()
    app.dependency_overrides[require_api_key] = lambda: "test-project"
    response = TestClient(app).post("/v1/chat/completions", json={
        "model": "local-pretrained", "messages": [{"role": "user", "content": "Hello"}],
    })
    assert response.status_code == 400
    assert "context window" in response.json()["detail"]


@pytest.mark.parametrize("architecture", ["gpt2", "mistral"])
def test_real_transformers_local_roundtrip_without_downloads(tmp_path, monkeypatch, architecture):
    import socket

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "Hello": 3, "assistant": 4},
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}assistant",
        model_input_names=["input_ids", "attention_mask"],
    )
    fast_tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(0)
    if architecture == "mistral":
        model = transformers.MistralForCausalLM(transformers.MistralConfig(
            vocab_size=5, max_position_embeddings=16, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, bos_token_id=1, eos_token_id=1, pad_token_id=0,
        ))
    else:
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=5, n_positions=16, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=1, eos_token_id=1, pad_token_id=0,
        ))
    model.save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "smtx-artifact.json").write_text(json.dumps({
        "source_model": "synthetic-local-test",
        "revision": "0" * 40,
        "license": "test-fixture-only",
        "architecture": architecture,
        "conversion_lineage": ["Random tiny test model; no candidate weights or capability claims."],
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp_path.iterdir() if path.is_file()
        },
    }))
    def no_network(*args, **kwargs):
        raise AssertionError("Local inference must not use the network.")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    provider_class = MistralProvider if architecture == "mistral" else LocalTransformersProvider
    provider = provider_class(str(tmp_path), device="cpu", dtype="float32")
    text, prompt_tokens, completion_tokens = provider.generate_from_messages(
        [SimpleNamespace(role="user", content="Hello")], max_new_tokens=2, temperature=0,
    )
    assert isinstance(text, str)
    assert prompt_tokens == 2
    assert 1 <= completion_tokens <= 2
    assert provider.model.device == torch.device("cpu")


@pytest.mark.parametrize("variable", ["WORLD_SIZE", "LOCAL_WORLD_SIZE"])
def test_tiny_foundation_rejects_distributed_launch(variable, monkeypatch, tmp_path):
    from supreme_modeltx.foundation.config import TrainingRunConfig
    from supreme_modeltx.foundation.training import train

    monkeypatch.setenv(variable, "2")
    cfg = TrainingRunConfig.model_validate({"training": {"output_dir": str(tmp_path / "run")}})
    with pytest.raises(ValueError, match="single process only"):
        train(cfg)
    assert not (tmp_path / "run").exists()
