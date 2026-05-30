"""
tests/unit/test_chat_endpoint.py — Unit tests for the wired /v1/chat/completions endpoint.

Tests cover:
  1. 503 when the inference backend is not configured.
  2. Real inference via a mocked _InferenceBackend (no GPU / checkpoint required).
  3. 500 propagation when the backend raises an unexpected error.
  4. _format_prompt helper produces the expected prompt string.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from supreme_modeltx.platform_api.api import engine as engine_module
from supreme_modeltx.platform_api.api.engine import _format_prompt, _InferenceBackend
from supreme_modeltx.platform_api.api.routers import chat as chat_router_module
from supreme_modeltx.platform_api.api.routers import models as models_router_module
from supreme_modeltx.platform_api.api.schemas import ChatMessage
from supreme_modeltx.platform_api.audit.log import AuditLog
from supreme_modeltx.platform_api.model_registry.registry import ModelEntry, ModelRegistry
from supreme_modeltx.platform_api.usage.metering import UsageLedger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client() -> TestClient:
    """Create a TestClient for the platform API app."""
    from supreme_modeltx.platform_api.api.app import create_app
    return TestClient(create_app(), raise_server_exceptions=False)


_AUTH_HEADERS = {"Authorization": "Bearer dev-secret"}


# ── Tests: _format_prompt ─────────────────────────────────────────────────────

class TestFormatPrompt:
    def test_single_user_message(self):
        msgs = [ChatMessage(role="user", content="Hello!")]
        prompt = _format_prompt(msgs)
        assert "user: Hello!" in prompt
        assert prompt.endswith("\nassistant:")

    def test_system_and_user(self):
        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
        ]
        prompt = _format_prompt(msgs)
        assert "system: You are helpful." in prompt
        assert "user: Hi" in prompt
        assert prompt.endswith("\nassistant:")

    def test_empty_messages(self):
        prompt = _format_prompt([])
        assert prompt == "\nassistant:"


# ── Tests: HTTP endpoint ───────────────────────────────────────────────────────

class TestChatCompletionsEndpoint:
    """HTTP-level tests for POST /v1/chat/completions."""

    def test_503_when_no_engine(self, monkeypatch):
        """Endpoint returns 503 when no inference backend is configured."""
        monkeypatch.setattr(engine_module, "_engine_backend", None)
        client = _make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "t-dev-6l", "messages": [{"role": "user", "content": "Hi"}]},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 503
        assert "SMTX_CHECKPOINT_PATH" in resp.json()["detail"]

    def test_returns_completion_with_mock_engine(self, monkeypatch):
        """Endpoint returns a proper ChatResponse when a backend is active."""

        class _MockBackend:
            def generate_from_messages(self, messages, max_new_tokens, temperature, top_p):
                return "Hello there!", 5, 2

        monkeypatch.setattr(engine_module, "_engine_backend", _MockBackend())
        client = _make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "t-dev-6l", "messages": [{"role": "user", "content": "Hey"}]},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "t-dev-6l"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Hello there!"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 5
        assert body["usage"]["completion_tokens"] == 2
        assert body["usage"]["total_tokens"] == 7
        assert body["id"].startswith("chatcmpl-")

    def test_500_when_backend_raises(self, monkeypatch):
        """Endpoint returns 500 when the backend raises an exception."""

        class _BrokenBackend:
            def generate_from_messages(self, *args, **kwargs):
                raise RuntimeError("GPU OOM")

        monkeypatch.setattr(engine_module, "_engine_backend", _BrokenBackend())
        client = _make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "t-dev-6l", "messages": [{"role": "user", "content": "Crash?"}]},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 500
        assert "GPU OOM" in resp.json()["detail"]

    def test_401_without_api_key(self, monkeypatch):
        """Endpoint returns 401 when no API key is supplied."""
        monkeypatch.setattr(engine_module, "_engine_backend", None)
        client = _make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "t-dev-6l", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 401

    def test_generation_params_forwarded(self, monkeypatch):
        """max_tokens / temperature / top_p from the request reach the backend."""
        captured: dict = {}

        class _CapturingBackend:
            def generate_from_messages(self, messages, max_new_tokens, temperature, top_p):
                captured.update(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return "ok", 1, 1

        monkeypatch.setattr(engine_module, "_engine_backend", _CapturingBackend())
        client = _make_client()
        client.post(
            "/v1/chat/completions",
            json={
                "model": "t-dev-6l",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 64,
                "temperature": 0.3,
                "top_p": 0.85,
            },
            headers=_AUTH_HEADERS,
        )
        assert captured["max_new_tokens"] == 64
        assert captured["temperature"] == pytest.approx(0.3)
        assert captured["top_p"] == pytest.approx(0.85)

    def test_persists_usage_and_audit_for_successful_completion(self, monkeypatch, tmp_path):
        class _MockBackend:
            def generate_from_messages(self, messages, max_new_tokens, temperature, top_p):
                return "Hello there!", 5, 2

        usage_ledger = UsageLedger(db_path=str(tmp_path / "usage.sqlite3"))
        audit_log = AuditLog(db_path=str(tmp_path / "audit.sqlite3"))
        monkeypatch.setattr(chat_router_module, "_usage_ledger", usage_ledger)
        monkeypatch.setattr(chat_router_module, "_audit_log", audit_log)
        monkeypatch.setattr(engine_module, "_engine_backend", _MockBackend())

        client = _make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "t-dev-6l", "messages": [{"role": "user", "content": "Hey"}]},
            headers=_AUTH_HEADERS,
        )

        assert resp.status_code == 200
        usage_summary = usage_ledger.summarise(project_id="dev-project")
        assert usage_summary.total_prompt_tokens == 5
        assert usage_summary.total_completion_tokens == 2
        assert usage_summary.total_requests == 1

        audit_events = audit_log.query(project_id="dev-project", event_type="chat.completion")
        assert len(audit_events) == 1
        assert audit_events[0].event_type == "chat.completion"
        assert audit_events[0].model == "t-dev-6l"

    def test_models_endpoint_exposes_checkpoint_metadata(self, monkeypatch, tmp_path):
        db_path = str(tmp_path / "models.sqlite3")
        registry = ModelRegistry(db_path=db_path)
        registry.register(
            ModelEntry(
                id="served-real",
                name="Served Real",
                variant="t-real",
                stage="production",
                checkpoint_path="/models/served-real.pt",
                tokenizer_path="/models/tokenizer.model",
                inference_dtype="float16",
            )
        )
        monkeypatch.setattr(models_router_module, "_registry", registry)
        client = _make_client()
        resp = client.get("/v1/models/", headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        served = next(model for model in resp.json() if model["id"] == "served-real")
        assert served["checkpoint_path"] == "/models/served-real.pt"
        assert served["tokenizer_path"] == "/models/tokenizer.model"
        assert served["inference_dtype"] == "float16"


# ── Tests: initialize_engine no-ops ───────────────────────────────────────────

class TestInitializeEngine:
    def test_no_env_vars_leaves_engine_none(self, monkeypatch):
        """initialize_engine is a no-op when env vars are absent."""
        monkeypatch.setattr(engine_module, "_engine_backend", None)
        monkeypatch.delenv("SMTX_CHECKPOINT_PATH", raising=False)
        monkeypatch.delenv("SMTX_TOKENIZER_PATH", raising=False)
        engine_module.initialize_engine()
        assert engine_module.get_engine() is None

    def test_missing_checkpoint_leaves_engine_none(self, monkeypatch, tmp_path):
        """initialize_engine is a no-op when the checkpoint file is absent."""
        monkeypatch.setattr(engine_module, "_engine_backend", None)
        monkeypatch.setenv("SMTX_CHECKPOINT_PATH", str(tmp_path / "no_such.pt"))
        monkeypatch.setenv("SMTX_TOKENIZER_PATH", str(tmp_path / "tok.model"))
        engine_module.initialize_engine()
        assert engine_module.get_engine() is None

    def test_idempotent_when_already_loaded(self, monkeypatch):
        """initialize_engine does not overwrite an already-loaded backend."""

        class _Sentinel:
            pass

        sentinel = _Sentinel()
        monkeypatch.setattr(engine_module, "_engine_backend", sentinel)
        engine_module.initialize_engine()
        assert engine_module._engine_backend is sentinel

    def test_registers_loaded_checkpoint_in_model_registry(self, monkeypatch, tmp_path):
        db_path = str(tmp_path / "platform.sqlite3")
        checkpoint = tmp_path / "model.pt"
        tokenizer = tmp_path / "tokenizer.model"
        checkpoint.write_bytes(b"weights")
        tokenizer.write_text("tok")

        class _MockInferenceEngine:
            def __init__(self, model_config, checkpoint_path, dtype):
                self.model_config = model_config
                self.checkpoint_path = checkpoint_path
                self.dtype = dtype

            def generate(self, *args, **kwargs):
                return None

        class _MockTokenizerWorkflow:
            def __init__(self, path):
                self.path = path

            def encode(self, text):
                return [1]

            def decode(self, ids):
                return "ok"

        import supreme_modeltx.model_core.inference.engine as inference_engine_module
        import supreme_modeltx.model_core.tokenizer.workflow as tokenizer_workflow_module

        monkeypatch.setattr(engine_module, "_engine_backend", None)
        monkeypatch.setenv("SUPREME_MODELTX_PLATFORM_DB_PATH", db_path)
        monkeypatch.setenv("SMTX_CHECKPOINT_PATH", str(checkpoint))
        monkeypatch.setenv("SMTX_TOKENIZER_PATH", str(tokenizer))
        monkeypatch.setenv("SMTX_MODEL_ID", "runtime-real-model")
        monkeypatch.setenv("SMTX_MODEL_NAME", "Runtime Real Model")
        monkeypatch.setattr(inference_engine_module, "InferenceEngine", _MockInferenceEngine)
        monkeypatch.setattr(tokenizer_workflow_module, "TokenizerWorkflow", _MockTokenizerWorkflow)

        engine_module.initialize_engine()
        entry = ModelRegistry(db_path=db_path).get_model("runtime-real-model")
        assert entry is not None
        assert entry.checkpoint_path == str(checkpoint.resolve())
        assert entry.tokenizer_path == str(tokenizer.resolve())
        assert entry.is_available is True
