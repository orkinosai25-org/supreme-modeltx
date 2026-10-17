"""Opt-in qualification of an operator-approved snapshot, never a download."""

import hashlib
import json
import os
from pathlib import Path
import re
import socket

import pytest


def test_approved_mistral_snapshot_api(tmp_path, monkeypatch, record_property):
    snapshot_path = os.environ.get("SMTX_MISTRAL_VALIDATION_PATH")
    if not snapshot_path:
        pytest.skip("Requires an approved local snapshot; synthetic tests are not qualification.")
    root = Path(snapshot_path)
    assert root.is_absolute() and root.is_dir(), "Supply an absolute approved local snapshot path."
    expected = os.environ.get("SMTX_MISTRAL_VALIDATION_MANIFEST_SHA256", "")
    assert re.fullmatch(r"[0-9a-f]{64}", expected), "Pin the independently approved manifest SHA-256."
    assert hashlib.sha256((root / "smtx-artifact.json").read_bytes()).hexdigest() == expected

    from fastapi.testclient import TestClient
    from supreme_modeltx.model_core.inference.providers import MistralProvider
    from supreme_modeltx.platform_api.api import engine
    from supreme_modeltx.platform_api.api.app import create_app
    from supreme_modeltx.platform_api.api.routers.auth import require_api_key

    def no_network(*args, **kwargs):
        raise AssertionError("Snapshot qualification must not access the network.")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(engine, "_engine_backend", None)
    monkeypatch.setenv("SMTX_INFERENCE_BACKEND", "mistral")
    monkeypatch.setenv("SMTX_PRETRAINED_MODEL_PATH", str(root))
    monkeypatch.setenv("SMTX_MODEL_ID", "qualified-mistral")
    monkeypatch.setenv("SUPREME_MODELTX_PLATFORM_DB_PATH", str(tmp_path / "registry.sqlite3"))
    monkeypatch.setenv("SUPREME_MODELTX_DEVICE", os.environ.get("SMTX_MISTRAL_VALIDATION_DEVICE", "cpu"))
    monkeypatch.setenv("SMTX_INFERENCE_DTYPE", os.environ.get("SMTX_MISTRAL_VALIDATION_DTYPE", "float32"))
    app = create_app()
    app.dependency_overrides[require_api_key] = lambda: "qualification-project"
    with TestClient(app) as client:
        assert client.get("/health/ready").json() == {"status": "ready", "backend": "mistral"}
        provider = engine.get_engine()
        assert isinstance(provider, MistralProvider)
        record_property("manifest_sha256", expected)
        record_property("provider", json.dumps(provider.metadata))
        response = client.post("/v1/chat/completions", json={
            "model": "qualified-mistral",
            "messages": [{"role": "user", "content": "Merhaba, kendini bir cümleyle tanıt."}],
            "temperature": 0, "max_tokens": 32,
        })
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["model"] == "qualified-mistral"
        assert result["choices"][0]["message"]["content"].strip()
        assert result["usage"]["prompt_tokens"] > 0
        assert 1 <= result["usage"]["completion_tokens"] <= 32
        assert result["usage"]["total_tokens"] == (
            result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
        )
        record_property("chat_smoke", json.dumps(result, ensure_ascii=False))
