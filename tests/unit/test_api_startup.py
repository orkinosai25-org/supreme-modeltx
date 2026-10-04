"""Regression tests for API startup without optional model dependencies."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("configure_inference", [False, True])
def test_health_without_pytorch(tmp_path, configure_inference):
    env = os.environ.copy()
    env["SUPREME_MODELTX_PLATFORM_DB_PATH"] = str(tmp_path / "platform.sqlite3")
    env["SUPREME_MODELTX_API_KEY"] = "test-only-key"
    for name in ("SMTX_CHECKPOINT_PATH", "SMTX_TOKENIZER_PATH"):
        env.pop(name, None)
        if configure_inference:
            artifact = tmp_path / name
            artifact.touch()
            env[name] = str(artifact)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import os
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("PyTorch is intentionally unavailable")

sys.meta_path.insert(0, BlockTorch())

from fastapi.testclient import TestClient
from supreme_modeltx.platform_api.api.app import create_app

with TestClient(create_app()) as client:
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["SUPREME_MODELTX_API_KEY"]},
        json={"model": "t-dev-6l", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 503, response.text

assert "torch" not in sys.modules
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
