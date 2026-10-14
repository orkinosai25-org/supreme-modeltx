"""Regression checks for the Azure code deployment contract."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_azure_installs_api_package_only():
    requirements = ROOT / "deployment/azure/requirements.txt"
    assert requirements.read_text().strip() == ".[api]"

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/main_sumotx2.yml").read_text()
    )
    build = workflow["jobs"]["build"]["steps"]
    setup = next(step for step in build if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == "3.11"
    package = next(step for step in build if step.get("name") == "Prepare API-only deployment package")
    for filename in ("pyproject.toml", "README.md", "LICENSE", "src", "deployment/azure/requirements.txt"):
        assert filename in package["run"]
    deploy = workflow["jobs"]["deploy"]
    assert "refs/heads/main" in deploy["if"]
    assert "pull_request" in deploy["if"]
    assert deploy["permissions"]["id-token"] == "write"


def test_azure_startup_loads_installed_api(tmp_path):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/main_sumotx2.yml").read_text()
    )
    configure = next(
        step for step in workflow["jobs"]["deploy"]["steps"]
        if step.get("name") == "Configure Python runtime and persistent MVP storage"
    )
    arguments = shlex.split(configure["run"])
    startup = shlex.split(arguments[arguments.index("--startup-file") + 1])
    assert startup[:3] == ["python", "-m", "uvicorn"]
    assert "--factory" in startup
    assert startup[startup.index("--host") + 1] == "0.0.0.0"
    assert startup[startup.index("--port") + 1] == "8000"
    assert arguments[arguments.index("--linux-fx-version") + 1] == "PYTHON|3.11"

    # Resolve the factory outside the checkout, as in the Oryx virtual environment.
    factory = startup[3]
    env = os.environ.copy()
    env["SUPREME_MODELTX_PLATFORM_DB_PATH"] = str(tmp_path / "platform.sqlite3")
    result = subprocess.run(
        [sys.executable, "-c", f"""
from fastapi.testclient import TestClient
from uvicorn.importer import import_from_string
app = import_from_string({factory!r})()
with TestClient(app) as client:
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {{"status": "ok", "version": "0.1.0"}}
"""],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("settings", "allowed"),
    [
        ({}, False),
        ({"SMTX_API_KEY": "dev-secret", "SMTX_KEY_SALT": "test-only-salt"}, False),
        ({"SMTX_API_KEY": "test-only-key", "SMTX_KEY_SALT": "supreme-modeltx-default-dev-salt-2026"}, False),
        ({"SMTX_API_KEY": "test-only-key", "SMTX_KEY_SALT": "test-only-salt"}, True),
        ({"SUPREME_MODELTX_API_KEY": "test-only-key", "SUPREME_MODELTX_KEY_SALT": "test-only-salt"}, True),
        ({"SUPREME_MODELTX_API_KEY": "", "SMTX_API_KEY": "test-only-key", "SMTX_KEY_SALT": "test-only-salt"}, False),
    ],
)
def test_azure_rejects_development_authentication(settings, allowed):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/main_sumotx2.yml").read_text()
    )
    validate = next(
        step for step in workflow["jobs"]["deploy"]["steps"]
        if step.get("name") == "Validate production authentication settings"
    )
    arguments = shlex.split(validate["run"])
    script = arguments[arguments.index("-c") + 1]
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps([{"name": name, "value": value} for name, value in settings.items()]),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is allowed
    assert result.stdout == ""
    if not allowed:
        assert "Configure a non-development" in result.stderr
