"""
tests/unit/test_platform_api.py — Unit tests for platform API schemas and services.
"""
import pytest

from supreme_modeltx.platform_api.auth.keys import issue_key, verify_api_key, revoke_key
from supreme_modeltx.platform_api.model_registry.registry import ModelEntry, ModelRegistry
from supreme_modeltx.platform_api.tenants.models import Project, ProjectCreate
from supreme_modeltx.platform_api.tenants.store import ProjectStore
from supreme_modeltx.platform_api.usage.metering import UsageEvent, UsageLedger
from supreme_modeltx.platform_api.deployment.service import ComputeSpec, DeploymentService


class TestApiKeys:
    def test_verify_dev_key(self):
        # The dev key is seeded from env (default: "dev-secret")
        import os
        dev_key = os.environ.get("SMTX_API_KEY", "dev-secret")
        project_id = verify_api_key(dev_key)
        assert project_id is not None

    def test_invalid_key_returns_none(self):
        assert verify_api_key("this-is-not-a-valid-key-xyz") is None

    def test_issue_and_verify(self):
        key = issue_key("test-project-123")
        assert len(key) > 0
        result = verify_api_key(key)
        assert result == "test-project-123"

    def test_revoke(self):
        key = issue_key("revoke-test-project")
        assert verify_api_key(key) == "revoke-test-project"
        assert revoke_key(key) is True
        assert verify_api_key(key) is None

    def test_revoke_nonexistent(self):
        assert revoke_key("nonexistent-key") is False


class TestModelRegistry:
    def test_default_entries(self):
        registry = ModelRegistry()
        models = registry.list_models()
        assert len(models) >= 1
        ids = [m.id for m in models]
        assert "t-dev-6l" in ids

    def test_get_existing(self):
        registry = ModelRegistry()
        entry = registry.get_model("t-dev-6l")
        assert entry is not None
        assert entry.family == "t-series"

    def test_get_nonexistent(self):
        registry = ModelRegistry()
        assert registry.get_model("does-not-exist") is None

    def test_register_and_get(self):
        registry = ModelRegistry()
        entry = ModelEntry(
            id="test-model",
            name="Test Model",
            variant="test",
            stage="development",
        )
        registry.register(entry)
        fetched = registry.get_model("test-model")
        assert fetched is not None
        assert fetched.name == "Test Model"

    def test_deregister(self):
        registry = ModelRegistry()
        entry = ModelEntry(id="to-remove", name="Remove Me", variant="remove")
        registry.register(entry)
        assert registry.deregister("to-remove") is True
        assert registry.get_model("to-remove") is None


class TestProjectStore:
    def test_dev_project_seeded(self):
        store = ProjectStore()
        projects = store.list_projects()
        assert any(p.id == "dev-project" for p in projects)

    def test_create_and_get(self):
        store = ProjectStore()
        body = ProjectCreate(name="My Project", description="Test", owner_email="test@example.com")
        created = store.create_project(body)
        assert created.name == "My Project"
        fetched = store.get_project(created.id)
        assert fetched is not None
        assert fetched.owner_email == "test@example.com"

    def test_get_nonexistent(self):
        store = ProjectStore()
        assert store.get_project("nonexistent-xyz") is None


class TestUsageLedger:
    def test_empty_summary(self):
        ledger = UsageLedger()
        summary = ledger.summarise("no-events-project")
        assert summary.total_requests == 0
        assert summary.total_prompt_tokens == 0

    def test_record_and_summarise(self):
        ledger = UsageLedger()
        ledger.record(UsageEvent(project_id="proj-a", model_id="t-dev-6l", prompt_tokens=10, completion_tokens=20))
        ledger.record(UsageEvent(project_id="proj-a", model_id="t-dev-6l", prompt_tokens=5, completion_tokens=15))
        ledger.record(UsageEvent(project_id="proj-b", model_id="t-dev-6l", prompt_tokens=100, completion_tokens=50))
        s = ledger.summarise("proj-a")
        assert s.total_requests == 2
        assert s.total_prompt_tokens == 15
        assert s.total_completion_tokens == 35
        # proj-b not included
        sb = ledger.summarise("proj-b")
        assert sb.total_requests == 1


class TestDeploymentService:
    def test_create_deployment(self):
        svc = DeploymentService()
        dep = svc.create("t-dev-6l", "dev-project")
        assert dep.model_id == "t-dev-6l"
        assert dep.status == "pending"
        assert dep.id is not None

    def test_get_deployment(self):
        svc = DeploymentService()
        dep = svc.create("t-dev-6l", "dev-project")
        fetched = svc.get(dep.id)
        assert fetched is not None
        assert fetched.id == dep.id

    def test_stop_deployment(self):
        svc = DeploymentService()
        dep = svc.create("t-dev-6l", "dev-project")
        assert svc.stop(dep.id) is True
        assert svc.get(dep.id).status == "stopped"

    def test_list_for_project(self):
        svc = DeploymentService()
        svc.create("t-dev-6l", "project-alpha")
        svc.create("t-dev-6l", "project-alpha")
        svc.create("t-dev-6l", "project-beta")
        alpha = svc.list_for_project("project-alpha")
        assert len(alpha) == 2
