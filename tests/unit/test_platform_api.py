"""
tests/unit/test_platform_api.py — Unit tests for platform API schemas and services.
"""
import pytest

from supreme_modeltx.platform_api.auth.keys import issue_key, verify_api_key, revoke_key
from supreme_modeltx.platform_api.auth.key_store import KeyMetadataStore
from supreme_modeltx.platform_api.model_registry.registry import ModelEntry, ModelRegistry
from supreme_modeltx.platform_api.tenants.models import Project, ProjectCreate
from supreme_modeltx.platform_api.tenants.store import ProjectStore
from supreme_modeltx.platform_api.usage.metering import UsageEvent, UsageLedger
from supreme_modeltx.platform_api.deployment.service import ComputeSpec, DeploymentService
from supreme_modeltx.platform_api.audit.log import AuditEvent, AuditLog
from supreme_modeltx.platform_api.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ResponsesRequest,
    ResponsesResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    KeyIssueRequest,
    KeyIssueResponse,
    KeyMetadata,
)


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


class TestAuditLog:
    def test_empty_query(self):
        log = AuditLog()
        events = log.query()
        assert events == []

    def test_record_and_query(self):
        log = AuditLog()
        log.record(AuditEvent(project_id="proj-a", event_type="chat.completion", model="t-dev-6l"))
        log.record(AuditEvent(project_id="proj-b", event_type="embeddings", model="t-dev-6l"))
        all_events = log.query()
        assert len(all_events) == 2

    def test_filter_by_project(self):
        log = AuditLog()
        log.record(AuditEvent(project_id="proj-a", event_type="chat.completion"))
        log.record(AuditEvent(project_id="proj-b", event_type="chat.completion"))
        a_events = log.query(project_id="proj-a")
        assert len(a_events) == 1
        assert a_events[0].project_id == "proj-a"

    def test_filter_by_event_type(self):
        log = AuditLog()
        log.record(AuditEvent(project_id="proj-a", event_type="chat.completion"))
        log.record(AuditEvent(project_id="proj-a", event_type="embeddings"))
        chat_events = log.query(event_type="chat.completion")
        assert len(chat_events) == 1
        assert chat_events[0].event_type == "chat.completion"

    def test_limit(self):
        log = AuditLog()
        for i in range(10):
            log.record(AuditEvent(project_id="proj-a", event_type="chat.completion"))
        limited = log.query(limit=3)
        assert len(limited) == 3

    def test_newest_first(self):
        from datetime import datetime, timezone, timedelta
        log = AuditLog()
        early = AuditEvent(project_id="p", event_type="e1")
        late = AuditEvent(project_id="p", event_type="e2")
        log.record(early)
        log.record(late)
        events = log.query()
        # newest first — last recorded should be first
        assert events[0].event_type == "e2"


class TestKeyMetadataStore:
    def _store(self):
        from datetime import datetime, timezone
        store = KeyMetadataStore()
        store.register(
            key_id="key-001",
            project_id="proj-a",
            label="prod-key",
            key_prefix="abcd1234",
            created_at=datetime.now(timezone.utc),
        )
        return store

    def test_register_and_get(self):
        store = self._store()
        meta = store.get_by_id("key-001")
        assert meta is not None
        assert meta.project_id == "proj-a"
        assert meta.label == "prod-key"
        assert meta.key_prefix == "abcd1234"

    def test_get_nonexistent(self):
        store = KeyMetadataStore()
        assert store.get_by_id("no-such-key") is None

    def test_list_all(self):
        store = self._store()
        keys = store.list_keys()
        assert len(keys) == 1

    def test_list_by_project(self):
        from datetime import datetime, timezone
        store = self._store()
        store.register(
            key_id="key-002",
            project_id="proj-b",
            label="other-key",
            key_prefix="zxzxzxzx",
            created_at=datetime.now(timezone.utc),
        )
        a_keys = store.list_keys(project_id="proj-a")
        assert len(a_keys) == 1
        assert a_keys[0].key_id == "key-001"

    def test_remove(self):
        store = self._store()
        assert store.remove("key-001") is True
        assert store.get_by_id("key-001") is None

    def test_remove_nonexistent(self):
        store = KeyMetadataStore()
        assert store.remove("ghost") is False


class TestNewSchemas:
    def test_responses_request_plain_text(self):
        req = ResponsesRequest(model="t-dev-6l", input="Hello")
        assert req.input == "Hello"
        assert req.max_output_tokens == 256

    def test_responses_request_messages(self):
        req = ResponsesRequest(
            model="t-dev-6l",
            input=[ChatMessage(role="user", content="Hi")],
        )
        assert isinstance(req.input, list)

    def test_embeddings_request_string(self):
        req = EmbeddingsRequest(model="t-dev-6l", input="Hello world")
        assert req.input == "Hello world"

    def test_embeddings_request_list(self):
        req = EmbeddingsRequest(model="t-dev-6l", input=["Hello", "World"])
        assert len(req.input) == 2

    def test_key_issue_request(self):
        req = KeyIssueRequest(project_id="proj-x", label="my-key")
        assert req.project_id == "proj-x"
        assert req.label == "my-key"

    def test_chat_response_has_object_field(self):
        resp = ChatResponse(
            id="test-1",
            model="t-dev-6l",
            choices=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        assert resp.object == "chat.completion"
