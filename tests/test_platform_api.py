"""Platform API schema and integration tests."""

import pytest
from fastapi.testclient import TestClient

from supreme_modeltx.platform_api.api.app import create_app
from supreme_modeltx.platform_api.auth.tokens import TokenStore, APIKey
from supreme_modeltx.platform_api.model_registry.registry import ModelRegistry
from supreme_modeltx.platform_api.usage.meter import UsageMeter
from supreme_modeltx.platform_api.tenants.models import Tenant, TenantStore
from supreme_modeltx.platform_api.deployment.manager import DeploymentManager


# ---------------------------------------------------------------------------
# Auth / token store tests
# ---------------------------------------------------------------------------


class TestAPIKey:
    def test_create_returns_key_and_secret(self):
        key, secret = APIKey.create(tenant_id="acme")
        assert key.key_id.startswith("supmtx_")
        assert len(secret) > 16
        assert not key.revoked

    def test_valid_secret_authenticates(self):
        key, secret = APIKey.create(tenant_id="acme")
        assert key.is_valid(secret)

    def test_wrong_secret_rejected(self):
        key, _ = APIKey.create(tenant_id="acme")
        assert not key.is_valid("wrong-secret")

    def test_revoked_key_rejected(self):
        key, secret = APIKey.create(tenant_id="acme")
        key.revoke()
        assert not key.is_valid(secret)


class TestTokenStore:
    def test_issue_and_authenticate(self):
        store = TokenStore()
        key, secret = store.issue("tenant-1")
        auth = store.authenticate(key.key_id, secret)
        assert auth is not None
        assert auth.tenant_id == "tenant-1"

    def test_wrong_secret_returns_none(self):
        store = TokenStore()
        key, _ = store.issue("tenant-1")
        assert store.authenticate(key.key_id, "bad") is None

    def test_revoke_blocks_auth(self):
        store = TokenStore()
        key, secret = store.issue("tenant-1")
        store.revoke(key.key_id)
        assert store.authenticate(key.key_id, secret) is None

    def test_list_keys_by_tenant(self):
        store = TokenStore()
        store.issue("tenant-a")
        store.issue("tenant-a")
        store.issue("tenant-b")
        assert len(store.list_keys("tenant-a")) == 2
        assert len(store.list_keys("tenant-b")) == 1


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_register_and_retrieve(self):
        registry = ModelRegistry()
        record = registry.register(
            model_id="supmtx-t101-v1",
            display_name="T101 v1",
            version="1.0.0",
            base_model="t101",
        )
        assert registry.get("supmtx-t101-v1") is record

    def test_list_available_empty(self):
        registry = ModelRegistry()
        assert registry.list_available() == []

    def test_list_available_with_checkpoint(self):
        registry = ModelRegistry()
        registry.register(
            model_id="supmtx-t101-v1",
            display_name="T101",
            version="1.0.0",
            base_model="t101",
            checkpoint_path="/checkpoints/t101",
        )
        assert len(registry.list_available()) == 1

    def test_default_model(self):
        registry = ModelRegistry()
        registry.register(
            model_id="supmtx-t101-v1",
            display_name="T101",
            version="1.0.0",
            base_model="t101",
            checkpoint_path="/ckpt",
            is_default=True,
        )
        assert registry.get_default().model_id == "supmtx-t101-v1"

    def test_deprecate(self):
        registry = ModelRegistry()
        registry.register(
            "m1", "M1", "1.0", "t101", checkpoint_path="/ckpt"
        )
        assert registry.deprecate("m1")
        assert registry.list_available() == []


# ---------------------------------------------------------------------------
# Usage meter tests
# ---------------------------------------------------------------------------


class TestUsageMeter:
    def test_record_and_summarise(self):
        meter = UsageMeter()
        meter.record("tenant-1", "t101", "chat", 10, 20)
        summary = meter.summarise("tenant-1")
        assert summary["request_count"] == 1
        assert summary["total_tokens"] == 30

    def test_multiple_records(self):
        meter = UsageMeter()
        meter.record("t1", "t101", "chat", 5, 5)
        meter.record("t1", "t101", "chat", 10, 10)
        summary = meter.summarise("t1")
        assert summary["request_count"] == 2
        assert summary["total_tokens"] == 30

    def test_tenant_isolation(self):
        meter = UsageMeter()
        meter.record("t1", "t101", "chat", 10, 10)
        meter.record("t2", "t101", "chat", 5, 5)
        assert meter.summarise("t1")["total_tokens"] == 20
        assert meter.summarise("t2")["total_tokens"] == 10


# ---------------------------------------------------------------------------
# Tenant store tests
# ---------------------------------------------------------------------------


class TestTenantStore:
    def test_create_and_get(self):
        store = TenantStore()
        tenant = store.create("acme", "Acme Corp")
        assert store.get("acme") is tenant

    def test_duplicate_raises(self):
        store = TenantStore()
        store.create("acme", "Acme Corp")
        with pytest.raises(ValueError, match="already exists"):
            store.create("acme", "Acme Corp 2")

    def test_add_project(self):
        store = TenantStore()
        tenant = store.create("acme", "Acme Corp")
        project = tenant.add_project("proj-001", "My Project")
        assert tenant.get_project("proj-001") is project


# ---------------------------------------------------------------------------
# Deployment manager tests
# ---------------------------------------------------------------------------


class TestDeploymentManager:
    def test_create_deployment(self):
        mgr = DeploymentManager()
        record = mgr.create_deployment("dep-1", "supmtx-t101-v1", "tenant-1")
        assert record.status == "pending"
        assert mgr.get("dep-1") is record

    def test_update_status(self):
        mgr = DeploymentManager()
        mgr.create_deployment("dep-1", "supmtx-t101-v1", "tenant-1")
        assert mgr.update_status("dep-1", "running")
        assert mgr.get("dep-1").status == "running"

    def test_list_by_tenant(self):
        mgr = DeploymentManager()
        mgr.create_deployment("dep-1", "m1", "t1")
        mgr.create_deployment("dep-2", "m1", "t1")
        mgr.create_deployment("dep-3", "m1", "t2")
        assert len(mgr.list_by_tenant("t1")) == 2


# ---------------------------------------------------------------------------
# FastAPI endpoint smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_key():
    token_store = TokenStore()
    registry = ModelRegistry()
    registry.register(
        model_id="supmtx-t101-v1",
        display_name="T101",
        version="1.0.0",
        base_model="t101",
        checkpoint_path="/ckpt",
    )
    app = create_app(token_store=token_store, model_registry=registry)
    client = TestClient(app)
    key_resp = client.post("/v1/keys", json={"tenant_id": "acme"})
    assert key_resp.status_code == 200
    data = key_resp.json()
    auth_header = f"{data['key_id']}:{data['secret']}"
    return client, auth_header


class TestHealthEndpoint:
    def test_health(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModelsEndpoint:
    def test_list_models_requires_auth(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_list_models_authenticated(self, client_with_key):
        client, auth = client_with_key
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {auth}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1


class TestChatEndpoint:
    def test_chat_completions(self, client_with_key):
        client, auth = client_with_key
        payload = {
            "model": "supmtx-t101-v1",
            "messages": [{"role": "user", "content": "Hello, world!"}],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {auth}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["usage"]["total_tokens"] > 0

    def test_chat_unknown_model(self, client_with_key):
        client, auth = client_with_key
        payload = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {auth}"},
        )
        assert resp.status_code == 404


class TestUsageEndpoint:
    def test_usage_returns_summary(self, client_with_key):
        client, auth = client_with_key
        payload = {
            "model": "supmtx-t101-v1",
            "messages": [{"role": "user", "content": "Test usage"}],
        }
        client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {auth}"},
        )
        resp = client.get("/v1/usage", headers={"Authorization": f"Bearer {auth}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_count"] == 1
        assert data["total_tokens"] > 0
