"""API key and bearer-token management for the platform API."""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from typing import Optional

from pydantic import BaseModel, Field

_PBKDF2_ITERATIONS = 260_000
_SALT_BYTES = 16


class APIKey(BaseModel):
    """Represents an issued API key."""

    key_id: str = Field(description="Opaque key identifier.")
    tenant_id: str = Field(description="Owning tenant.")
    project_id: Optional[str] = Field(default=None, description="Scoped project, if any.")
    hashed_secret: str = Field(description="PBKDF2-HMAC-SHA256 hash of the raw secret (hex salt:hash).")
    created_at: float = Field(default_factory=time.time)
    expires_at: Optional[float] = Field(default=None)
    revoked: bool = Field(default=False)

    @classmethod
    def create(cls, tenant_id: str, project_id: str | None = None) -> tuple["APIKey", str]:
        """Issue a new API key.

        Returns:
            Tuple of (APIKey record, raw secret). The raw secret is shown
            once and must be stored securely by the caller.
        """
        raw_secret = secrets.token_urlsafe(32)
        salt = os.urandom(_SALT_BYTES)
        dk = hashlib.pbkdf2_hmac("sha256", raw_secret.encode(), salt, _PBKDF2_ITERATIONS)
        hashed = salt.hex() + ":" + dk.hex()
        key = cls(
            key_id=f"smtx_{secrets.token_hex(8)}",
            tenant_id=tenant_id,
            project_id=project_id,
            hashed_secret=hashed,
        )
        return key, raw_secret

    def is_valid(self, raw_secret: str) -> bool:
        """Validate a raw secret against the stored hash."""
        if self.revoked:
            return False
        if self.expires_at is not None and time.time() > self.expires_at:
            return False
        try:
            salt_hex, dk_hex = self.hashed_secret.split(":", 1)
        except ValueError:
            return False
        salt = bytes.fromhex(salt_hex)
        candidate = hashlib.pbkdf2_hmac("sha256", raw_secret.encode(), salt, _PBKDF2_ITERATIONS)
        return secrets.compare_digest(candidate.hex(), dk_hex)

    def revoke(self) -> None:
        self.revoked = True


class TokenStore:
    """In-memory store for API keys (replace with persistent backend in production)."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}

    def issue(self, tenant_id: str, project_id: str | None = None) -> tuple[APIKey, str]:
        key, raw_secret = APIKey.create(tenant_id, project_id)
        self._keys[key.key_id] = key
        return key, raw_secret

    def lookup(self, key_id: str) -> APIKey | None:
        return self._keys.get(key_id)

    def authenticate(self, key_id: str, raw_secret: str) -> APIKey | None:
        key = self._keys.get(key_id)
        if key is None or not key.is_valid(raw_secret):
            return None
        return key

    def revoke(self, key_id: str) -> bool:
        key = self._keys.get(key_id)
        if key is None:
            return False
        key.revoke()
        return True

    def list_keys(self, tenant_id: str) -> list[APIKey]:
        return [k for k in self._keys.values() if k.tenant_id == tenant_id]
