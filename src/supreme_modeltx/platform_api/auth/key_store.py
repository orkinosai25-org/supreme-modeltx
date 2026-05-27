"""
platform_api/auth/key_store.py — In-memory key metadata store.

Stores human-readable metadata for issued API keys (id, label, prefix, project).
Does NOT store the plain-text key or the scrypt hash — those live in keys.py.

Production: replace with a database table.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from supreme_modeltx.platform_api.api.schemas import KeyMetadata


class KeyMetadataStore:
    """In-memory store for API key metadata."""

    def __init__(self) -> None:
        # key_id → KeyMetadata
        self._store: dict[str, KeyMetadata] = {}

    def register(
        self,
        *,
        key_id: str,
        project_id: str,
        label: str,
        key_prefix: str,
        created_at: datetime,
    ) -> KeyMetadata:
        meta = KeyMetadata(
            key_id=key_id,
            project_id=project_id,
            label=label,
            key_prefix=key_prefix,
            created_at=created_at,
        )
        self._store[key_id] = meta
        return meta

    def get_by_id(self, key_id: str) -> Optional[KeyMetadata]:
        return self._store.get(key_id)

    def list_keys(self, project_id: Optional[str] = None) -> list[KeyMetadata]:
        keys = list(self._store.values())
        if project_id is not None:
            keys = [k for k in keys if k.project_id == project_id]
        return keys

    def remove(self, key_id: str) -> bool:
        if key_id in self._store:
            del self._store[key_id]
            return True
        return False


_GLOBAL_KEY_STORE = KeyMetadataStore()


def get_key_store() -> KeyMetadataStore:
    """Return the module-level key metadata store singleton."""
    return _GLOBAL_KEY_STORE
