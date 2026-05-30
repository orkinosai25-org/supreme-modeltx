"""platform_api/auth/key_store.py — SQLite-backed key metadata store."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from supreme_modeltx.platform_api.api.schemas import KeyMetadata
from supreme_modeltx.platform_api.persistence.sqlite import connect, resolve_db_path


class KeyMetadataStore:
    """SQLite-backed store for API key metadata."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = resolve_db_path(db_path)
        self._initialize()

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
        with connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO key_metadata (key_id, project_id, label, key_prefix, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (meta.key_id, meta.project_id, meta.label, meta.key_prefix, meta.created_at.isoformat()),
            )
            conn.commit()
        return meta

    def get_by_id(self, key_id: str) -> Optional[KeyMetadata]:
        with connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT key_id, project_id, label, key_prefix, created_at
                FROM key_metadata
                WHERE key_id = ?
                """,
                (key_id,),
            ).fetchone()
        return self._row_to_metadata(row) if row else None

    def list_keys(self, project_id: Optional[str] = None) -> list[KeyMetadata]:
        query = """
            SELECT key_id, project_id, label, key_prefix, created_at
            FROM key_metadata
        """
        params: tuple[str, ...] = ()
        if project_id is not None:
            query += " WHERE project_id = ?"
            params = (project_id,)
        query += " ORDER BY created_at DESC"
        with connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_metadata(row) for row in rows]

    def remove(self, key_id: str) -> bool:
        with connect(self._db_path) as conn:
            result = conn.execute("DELETE FROM key_metadata WHERE key_id = ?", (key_id,))
            conn.commit()
        return result.rowcount > 0

    def _initialize(self) -> None:
        with connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS key_metadata (
                    key_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _row_to_metadata(row: tuple[str, str, str, str, str]) -> KeyMetadata:
        return KeyMetadata(
            key_id=row[0],
            project_id=row[1],
            label=row[2],
            key_prefix=row[3],
            created_at=row[4],
        )


_GLOBAL_KEY_STORE = KeyMetadataStore()


def get_key_store() -> KeyMetadataStore:
    """Return the module-level key metadata store singleton."""
    return _GLOBAL_KEY_STORE
