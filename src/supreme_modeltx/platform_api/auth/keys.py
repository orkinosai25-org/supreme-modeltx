"""
platform_api/auth/keys.py — API key management.

In production this should be backed by a secure key store (e.g. a database
with hashed keys).  For scaffolding, we use a simple in-memory registry
seeded from the environment.

Security notes:
  - Keys are derived using hashlib.scrypt (computationally expensive) to
    resist offline brute-force attacks if the key store is ever exfiltrated.
  - A fixed, deterministic salt is used per-deployment
    (set SUPREME_MODELTX_KEY_SALT; legacy SMTX_KEY_SALT also supported).
    In production, use a per-key random salt stored alongside the hash.
  - Keys are compared with hmac.compare_digest to resist timing attacks.
  - Never log API key values.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from supreme_modeltx.platform_api.persistence.sqlite import connect, resolve_db_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scrypt parameters — must be consistent across a deployment.
# N=2^14, r=8, p=1 is a reasonable minimum; increase N for higher security.
# ---------------------------------------------------------------------------
_SCRYPT_N = 1 << 14  # 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_KEYLEN = 32

# Per-deployment salt (in production: unique per instance, stored securely).
_SALT = os.environ.get(
    "SUPREME_MODELTX_KEY_SALT",
    os.environ.get("SMTX_KEY_SALT", "supreme-modeltx-default-dev-salt-2026"),
).encode()

_DB_PATH = resolve_db_path()


def _hash_key(key: str) -> str:
    """Derive a strong hash of *key* using scrypt."""
    derived = hashlib.scrypt(
        key.encode(),
        salt=_SALT,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_KEYLEN,
    )
    return derived.hex()


def _seed_from_env() -> None:
    """Seed dev key from environment (development only)."""
    env_key = os.environ.get("SUPREME_MODELTX_API_KEY", os.environ.get("SMTX_API_KEY", "dev-secret"))
    project_id = os.environ.get(
        "SUPREME_MODELTX_DEV_PROJECT_ID",
        os.environ.get("SMTX_DEV_PROJECT_ID", "dev-project"),
    )
    with connect(_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO api_keys (key_hash, project_id, created_at)
            VALUES (?, ?, ?)
            """,
            (_hash_key(env_key), project_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    logger.debug("Dev API key seeded for project: %s", project_id)


_seed_from_env()


def issue_key(project_id: str) -> str:
    """Issue a new random API key for *project_id*.

    Returns the plain-text key (shown once; store securely).
    """
    key = f"supmtx_{secrets.token_hex(32)}"
    with connect(_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO api_keys (key_hash, project_id, created_at)
            VALUES (?, ?, ?)
            """,
            (_hash_key(key), project_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    logger.info("Issued API key for project: %s", project_id)
    return key


def verify_api_key(key: str) -> Optional[str]:
    """Return the project_id associated with *key*, or None if invalid."""
    key_hash = _hash_key(key)
    with connect(_DB_PATH) as conn:
        rows = conn.execute("SELECT key_hash, project_id FROM api_keys").fetchall()
    # Use constant-time comparison to avoid timing attacks.
    for stored_hash, project_id in rows:
        if hmac.compare_digest(stored_hash, key_hash):
            return project_id
    return None


def revoke_key(key: str) -> bool:
    """Remove a key from the store. Returns True if it existed."""
    key_hash = _hash_key(key)
    with connect(_DB_PATH) as conn:
        result = conn.execute("DELETE FROM api_keys WHERE key_hash = ?", (key_hash,))
        conn.commit()
    return result.rowcount > 0
