"""
platform_api/auth/keys.py — API key management.

In production this should be backed by a secure key store (e.g. a database
with hashed keys).  For scaffolding, we use a simple in-memory registry
seeded from the environment.

Security notes:
  - Keys are derived using hashlib.scrypt (computationally expensive) to
    resist offline brute-force attacks if the key store is ever exfiltrated.
  - A fixed, deterministic salt is used per-deployment (set SMTX_KEY_SALT).
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
from typing import Optional

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
_SALT = os.environ.get("SMTX_KEY_SALT", "smtx-default-dev-salt-2025").encode()

# ---------------------------------------------------------------------------
# In-memory key store (replace with DB-backed store in production)
# ---------------------------------------------------------------------------
# Maps scrypt_hex(key) → project_id
_KEY_STORE: dict[str, str] = {}


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
    """Seed dev key from SMTX_API_KEY env var (development only)."""
    env_key = os.environ.get("SMTX_API_KEY", "dev-secret")
    project_id = os.environ.get("SMTX_DEV_PROJECT_ID", "dev-project")
    _KEY_STORE[_hash_key(env_key)] = project_id
    logger.debug("Dev API key seeded for project: %s", project_id)


_seed_from_env()


def issue_key(project_id: str) -> str:
    """Issue a new random API key for *project_id*.

    Returns the plain-text key (shown once; store securely).
    """
    key = secrets.token_hex(32)
    _KEY_STORE[_hash_key(key)] = project_id
    logger.info("Issued API key for project: %s", project_id)
    return key


def verify_api_key(key: str) -> Optional[str]:
    """Return the project_id associated with *key*, or None if invalid."""
    key_hash = _hash_key(key)
    # Use constant-time comparison to avoid timing attacks
    for stored_hash, project_id in _KEY_STORE.items():
        if hmac.compare_digest(stored_hash, key_hash):
            return project_id
    return None


def revoke_key(key: str) -> bool:
    """Remove a key from the store. Returns True if it existed."""
    key_hash = _hash_key(key)
    if key_hash in _KEY_STORE:
        del _KEY_STORE[key_hash]
        return True
    return False
