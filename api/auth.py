"""
auth.py — SUMOTX API token-based authentication dependency

Phase 1 implementation uses a simple bearer-token check against the
SUMOTX_API_KEY environment variable.  Entra ID / Azure AD integration
is planned for a later phase.

Environment variables:
    SUMOTX_API_KEY  — shared secret used to validate Bearer tokens
                      (default: "dev-secret" for local development only)
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("smtx.api.auth")

_bearer_scheme = HTTPBearer(auto_error=True)

_VALID_TOKEN: str = os.environ.get("SUMOTX_API_KEY", "dev-secret")


def require_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """
    FastAPI dependency that validates the Bearer token in every protected
    request.  Raises HTTP 401 if the token is missing or incorrect.

    Returns the validated token string so downstream handlers can log it.
    """
    if credentials.credentials != _VALID_TOKEN:
        logger.warning("Rejected request with invalid Bearer token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
