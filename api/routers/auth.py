"""
routers/auth.py — POST /v1/auth/token

Phase 1: exchanges a raw API key for a short-lived Bearer token.
The token issued here is identical to the raw key in this phase;
a proper JWT / Entra ID flow replaces this in a later phase.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, status

from api.schemas import TokenRequest, TokenResponse

logger = logging.getLogger("smtx.api.auth_router")

router = APIRouter(prefix="/auth", tags=["Authentication"])

_VALID_API_KEY: str = os.environ.get("SUMOTX_API_KEY", "dev-secret")


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Exchange API key for Bearer token",
    description=(
        "Validates the supplied API key and returns a short-lived Bearer token. "
        "Include the token in subsequent requests as `Authorization: Bearer <token>`. "
        "Entra ID / Azure AD integration is planned for a later release."
    ),
)
def get_token(body: TokenRequest) -> TokenResponse:
    if body.api_key != _VALID_API_KEY:
        logger.warning("Failed authentication attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    logger.info("Issued Bearer token.")
    return TokenResponse(
        access_token=body.api_key,
        token_type="bearer",
        expires_in=3600,
    )
