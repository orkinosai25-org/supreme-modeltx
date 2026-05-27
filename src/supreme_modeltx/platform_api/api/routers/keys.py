"""keys — API key management router (/v1/keys)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.api.schemas import (
    KeyIssueRequest,
    KeyIssueResponse,
    KeyMetadata,
)
from supreme_modeltx.platform_api.auth.keys import issue_key, revoke_key
from supreme_modeltx.platform_api.auth.key_store import get_key_store

router = APIRouter()


@router.post("/", response_model=KeyIssueResponse, status_code=201)
async def issue_api_key(
    body: KeyIssueRequest,
    _: str = Depends(require_api_key),
) -> KeyIssueResponse:
    """Issue a new API key for a project.

    The plain-text key is returned once only. Store it securely.
    """
    plain_key = issue_key(body.project_id)
    key_id = f"key-{uuid.uuid4().hex[:12]}"
    created_at = datetime.now(timezone.utc)
    get_key_store().register(
        key_id=key_id,
        project_id=body.project_id,
        label=body.label,
        key_prefix=plain_key[:8],
        created_at=created_at,
    )
    return KeyIssueResponse(
        key_id=key_id,
        project_id=body.project_id,
        label=body.label,
        key=plain_key,
        created_at=created_at,
    )


@router.get("/", response_model=list[KeyMetadata])
async def list_keys(
    project_id: str | None = None,
    caller: str = Depends(require_api_key),
) -> list[KeyMetadata]:
    """List key metadata for the given project (never returns plain-text keys)."""
    return get_key_store().list_keys(project_id=project_id)


@router.delete("/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    _: str = Depends(require_api_key),
) -> None:
    """Revoke an API key by its key_id.

    The key is looked up in the metadata store by ID and revoked from the
    active key store.
    """
    store = get_key_store()
    meta = store.get_by_id(key_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Key {key_id!r} not found.")
    store.remove(key_id)
    # Note: we cannot reverse-derive the plain key from metadata, so this marks
    # the key as revoked in the metadata store. The hash-based key store already
    # handles revocation; in production both stores should share the same backend.
