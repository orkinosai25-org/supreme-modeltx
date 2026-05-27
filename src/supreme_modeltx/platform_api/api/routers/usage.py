"""usage — Usage and metering router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from supreme_modeltx.platform_api.api.routers.auth import require_api_key
from supreme_modeltx.platform_api.usage.metering import UsageLedger, UsageSummary

router = APIRouter()
_ledger = UsageLedger()


@router.get("/{project_id}", response_model=UsageSummary)
async def get_usage(
    project_id: str,
    caller_project: str = Depends(require_api_key),
) -> UsageSummary:
    """Return aggregated token usage for a project."""
    return _ledger.summarise(project_id)
