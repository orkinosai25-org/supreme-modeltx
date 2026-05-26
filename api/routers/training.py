"""
routers/training.py — Training run lifecycle

POST /v1/training-runs
GET  /v1/training-runs
GET  /v1/training-runs/{run_id}
GET  /v1/training-runs/{run_id}/logs
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_token
from api.schemas import (
    TrainingLogEntry,
    TrainingRun,
    TrainingRunCreate,
    TrainingRunList,
    TrainingRunLogs,
    TrainingRunStatus,
)

logger = logging.getLogger("smtx.api.training")

router = APIRouter(prefix="/training-runs", tags=["Training"])

# In-memory store (replace with a real database in production)
_runs: dict[str, TrainingRun] = {}


@router.post(
    "",
    response_model=TrainingRun,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a training run",
    description=(
        "Submits a new fine-tuning job to Azure Batch.  "
        "Returns immediately with `status: queued`; poll "
        "`GET /v1/training-runs/{id}` to track progress."
    ),
)
def create_training_run(
    body: TrainingRunCreate,
    _token: Annotated[str, Depends(require_token)],
) -> TrainingRun:
    run = TrainingRun(
        project_id=body.project_id,
        base_model=body.base_model,
        dataset_path=body.dataset_path,
        num_epochs=body.num_epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
        warmup_steps=body.warmup_steps,
        output_model_name=body.output_model_name,
        status=TrainingRunStatus.queued,
    )
    _runs[run.id] = run
    logger.info(
        "Training run queued id=%s base_model=%s dataset=%s",
        run.id,
        run.base_model,
        run.dataset_path,
    )
    # TODO: submit Azure Batch job and store azure_batch_job_id
    return run


@router.get(
    "",
    response_model=TrainingRunList,
    summary="List training runs",
    description="Returns all training runs for the authenticated principal.",
)
def list_training_runs(
    _token: Annotated[str, Depends(require_token)],
) -> TrainingRunList:
    items = list(_runs.values())
    return TrainingRunList(items=items, total=len(items))


@router.get(
    "/{run_id}",
    response_model=TrainingRun,
    summary="Get a training run",
    description="Returns full metadata and current status for a single training run.",
    responses={404: {"description": "Training run not found"}},
)
def get_training_run(
    run_id: str,
    _token: Annotated[str, Depends(require_token)],
) -> TrainingRun:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training run not found.")
    return run


@router.get(
    "/{run_id}/logs",
    response_model=TrainingRunLogs,
    summary="Stream training run logs",
    description=(
        "Returns the captured log output for a training run.  "
        "Logs are sourced from the Azure Batch task stdout/stderr."
    ),
    responses={404: {"description": "Training run not found"}},
)
def get_training_run_logs(
    run_id: str,
    _token: Annotated[str, Depends(require_token)],
) -> TrainingRunLogs:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training run not found.")
    # Placeholder logs — real implementation reads from Azure Blob Storage
    placeholder_logs: List[TrainingLogEntry] = [
        TrainingLogEntry(
            timestamp=datetime.now(tz=timezone.utc),
            level="INFO",
            message=f"Training run {run_id} is {run.status.value}. Detailed logs will appear here once the Azure Batch job is running.",
        )
    ]
    return TrainingRunLogs(run_id=run_id, logs=placeholder_logs)
