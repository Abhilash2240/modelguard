from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_training_job_service, require_viewer
from app.core.errors import EntityNotFoundError
from app.db import models
from app.schemas.training import TrainingJobResponse, TrainingRunResponse
from app.services.training_job_service import TrainingJobService

router = APIRouter(prefix="/training", tags=["Training"])


@router.get("/jobs", response_model=list[TrainingJobResponse])
def list_training_jobs(
    status_filter: str | None = Query(None, alias="status", description="Filter by training job status"),
    dataset_version_id: str | None = Query(None, description="Filter by dataset version ID"),
    current_user: models.User = Depends(require_viewer),
    service: TrainingJobService = Depends(get_training_job_service),
) -> list[TrainingJobResponse]:
    """List training jobs with nested training run executions. Protected: Viewer+"""
    jobs = service.list_training_jobs(status=status_filter, dataset_version_id=dataset_version_id)
    return [TrainingJobResponse.model_validate(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(
    job_id: str,
    current_user: models.User = Depends(require_viewer),
    service: TrainingJobService = Depends(get_training_job_service),
) -> TrainingJobResponse:
    """Retrieve details for a specific training job. Protected: Viewer+"""
    try:
        job = service.get_training_job(job_id)
        return TrainingJobResponse.model_validate(job)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get("/runs", response_model=list[TrainingRunResponse])
def list_training_runs(
    status_filter: str | None = Query(None, alias="status", description="Filter by training run status"),
    dataset_version_id: str | None = Query(None, description="Filter by dataset version ID"),
    current_user: models.User = Depends(require_viewer),
    service: TrainingJobService = Depends(get_training_job_service),
) -> list[TrainingRunResponse]:
    """List executed training runs with metrics and artifact URIs. Protected: Viewer+"""
    runs = service.list_training_runs(status=status_filter, dataset_version_id=dataset_version_id)
    return [TrainingRunResponse.model_validate(r) for r in runs]


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
def get_training_run(
    run_id: str,
    current_user: models.User = Depends(require_viewer),
    service: TrainingJobService = Depends(get_training_job_service),
) -> TrainingRunResponse:
    """Retrieve details for a specific training run. Protected: Viewer+"""
    try:
        run = service.get_training_run(run_id)
        return TrainingRunResponse.model_validate(run)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
