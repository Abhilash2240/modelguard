import shutil
import tempfile
from pathlib import Path

from app.api.deps import get_dataset_service
from app.core.errors import EntityNotFoundError, IntegrityError, InvalidOperationError, StorageError
from app.schemas.dataset import (
    DatasetCreate,
    DatasetDetailResponse,
    DatasetResponse,
    DatasetVersionResponse,
)
from app.schemas.validation import ValidationRunResponse
from app.services.dataset_service import DatasetService
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

router = APIRouter(prefix="", tags=["Datasets"])


@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def register_dataset(
    payload: DatasetCreate,
    service: DatasetService = Depends(get_dataset_service),
):
    """Register a new logical dataset."""
    try:
        return service.register_dataset(payload)
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(service: DatasetService = Depends(get_dataset_service)):
    """List all registered datasets."""
    return service.list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset(dataset_id: str, service: DatasetService = Depends(get_dataset_service)):
    """Get dataset details with versions."""
    try:
        return service.get_dataset(dataset_id)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/datasets/{dataset_id}/versions",
    response_model=DatasetVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset_version(
    dataset_id: str,
    version_tag: str = Form(..., description="e.g. v1.0.0"),
    file: UploadFile = File(...),
    service: DatasetService = Depends(get_dataset_service),
):
    """Upload a CSV dataset snapshot and create an immutable version."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are currently supported for dataset snapshots.",
        )

    # Save uploaded file temporarily to compute checksum and inspect schema
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        version = service.create_dataset_version(
            dataset_id=dataset_id,
            version_tag=version_tag,
            source_file_path=tmp_path,
        )
        return version
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/dataset-versions/{version_id}", response_model=DatasetVersionResponse)
def get_dataset_version(version_id: str, service: DatasetService = Depends(get_dataset_service)):
    """Get metadata for an immutable dataset version snapshot."""
    try:
        return service.get_dataset_version(version_id)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/dataset-versions/{version_id}/validate", response_model=ValidationRunResponse)
def validate_dataset_version(
    version_id: str,
    contract_version: str = "v1",
    service: DatasetService = Depends(get_dataset_service),
):
    """Validate a dataset version against a versioned Pandera contract."""
    try:
        return service.validate_version(version_id=version_id, contract_version=contract_version)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except IntegrityError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/validation-runs/{run_id}", response_model=ValidationRunResponse)
def get_validation_run(
    run_id: str,
    service: DatasetService = Depends(get_dataset_service),
):
    """Retrieve full audit findings for a validation run."""
    from app.db.models import ValidationRun

    run = service.db.query(ValidationRun).filter(ValidationRun.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Validation run '{run_id}' not found."
        )
    return run
