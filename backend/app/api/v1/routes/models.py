from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_deployment_service, get_model_registry_service
from app.core.errors import EntityNotFoundError, InvalidOperationError, ModelRegistryError
from app.schemas.deployments import DeploymentResponse
from app.schemas.models import ModelCandidateCreate, ModelCandidateResponse
from app.services.deployment_service import DeploymentService
from app.services.model_registry_service import ModelRegistryService

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=list[ModelCandidateResponse])
def list_models(
    status: str | None = Query(None, description="Filter by candidate status"),
    dataset_version_id: str | None = Query(None, description="Filter by dataset version ID"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> list[ModelCandidateResponse]:
    """List registered model candidates with optional status and dataset version filtering."""
    candidates = registry_service.list_candidates(status=status, dataset_version_id=dataset_version_id)
    return [ModelCandidateResponse.model_validate(c) for c in candidates]


@router.post("/register", response_model=ModelCandidateResponse, status_code=status.HTTP_201_CREATED)
def register_model_candidate(
    payload: ModelCandidateCreate,
    registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> ModelCandidateResponse:
    """Register a trained candidate model from a completed TrainingRun."""
    try:
        candidate = registry_service.register_candidate(
            training_run_id=payload.training_run_id,
            model_name=payload.model_name or "metro-traffic-volume",
        )
        return ModelCandidateResponse.model_validate(candidate)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except ModelRegistryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/{model_id}", response_model=ModelCandidateResponse)
def get_model(
    model_id: str,
    registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> ModelCandidateResponse:
    """Retrieve details and lineage for a specific model candidate."""
    try:
        candidate = registry_service.get_candidate(model_id)
        return ModelCandidateResponse.model_validate(candidate)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post("/{model_id}/stage", response_model=DeploymentResponse)
def stage_model(
    model_id: str,
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Stage an exact model candidate and run automated smoke checks."""
    try:
        deployment = deployment_service.stage_candidate(model_id)
        return DeploymentResponse.model_validate(deployment)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
