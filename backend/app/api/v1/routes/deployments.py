from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_deployment_service
from app.core.errors import DeploymentError, EntityNotFoundError, InvalidOperationError
from app.schemas.deployments import ApprovalRequest, DeploymentResponse, RejectionRequest
from app.services.deployment_service import DeploymentService

router = APIRouter(prefix="/deployments", tags=["Deployments"])


@router.get("", response_model=list[DeploymentResponse])
def list_deployments(
    environment: str | None = Query(
        None, description="Filter deployments by environment name (e.g. staging, production)"
    ),
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> list[DeploymentResponse]:
    """List deployment records across staging and production environments."""
    deployments = deployment_service.list_deployments(environment_name=environment)
    return [DeploymentResponse.model_validate(d) for d in deployments]


@router.post("/{deployment_id}/approve", response_model=DeploymentResponse)
def approve_deployment(
    deployment_id: str,
    payload: ApprovalRequest,
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Record explicit human approval for a staged model candidate."""
    try:
        deployment = deployment_service.approve_deployment(
            deployment_id=deployment_id,
            actor=payload.actor,
            rationale=payload.rationale,
        )
        return DeploymentResponse.model_validate(deployment)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@router.post("/{deployment_id}/reject", response_model=DeploymentResponse)
def reject_deployment(
    deployment_id: str,
    payload: RejectionRequest,
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Record explicit human rejection for a staged model candidate, preserving production unchanged."""
    try:
        deployment = deployment_service.reject_deployment(
            deployment_id=deployment_id,
            actor=payload.actor,
            reason=payload.reason,
        )
        return DeploymentResponse.model_validate(deployment)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@router.post("/{deployment_id}/promote", response_model=DeploymentResponse)
def promote_deployment(
    deployment_id: str,
    payload: ApprovalRequest | None = None,
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Promote an approved staged deployment to production champion."""
    actor = payload.actor if payload else "admin"
    try:
        deployment = deployment_service.promote_deployment(
            deployment_id=deployment_id,
            actor=actor,
        )
        return DeploymentResponse.model_validate(deployment)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except DeploymentError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
