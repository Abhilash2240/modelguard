from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (
    get_deployment_service,
    require_approver,
    require_operator,
    require_viewer,
)
from app.core.errors import (
    DeploymentError,
    EntityNotFoundError,
    InvalidOperationError,
    RollbackError,
)
from app.db import models
from app.schemas.deployments import (
    ApprovalRequest,
    DeploymentResponse,
    RejectionRequest,
    RollbackRequest,
)
from app.services.deployment_service import DeploymentService

router = APIRouter(prefix="/deployments", tags=["Deployments"])


@router.get("", response_model=list[DeploymentResponse])
def list_deployments(
    environment: str | None = Query(
        None, description="Filter deployments by environment name (e.g. staging, production)"
    ),
    current_user: models.User = Depends(require_viewer),
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> list[DeploymentResponse]:
    """List deployment records across staging and production environments. Protected: Viewer+"""
    deployments = deployment_service.list_deployments(environment_name=environment)
    return [DeploymentResponse.model_validate(d) for d in deployments]


@router.post("/{deployment_id}/approve", response_model=DeploymentResponse)
def approve_deployment(
    deployment_id: str,
    payload: ApprovalRequest,
    current_user: models.User = Depends(require_approver),
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Record explicit human approval for a staged model candidate.

    Protected route: Requires approver or admin role.
    Security: Derives actor from authenticated principal, eliminating client impersonation.
    """
    try:
        deployment = deployment_service.approve_deployment(
            deployment_id=deployment_id,
            actor=current_user.username,
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
    current_user: models.User = Depends(require_approver),
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Record explicit human rejection for a staged model candidate.

    Protected route: Requires approver or admin role.
    Security: Derives actor from authenticated principal.
    """
    try:
        deployment = deployment_service.reject_deployment(
            deployment_id=deployment_id,
            actor=current_user.username,
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
    current_user: models.User = Depends(require_approver),
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Promote an approved staged deployment to production champion.

    Protected route: Requires approver or admin role.
    Security: Derives actor from authenticated principal.
    """
    try:
        deployment = deployment_service.promote_deployment(
            deployment_id=deployment_id,
            actor=current_user.username,
        )
        return DeploymentResponse.model_validate(deployment)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except DeploymentError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/{deployment_id}/rollback", response_model=DeploymentResponse)
def rollback_deployment(
    deployment_id: str,
    payload: RollbackRequest,
    current_user: models.User = Depends(require_operator),
    deployment_service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Safely roll back active production serving pointer to a verified previous candidate.

    Protected route: Requires operator or admin role.
    Security: Derives actor from authenticated principal.
    """
    try:
        deployment = deployment_service.rollback_deployment(
            deployment_id=deployment_id,
            actor=current_user.username,
            reason=payload.reason,
            target_candidate_id=payload.target_candidate_id,
        )
        return DeploymentResponse.model_validate(deployment)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except (InvalidOperationError, RollbackError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
