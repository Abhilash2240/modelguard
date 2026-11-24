from collections.abc import Sequence

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db import models
from app.db.session import get_db
from app.services.alert_service import AlertService
from app.services.dataset_service import DatasetService
from app.services.deployment_service import DeploymentService
from app.services.drift_service import DriftAnalysisService
from app.services.feedback_service import FeedbackService
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService
from app.services.training_job_service import TrainingJobService

_http_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    auth_creds: HTTPAuthorizationCredentials | None = Depends(_http_bearer),
    db: Session = Depends(get_db),
) -> models.User:
    """Extract and verify authenticated user from Bearer header or HttpOnly session cookie."""
    token: str | None = None

    # 1. Check Authorization: Bearer <token> header
    if auth_creds and auth_creds.credentials:
        token = auth_creds.credentials
    # 2. Check HttpOnly cookie fallback
    elif "mg_access_token" in request.cookies:
        token = request.cookies.get("mg_access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload["sub"]
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user record no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive or disabled.",
        )

    return user


def require_role(allowed_roles: Sequence[str]):
    """Factory dependency enforcing that current user possesses an authorized role."""

    def role_checker(current_user: models.User = Depends(get_current_user)) -> models.User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of the following roles: {list(allowed_roles)}. Current role: '{current_user.role}'.",
            )
        return current_user

    return role_checker


# Role-based dependency shortcuts matching ModelGuard governance matrix
require_viewer = require_role(
    [
        models.UserRole.VIEWER.value,
        models.UserRole.OPERATOR.value,
        models.UserRole.APPROVER.value,
        models.UserRole.ADMIN.value,
    ]
)

require_operator = require_role(
    [
        models.UserRole.OPERATOR.value,
        models.UserRole.ADMIN.value,
    ]
)

require_approver = require_role(
    [
        models.UserRole.APPROVER.value,
        models.UserRole.ADMIN.value,
    ]
)

require_admin = require_role(
    [
        models.UserRole.ADMIN.value,
    ]
)


# Service dependencies
def get_dataset_service(db: Session = Depends(get_db)) -> DatasetService:
    return DatasetService(db=db)


def get_model_registry_service(db: Session = Depends(get_db)) -> ModelRegistryService:
    return ModelRegistryService(db=db)


def get_deployment_service(db: Session = Depends(get_db)) -> DeploymentService:
    return DeploymentService(db=db)


def get_inference_service(db: Session = Depends(get_db)) -> InferenceService:
    return InferenceService(db=db)


def get_alert_service(db: Session = Depends(get_db)) -> AlertService:
    return AlertService(db=db)


def get_feedback_service(db: Session = Depends(get_db)) -> FeedbackService:
    return FeedbackService(db=db)


def get_drift_service(db: Session = Depends(get_db)) -> DriftAnalysisService:
    return DriftAnalysisService(db=db)


def get_training_job_service(db: Session = Depends(get_db)) -> TrainingJobService:
    return TrainingJobService(db=db)
