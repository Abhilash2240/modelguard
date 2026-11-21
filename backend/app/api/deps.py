from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.alert_service import AlertService
from app.services.dataset_service import DatasetService
from app.services.deployment_service import DeploymentService
from app.services.drift_service import DriftAnalysisService
from app.services.feedback_service import FeedbackService
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService


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
