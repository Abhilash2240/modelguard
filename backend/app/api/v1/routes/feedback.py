from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_feedback_service
from app.core.errors import EntityNotFoundError, FeedbackConflictError
from app.schemas.monitoring import FeedbackCreate, FeedbackResponse, QualityReportResponse
from app.services.feedback_service import FeedbackService

router = APIRouter(tags=["Feedback & Quality"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
def submit_feedback(
    payload: FeedbackCreate,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    """Submit ground-truth outcome linked to an existing inference identifier with strict idempotency."""
    try:
        feedback = feedback_service.submit_feedback(
            inference_id=payload.inference_id,
            actual_value=payload.actual_value,
            submitted_by=payload.submitted_by,
            notes=payload.notes,
        )
        return FeedbackResponse.model_validate(feedback)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except FeedbackConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@router.get("/quality", response_model=QualityReportResponse)
def get_rolling_quality(
    model_name: str = Query("metro-traffic-volume", description="Name of registered model"),
    model_version: str | None = Query(None, description="Optional exact model version"),
    min_samples: int = Query(10, description="Minimum labeled samples required before computing metrics"),
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> QualityReportResponse:
    """Evaluate rolling MAE, RMSE, and R2 against delayed feedback."""
    report = feedback_service.calculate_rolling_quality(
        model_name=model_name,
        model_version=model_version,
        min_samples=min_samples,
    )
    return QualityReportResponse.model_validate(report)


@router.get("/quality/history", response_model=list[QualityReportResponse])
def list_quality_history(
    model_version: str | None = Query(None, description="Filter reports by model version"),
    limit: int = Query(50, description="Maximum number of historical reports to return"),
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> list[QualityReportResponse]:
    """Retrieve historical quality evaluation reports."""
    reports = feedback_service.list_quality_reports(model_version=model_version, limit=limit)
    return [QualityReportResponse.model_validate(r) for r in reports]
