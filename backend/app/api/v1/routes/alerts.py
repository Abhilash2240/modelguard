from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_alert_service
from app.core.errors import EntityNotFoundError, InvalidOperationError
from app.schemas.monitoring import AlertAcknowledgeRequest, AlertResolveRequest, AlertResponse
from app.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=list[AlertResponse])
def list_alerts(
    status: str | None = Query(None, description="Filter alerts by status (OPEN, ACKNOWLEDGED, RESOLVED)"),
    severity: str | None = Query(None, description="Filter alerts by severity (INFO, WARNING, CRITICAL)"),
    model_version: str | None = Query(None, description="Filter alerts by model version"),
    alert_service: AlertService = Depends(get_alert_service),
) -> list[AlertResponse]:
    """List persistent monitoring alerts with optional status and severity filters."""
    alerts = alert_service.list_alerts(status=status, severity=severity, model_version=model_version)
    return [AlertResponse.model_validate(a) for a in alerts]


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: str,
    alert_service: AlertService = Depends(get_alert_service),
) -> AlertResponse:
    """Retrieve details for a specific alert."""
    try:
        alert = alert_service.get_alert(alert_id)
        return AlertResponse.model_validate(alert)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(
    alert_id: str,
    payload: AlertAcknowledgeRequest,
    alert_service: AlertService = Depends(get_alert_service),
) -> AlertResponse:
    """Acknowledge an open alert."""
    try:
        alert = alert_service.acknowledge_alert(alert_id=alert_id, actor=payload.actor)
        return AlertResponse.model_validate(alert)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
def resolve_alert(
    alert_id: str,
    payload: AlertResolveRequest,
    alert_service: AlertService = Depends(get_alert_service),
) -> AlertResponse:
    """Resolve an alert with operational notes."""
    try:
        alert = alert_service.resolve_alert(alert_id=alert_id, actor=payload.actor, notes=payload.notes)
        return AlertResponse.model_validate(alert)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InvalidOperationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
