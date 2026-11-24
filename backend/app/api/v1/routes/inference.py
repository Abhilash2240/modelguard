from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_inference_service, require_operator, require_viewer
from app.core.errors import EntityNotFoundError, InferenceError, InputContractViolationError
from app.db import models
from app.schemas.inference import PredictionRequest, PredictionResponse
from app.services.inference_service import InferenceService

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    current_user: models.User = Depends(require_viewer),
    inference_service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Execute model prediction using the active, approved production champion.

    Protected route: Requires authenticated viewer role or higher.
    Production restriction: Strictly serves the promoted production champion.
    Explicit model version testing must use the diagnostic endpoint.
    """
    try:
        return inference_service.predict(payload)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InputContractViolationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except InferenceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/inference/diagnostic", response_model=PredictionResponse)
def predict_diagnostic(
    payload: PredictionRequest,
    current_user: models.User = Depends(require_operator),
    inference_service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Execute non-production diagnostic inference against a specific candidate version.

    Protected route: Requires operator or admin role.
    Governance safeguard: Operates in isolated diagnostic mode without modifying champion routing.
    """
    try:
        return inference_service.predict_diagnostic(payload, actor=current_user.username)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InputContractViolationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except InferenceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
