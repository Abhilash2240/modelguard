from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_inference_service
from app.core.errors import EntityNotFoundError, InferenceError, InputContractViolationError
from app.schemas.inference import PredictionRequest, PredictionResponse
from app.services.inference_service import InferenceService

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Execute model prediction using the active production champion or an exact specified version."""
    try:
        return inference_service.predict(payload)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except InputContractViolationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except InferenceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
