from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    inputs: list[dict[str, Any]] = Field(..., min_length=1, description="List of feature records matching contract")
    model_version: str | None = Field(
        None, description="Optional exact model version to target; defaults to active champion"
    )


class PredictionResponse(BaseModel):
    inference_id: str
    model_name: str
    model_version: str
    predictions: list[float]
    latency_ms: float
    timestamp: datetime
