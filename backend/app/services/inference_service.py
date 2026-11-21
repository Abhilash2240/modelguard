import time
import uuid
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import (
    EntityNotFoundError,
    InferenceError,
    InputContractViolationError,
)
from app.core.logging import logger
from app.db import models
from app.ml.features import prepare_features
from app.schemas.inference import PredictionRequest, PredictionResponse


def utc_now() -> datetime:
    return datetime.now(UTC)


REQUIRED_FEATURE_NAMES = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
]


class InferenceService:
    """Service providing version-aware, contract-validated model serving and telemetry persistence."""

    def __init__(self, db: Session, tracking_uri: str | None = None) -> None:
        self.db = db
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self._model_cache: dict[str, Any] = {}

    def _resolve_candidate(self, model_version: str | None = None) -> models.ModelCandidate:
        """Resolve exact candidate record from explicit version or active production champion."""
        if model_version:
            candidate = (
                self.db.query(models.ModelCandidate)
                .filter(models.ModelCandidate.mlflow_model_version == str(model_version))
                .first()
            )
            if not candidate:
                raise EntityNotFoundError(f"ModelCandidate with version '{model_version}' not found.")
            return candidate

        # Default to active production champion
        prod_env = (
            self.db.query(models.DeploymentEnvironment)
            .filter(models.DeploymentEnvironment.name == "production")
            .first()
        )
        if not prod_env or not prod_env.active_candidate_id:
            raise EntityNotFoundError(
                "No active production model deployed. Please stage, approve, and promote a model candidate first."
            )

        candidate = prod_env.active_candidate
        if not candidate:
            raise EntityNotFoundError("Active production candidate record missing from database.")
        return candidate

    def _get_model(self, candidate: models.ModelCandidate) -> Any:
        """Load and cache model pipeline by exact immutable model URI."""
        model_uri = candidate.model_uri
        if model_uri not in self._model_cache:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                model = mlflow.sklearn.load_model(model_uri)
                self._model_cache[model_uri] = model
            except Exception as e:
                raise InferenceError(f"Failed to load model from URI '{model_uri}': {e}") from e
        return self._model_cache[model_uri]

    def validate_inputs(self, inputs: list[dict[str, Any]]) -> pd.DataFrame:
        """Validate input payload against serving contract before calling estimator."""
        if not inputs:
            raise InputContractViolationError("Prediction input payload must contain at least one record.")

        for idx, record in enumerate(inputs):
            missing = [f for f in REQUIRED_FEATURE_NAMES if f not in record]
            if missing:
                raise InputContractViolationError(
                    f"Record at index {idx} violates input contract. Missing required feature(s): {missing}"
                )

        df = pd.DataFrame(inputs)

        # Validate timestamp parseability
        try:
            pd.to_datetime(df["date_time"])
        except Exception as e:
            raise InputContractViolationError(f"Invalid timestamp in 'date_time' column: {e}") from e

        # Validate numeric types and finite values
        for num_col in ("temp", "rain_1h", "snow_1h", "clouds_all"):
            try:
                converted = pd.to_numeric(df[num_col], errors="coerce")
                if num_col == "clouds_all" and converted.isnull().any():
                    raise InputContractViolationError(
                        f"Column '{num_col}' contains invalid or null non-numeric values."
                    )
            except Exception as e:
                raise InputContractViolationError(f"Column '{num_col}' type coercion failed: {e}") from e

        return df

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Execute version-aware inference with contract enforcement and telemetry persistence."""
        start_time = time.perf_counter()
        request_time = utc_now()
        inference_id = str(uuid.uuid4())

        # Validate inputs strictly prior to resolving or loading estimator
        df_raw = self.validate_inputs(request.inputs)
        X, _ = prepare_features(df_raw, target_col="traffic_volume")

        candidate = self._resolve_candidate(request.model_version)
        model = self._get_model(candidate)

        try:
            raw_predictions = model.predict(X)
        except Exception as e:
            # Persist failure telemetry
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            event = models.PredictionEvent(
                inference_id=inference_id,
                candidate_id=candidate.id,
                model_name=candidate.mlflow_model_name,
                model_version=candidate.mlflow_model_version,
                request_time=request_time,
                latency_ms=latency_ms,
                status="ERROR",
                prediction_json={"error": str(e)},
                input_summary_json={"row_count": len(request.inputs), "error": str(e)},
            )
            self.db.add(event)
            self.db.commit()
            raise InferenceError(f"Model estimation failed during execution: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        preds_list = [float(p) for p in raw_predictions]

        # Verify finite predictions
        if not np.all(np.isfinite(raw_predictions)):
            raise InferenceError("Model produced non-finite (NaN or Inf) predictions.")

        # Construct privacy-safe input summary (no raw payload retained)
        input_summary = {
            "row_count": len(request.inputs),
            "features_received": list(request.inputs[0].keys()),
            "null_counts": {k: int(v) for k, v in df_raw.isnull().sum().items()},
        }

        event = models.PredictionEvent(
            inference_id=inference_id,
            candidate_id=candidate.id,
            model_name=candidate.mlflow_model_name,
            model_version=candidate.mlflow_model_version,
            request_time=request_time,
            latency_ms=latency_ms,
            status="SUCCESS",
            prediction_json={"count": len(preds_list), "sample_first": preds_list[0] if preds_list else None},
            input_summary_json=input_summary,
        )
        self.db.add(event)
        self.db.commit()

        logger.info(
            f"Inference '{inference_id}' succeeded: model '{candidate.mlflow_model_name}' v{candidate.mlflow_model_version}, {len(preds_list)} records, {latency_ms:.2f}ms."
        )

        return PredictionResponse(
            inference_id=inference_id,
            model_name=candidate.mlflow_model_name,
            model_version=candidate.mlflow_model_version,
            predictions=preds_list,
            latency_ms=latency_ms,
            timestamp=request_time,
        )
