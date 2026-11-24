import time
import uuid
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core import metrics
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

# Fixed reference bin edges for privacy-preserving numerical distribution telemetry
FEATURE_HISTOGRAM_BINS: dict[str, list[float]] = {
    "temp": [240.0, 255.0, 265.0, 275.0, 285.0, 295.0, 305.0, 315.0, 330.0],
    "rain_1h": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0],
    "snow_1h": [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    "clouds_all": [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
}


class InferenceService:
    """Service providing version-aware, contract-validated model serving and telemetry persistence."""

    def __init__(self, db: Session, tracking_uri: str | None = None) -> None:
        self.db = db
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self._model_cache: dict[str, Any] = {}

    def _resolve_candidate(
        self,
        model_version: str | None = None,
        is_diagnostic: bool = False,
    ) -> models.ModelCandidate:
        """Resolve candidate record.

        In production inference (is_diagnostic=False), strictly resolves the active approved
        production champion. If an explicit model_version is passed, verifies it matches the champion.
        In diagnostic mode (is_diagnostic=True), allows evaluating any registered candidate.
        """
        if is_diagnostic and model_version:
            candidate = (
                self.db.query(models.ModelCandidate)
                .filter(models.ModelCandidate.mlflow_model_version == str(model_version))
                .first()
            )
            if not candidate:
                raise EntityNotFoundError(
                    f"ModelCandidate with version '{model_version}' not found for diagnostic inference."
                )
            return candidate

        # Default / Production: Resolve active production champion
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

        if candidate.status != "PROMOTED":
            raise InferenceError(
                f"Active candidate '{candidate.id}' is in status '{candidate.status}', not 'PROMOTED'. Production inference blocked."
            )

        # Enforce champion policy: caller cannot silently bypass production champion in production inference
        if model_version and str(model_version) != str(candidate.mlflow_model_version):
            raise InferenceError(
                f"Model version '{model_version}' is not the active production champion (active version: '{candidate.mlflow_model_version}'). "
                f"Production inference serves only the promoted champion. For non-production diagnostics, use /api/v1/inference/diagnostic."
            )

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
                metrics.record_model_load_failure(candidate.mlflow_model_name, "production")
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

        # Construct privacy-safe input summary with sufficient statistics (no raw payload retained)
        numeric_stats = {}
        for num_col in ("temp", "rain_1h", "snow_1h", "clouds_all"):
            if num_col in df_raw.columns:
                s = pd.to_numeric(df_raw[num_col], errors="coerce").dropna()
                if len(s) > 0:
                    bins = FEATURE_HISTOGRAM_BINS.get(num_col, [])
                    bin_counts = []
                    underflow = 0
                    overflow = 0
                    if len(bins) >= 2:
                        counts, _ = np.histogram(s.to_numpy(), bins=bins)
                        bin_counts = [int(c) for c in counts]
                        underflow = int((s < bins[0]).sum())
                        overflow = int((s > bins[-1]).sum())

                    numeric_stats[num_col] = {
                        "count": int(len(s)),
                        "mean": float(s.mean()),
                        "std": float(s.std()) if len(s) > 1 else 0.0,
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "q25": float(s.quantile(0.25)),
                        "q50": float(s.median()),
                        "q75": float(s.quantile(0.75)),
                        "bin_edges": bins,
                        "bin_counts": bin_counts,
                        "underflow": underflow,
                        "overflow": overflow,
                    }

        categorical_stats = {}
        for cat_col in ("holiday", "weather_main", "weather_description"):
            if cat_col in df_raw.columns:
                missing_cnt = int(df_raw[cat_col].isna().sum())
                val_counts = df_raw[cat_col].dropna().astype(str).value_counts().to_dict()
                categorical_stats[cat_col] = {
                    "missing_count": missing_cnt,
                    "frequencies": {str(k): int(v) for k, v in val_counts.items()},
                }

        ref_ds = getattr(candidate, "dataset_version", None)
        input_summary = {
            "row_count": len(request.inputs),
            "features_received": list(request.inputs[0].keys()),
            "null_counts": {k: int(v) for k, v in df_raw.isnull().sum().items()},
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "reference_dataset_version_id": candidate.dataset_version_id,
            "reference_dataset_tag": ref_ds.version_tag if ref_ds else "v1.0.0",
            "reference_checksum": ref_ds.sha256_hash if ref_ds else "",
        }

        # Track inference metrics in Prometheus
        metrics.track_inference(candidate.mlflow_model_name, candidate.mlflow_model_version, latency_ms / 1000.0)

        event = models.PredictionEvent(
            inference_id=inference_id,
            candidate_id=candidate.id,
            model_name=candidate.mlflow_model_name,
            model_version=candidate.mlflow_model_version,
            request_time=request_time,
            latency_ms=latency_ms,
            status="SUCCESS",
            prediction_json={"predictions": preds_list, "count": len(preds_list)},
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

    def predict_diagnostic(self, request: PredictionRequest, actor: str) -> PredictionResponse:
        """Execute non-production diagnostic inference without altering production champion routing."""
        start_time = time.perf_counter()
        request_time = utc_now()
        inference_id = str(uuid.uuid4())

        df_raw = self.validate_inputs(request.inputs)
        X, _ = prepare_features(df_raw, target_col="traffic_volume")

        # In diagnostic mode, resolve candidate explicitly without requiring it to be production champion
        candidate = self._resolve_candidate(request.model_version, is_diagnostic=True)
        model = self._get_model(candidate)

        try:
            raw_predictions = model.predict(X)
        except Exception as e:
            raise InferenceError(f"Diagnostic estimation failed: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        preds_list = [float(p) for p in raw_predictions]

        if not np.all(np.isfinite(raw_predictions)):
            raise InferenceError("Diagnostic model produced non-finite predictions.")

        input_summary = {
            "row_count": len(request.inputs),
            "diagnostic_actor": actor,
            "environment": "diagnostic",
            "features_received": list(request.inputs[0].keys()),
        }

        event = models.PredictionEvent(
            inference_id=inference_id,
            candidate_id=candidate.id,
            model_name=candidate.mlflow_model_name,
            model_version=candidate.mlflow_model_version,
            request_time=request_time,
            latency_ms=latency_ms,
            status="SUCCESS",
            prediction_json={"predictions": preds_list, "count": len(preds_list), "is_diagnostic": True},
            input_summary_json=input_summary,
        )
        self.db.add(event)
        self.db.commit()

        logger.info(
            f"Diagnostic inference '{inference_id}' by '{actor}': model '{candidate.mlflow_model_name}' v{candidate.mlflow_model_version}."
        )

        return PredictionResponse(
            inference_id=inference_id,
            model_name=candidate.mlflow_model_name,
            model_version=candidate.mlflow_model_version,
            predictions=preds_list,
            latency_ms=latency_ms,
            timestamp=request_time,
        )
