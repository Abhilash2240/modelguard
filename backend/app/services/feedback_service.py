import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sqlalchemy.orm import Session

from app.core import metrics
from app.core.errors import (
    EntityNotFoundError,
    FeedbackConflictError,
)
from app.core.logging import logger
from app.db import models
from app.services.alert_service import AlertService


def utc_now() -> datetime:
    return datetime.now(UTC)


class FeedbackService:
    """Service managing delayed ground-truth feedback, idempotency, and rolling quality calculations."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.alert_service = AlertService(db)

    def submit_feedback(
        self,
        inference_id: str,
        actual_value: float,
        submitted_by: str = "operator",
        notes: str | None = None,
    ) -> models.PredictionFeedback:
        """Submit ground-truth label for a past inference event with strict idempotency."""
        # 1. Enforce inference ID validity
        event = (
            self.db.query(models.PredictionEvent).filter(models.PredictionEvent.inference_id == inference_id).first()
        )
        if not event:
            raise EntityNotFoundError(f"Inference record with inference_id '{inference_id}' not found.")

        # 2. Check for existing feedback
        existing = (
            self.db.query(models.PredictionFeedback)
            .filter(models.PredictionFeedback.inference_id == inference_id)
            .first()
        )
        if existing:
            # Idempotency check: if values match within tolerance, return existing record
            if math.isclose(existing.actual_value, actual_value, rel_tol=1e-5, abs_tol=1e-5):
                logger.info(
                    f"Idempotent feedback re-submission for inference '{inference_id}' (actual_value={actual_value})."
                )
                return existing
            # Conflicting label submitted: reject
            raise FeedbackConflictError(
                f"Conflicting ground-truth value ({actual_value}) submitted for inference '{inference_id}'. "
                f"Previously recorded ground truth was {existing.actual_value}."
            )

        # 3. Create and persist new feedback (preserving exact original prediction record untouched)
        feedback = models.PredictionFeedback(
            inference_id=inference_id,
            actual_value=float(actual_value),
            submitted_by=submitted_by,
            notes=notes,
            submitted_at=utc_now(),
        )
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)

        logger.info(
            f"Recorded ground truth ({actual_value}) for inference '{inference_id}' on model {event.model_name}:v{event.model_version}."
        )
        return feedback

    def get_feedback(self, inference_id: str) -> models.PredictionFeedback | None:
        return (
            self.db.query(models.PredictionFeedback)
            .filter(models.PredictionFeedback.inference_id == inference_id)
            .first()
        )

    def calculate_rolling_quality(
        self,
        model_name: str = "metro-traffic-volume",
        model_version: str | None = None,
        min_samples: int = 10,
        max_mae_threshold: float = 450.0,
        min_r2_threshold: float = 0.70,
    ) -> models.ModelQualityReport:
        """Calculate rolling MAE, RMSE, and R2 against delayed feedback.

        Handles insufficient labels explicitly to prevent misleading quality metrics.
        """
        # Resolve target model version if not specified (resolve active production champion)
        if not model_version:
            prod_env = (
                self.db.query(models.DeploymentEnvironment)
                .filter(models.DeploymentEnvironment.name == "production")
                .first()
            )
            if prod_env and prod_env.active_candidate:
                model_version = prod_env.active_candidate.mlflow_model_version
            else:
                model_version = "1"

        # Query prediction events for this model version
        events_query = self.db.query(models.PredictionEvent).filter(
            models.PredictionEvent.model_name == model_name,
            models.PredictionEvent.model_version == model_version,
            models.PredictionEvent.status == "SUCCESS",
        )
        total_sample_count = events_query.count()

        # Join with prediction_feedbacks
        joined_records = (
            self.db.query(models.PredictionEvent, models.PredictionFeedback)
            .join(
                models.PredictionFeedback,
                models.PredictionEvent.inference_id == models.PredictionFeedback.inference_id,
            )
            .filter(
                models.PredictionEvent.model_name == model_name,
                models.PredictionEvent.model_version == model_version,
                models.PredictionEvent.status == "SUCCESS",
            )
            .all()
        )

        labeled_sample_count = len(joined_records)
        label_coverage = float(labeled_sample_count / total_sample_count * 100.0) if total_sample_count > 0 else 0.0

        now = utc_now()

        # Handle Insufficient Labels Explicitly
        if labeled_sample_count < min_samples:
            status = "INSUFFICIENT_LABELS"
            report = models.ModelQualityReport(
                model_name=model_name,
                model_version=model_version,
                window_start=None,
                window_end=now,
                sample_count=total_sample_count,
                labeled_sample_count=labeled_sample_count,
                label_coverage=label_coverage,
                mae=None,
                rmse=None,
                r2=None,
                status=status,
                details_json={
                    "message": (
                        f"Insufficient labeled samples ({labeled_sample_count} < {min_samples}). "
                        "Quality metrics deferred to prevent statistical noise."
                    ),
                    "min_samples_required": min_samples,
                },
                created_at=now,
            )
            self.db.add(report)
            self.db.commit()
            self.db.refresh(report)
            return report

        # Extract predictions and actuals
        y_true = []
        y_pred = []
        for event, feedback in joined_records:
            preds = event.prediction_json.get("predictions") if event.prediction_json else None
            if preds and len(preds) > 0:
                y_pred.append(float(preds[0]))
                y_true.append(float(feedback.actual_value))

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))

        # Check R2 score validity (needs variance in y_true)
        if len(y_true_arr) >= 2 and np.var(y_true_arr) > 1e-9:
            r2 = float(r2_score(y_true_arr, y_pred_arr))
        else:
            r2 = 0.0

        status = "SUFFICIENT_LABELS"
        details: dict[str, Any] = {
            "min_samples_required": min_samples,
            "y_true_mean": float(np.mean(y_true_arr)),
            "y_pred_mean": float(np.mean(y_pred_arr)),
            "mae_threshold": max_mae_threshold,
            "r2_threshold": min_r2_threshold,
        }

        # Record metrics in Prometheus
        metrics.record_quality_metric(model_name, model_version, "mae", mae)
        metrics.record_quality_metric(model_name, model_version, "rmse", rmse)
        metrics.record_quality_metric(model_name, model_version, "r2", r2)

        report = models.ModelQualityReport(
            model_name=model_name,
            model_version=model_version,
            window_start=None,
            window_end=now,
            sample_count=total_sample_count,
            labeled_sample_count=labeled_sample_count,
            label_coverage=label_coverage,
            mae=mae,
            rmse=rmse,
            r2=r2,
            status=status,
            details_json=details,
            created_at=now,
        )
        self.db.add(report)
        self.db.flush()

        # Check if quality degraded beyond thresholds
        degraded = False
        reasons = []
        if mae > max_mae_threshold:
            degraded = True
            reasons.append(f"MAE ({mae:.2f}) exceeded threshold ({max_mae_threshold:.2f})")
        if r2 < min_r2_threshold:
            degraded = True
            reasons.append(f"R² ({r2:.4f}) fell below threshold ({min_r2_threshold:.4f})")

        if degraded:
            self.alert_service.create_alert(
                model_name=model_name,
                model_version=model_version,
                alert_type="QUALITY_DEGRADATION",
                severity="CRITICAL" if mae > max_mae_threshold * 1.25 else "WARNING",
                message=f"Model quality degraded: {'; '.join(reasons)}",
                triggering_evidence={
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "labeled_sample_count": labeled_sample_count,
                    "reasons": reasons,
                },
                report_id=report.id,
            )

        self.db.commit()
        self.db.refresh(report)
        return report

    def list_quality_reports(
        self,
        model_version: str | None = None,
        limit: int = 50,
    ) -> list[models.ModelQualityReport]:
        query = self.db.query(models.ModelQualityReport)
        if model_version:
            query = query.filter(models.ModelQualityReport.model_version == model_version)
        return query.order_by(models.ModelQualityReport.created_at.desc()).limit(limit).all()
