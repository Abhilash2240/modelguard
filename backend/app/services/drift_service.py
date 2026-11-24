from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from app.core import metrics
from app.core.errors import EntityNotFoundError
from app.db import models
from app.services.alert_service import AlertService


def utc_now() -> datetime:
    return datetime.now(UTC)


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and current populations."""
    ref_clean = reference[np.isfinite(reference)]
    curr_clean = current[np.isfinite(current)]
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Create quantile bins on reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)  # handle discrete values
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    eps = 1e-4
    ref_pct = (ref_counts + eps) / (len(ref_clean) + eps * len(ref_counts))
    curr_pct = (curr_counts + eps) / (len(curr_clean) + eps * len(curr_counts))

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi_val)


def compute_categorical_tvd(reference_series: pd.Series, current_series: pd.Series) -> float:
    """Compute Total Variation Distance (TVD) across categorical frequency distributions."""
    ref_counts = reference_series.value_counts(normalize=True)
    curr_counts = current_series.value_counts(normalize=True)

    all_categories = set(ref_counts.index).union(set(curr_counts.index))
    tvd = 0.0
    for cat in all_categories:
        p = ref_counts.get(cat, 0.0)
        q = curr_counts.get(cat, 0.0)
        tvd += abs(p - q)
    return float(0.5 * tvd)


class DriftAnalysisService:
    """Interchangeable drift detection service analyzing input feature drift and prediction drift."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.alert_service = AlertService(db)

    def load_reference_dataset(self, dataset_version_id: str | None = None) -> pd.DataFrame:
        """Load reference baseline DataFrame from registered immutable dataset snapshot."""
        query = self.db.query(models.DatasetVersion)
        if dataset_version_id:
            version_record = query.filter(models.DatasetVersion.id == dataset_version_id).first()
        else:
            version_record = query.order_by(models.DatasetVersion.created_at.desc()).first()

        if not version_record:
            raise EntityNotFoundError("No registered DatasetVersion found for drift reference.")

        storage_path = version_record.storage_uri
        df = pd.read_csv(storage_path)
        return df

    def analyze_input_drift(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame | None = None,
        model_name: str = "metro-traffic-volume",
        model_version: str = "1",
        reference_window: tuple[datetime | None, datetime | None] = (None, None),
        current_window: tuple[datetime | None, datetime | None] = (None, None),
        alpha: float = 0.05,
        psi_threshold: float = 0.25,
        policy_version: str = "v1.0.0",
        evidence_source: str = "REAL_PRODUCTION",
    ) -> models.MonitoringReport:
        """Analyze numeric, categorical, and data-quality drift across input features.

        Strict governance rule: Records explicit limitations stating that feature drift
        measures distribution divergence and does not constitute proof of model performance degradation.
        """
        if reference_df is None:
            reference_df = self.load_reference_dataset()

        numeric_features = ["temp", "rain_1h", "snow_1h", "clouds_all"]
        categorical_features = ["holiday", "weather_main", "weather_description"]

        feature_metrics: list[dict[str, Any]] = []
        drifting_features: list[str] = []

        # 1. Evaluate Numeric Features
        for feat in numeric_features:
            if feat in reference_df.columns and feat in current_df.columns:
                ref_s = pd.to_numeric(reference_df[feat], errors="coerce")
                curr_s = pd.to_numeric(current_df[feat], errors="coerce")

                ref_clean = ref_s.dropna().to_numpy()
                curr_clean = curr_s.dropna().to_numpy()

                missing_ref = float(ref_s.isna().mean())
                missing_curr = float(curr_s.isna().mean())

                # Two-sample Kolmogorov-Smirnov test
                if len(ref_clean) > 0 and len(curr_clean) > 0:
                    ks_res = stats.ks_2samp(ref_clean, curr_clean)
                    ks_stat = float(ks_res.statistic)
                    p_val = float(ks_res.pvalue)
                    psi_val = compute_psi(ref_clean, curr_clean)
                else:
                    ks_stat = 0.0
                    p_val = 1.0
                    psi_val = 0.0

                is_drift = bool(p_val < alpha or psi_val > psi_threshold or abs(missing_curr - missing_ref) > 0.10)
                if is_drift:
                    drifting_features.append(feat)

                feature_metrics.append(
                    {
                        "feature_name": feat,
                        "feature_type": "numeric",
                        "detection_method": "KS_TEST_AND_PSI",
                        "test_statistic": ks_stat,
                        "p_value": p_val,
                        "psi": psi_val,
                        "threshold": alpha,
                        "is_drift_detected": is_drift,
                        "missing_rate_reference": missing_ref,
                        "missing_rate_current": missing_curr,
                    }
                )

        # 2. Evaluate Categorical Features
        for feat in categorical_features:
            if feat in reference_df.columns and feat in current_df.columns:
                # Calculate missingness strictly BEFORE string conversion to prevent NaN string masking
                missing_ref = float(reference_df[feat].isna().mean())
                missing_curr = float(current_df[feat].isna().mean())

                ref_s = reference_df[feat].dropna().astype(str)
                curr_s = current_df[feat].dropna().astype(str)

                tvd = compute_categorical_tvd(ref_s, curr_s)
                unseen_categories = set(curr_s.unique()) - set(ref_s.unique())
                # TVD > 0.20, missing rate change, or unobserved categories indicates category drift
                is_drift = bool(tvd > 0.20 or abs(missing_curr - missing_ref) > 0.10 or len(unseen_categories) > 0)
                if is_drift:
                    drifting_features.append(feat)

                feature_metrics.append(
                    {
                        "feature_name": feat,
                        "feature_type": "categorical",
                        "detection_method": "TOTAL_VARIATION_DISTANCE",
                        "test_statistic": tvd,
                        "p_value": None,
                        "threshold": 0.20,
                        "is_drift_detected": is_drift,
                        "missing_rate_reference": missing_ref,
                        "missing_rate_current": missing_curr,
                        "unseen_categories": list(unseen_categories),
                    }
                )

        is_overall_drift = len(drifting_features) > 0
        now = utc_now()

        summary = {
            "total_features_evaluated": len(feature_metrics),
            "drifting_feature_count": len(drifting_features),
            "drifting_features": drifting_features,
            "significance_level": alpha,
            "psi_threshold": psi_threshold,
            "evidence_source": evidence_source,
        }

        limitations_text = (
            "Input drift measures feature-level covariate divergence between reference and current observation windows. "
            "Input distribution changes do NOT prove model accuracy degradation without ground-truth verification."
        )

        report = models.MonitoringReport(
            report_type="DATA_DRIFT",
            model_name=model_name,
            model_version=model_version,
            reference_window_start=reference_window[0],
            reference_window_end=reference_window[1],
            current_window_start=current_window[0],
            current_window_end=current_window[1],
            reference_sample_count=len(reference_df),
            current_sample_count=len(current_df),
            detection_method="KS_TEST_PSI_AND_TVD",
            policy_version=policy_version,
            is_drift_detected=is_overall_drift,
            summary_json=summary,
            metrics_json={"features": feature_metrics},
            limitations=limitations_text,
            created_at=now,
        )
        self.db.add(report)
        self.db.flush()

        # Update Prometheus drift status
        metrics.set_drift_status(model_name, model_version, "data_drift", is_overall_drift)

        # Trigger alert if drift detected
        if is_overall_drift:
            severity = "CRITICAL" if len(drifting_features) >= 3 else "WARNING"
            self.alert_service.create_alert(
                model_name=model_name,
                model_version=model_version,
                alert_type="DATA_DRIFT",
                severity=severity,
                message=f"Input data drift detected on {len(drifting_features)} feature(s): {', '.join(drifting_features)}",
                triggering_evidence={
                    "drifting_features": drifting_features,
                    "drifting_count": len(drifting_features),
                    "summary": summary,
                },
                report_id=report.id,
            )

        self.db.commit()
        self.db.refresh(report)
        return report

    def analyze_prediction_drift(
        self,
        current_predictions: np.ndarray | list[float],
        reference_predictions: np.ndarray | list[float],
        model_name: str = "metro-traffic-volume",
        model_version: str = "1",
        reference_window: tuple[datetime | None, datetime | None] = (None, None),
        current_window: tuple[datetime | None, datetime | None] = (None, None),
        alpha: float = 0.05,
        psi_threshold: float = 0.25,
        policy_version: str = "v1.0.0",
        evidence_source: str = "REAL_PRODUCTION",
    ) -> models.MonitoringReport:
        """Analyze prediction distribution drift strictly separated from input drift.

        Records statistical summaries, tests, and explicit limitations distinguishing prediction drift
        from ground-truth quality degradation.
        """
        ref_arr = np.array(reference_predictions, dtype=float)
        curr_arr = np.array(current_predictions, dtype=float)

        ref_clean = ref_arr[np.isfinite(ref_arr)]
        curr_clean = curr_arr[np.isfinite(curr_arr)]

        if len(ref_clean) > 0 and len(curr_clean) > 0:
            ks_res = stats.ks_2samp(ref_clean, curr_clean)
            ks_stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)
            psi_val = compute_psi(ref_clean, curr_clean)
        else:
            ks_stat = 0.0
            p_val = 1.0
            psi_val = 0.0

        is_drift_detected = bool(p_val < alpha or psi_val > psi_threshold)

        summary = {
            "reference_mean": float(np.mean(ref_clean)) if len(ref_clean) > 0 else 0.0,
            "current_mean": float(np.mean(curr_clean)) if len(curr_clean) > 0 else 0.0,
            "reference_std": float(np.std(ref_clean)) if len(ref_clean) > 0 else 0.0,
            "current_std": float(np.std(curr_clean)) if len(curr_clean) > 0 else 0.0,
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "psi": psi_val,
            "alpha": alpha,
            "psi_threshold": psi_threshold,
            "evidence_source": evidence_source,
        }

        limitations_text = (
            "Prediction drift measures shifts in model output distribution across observation windows. "
            "Output shifts may indicate seasonality or macro demand changes and must not be conflated "
            "with input drift or verified model degradation."
        )

        now = utc_now()
        report = models.MonitoringReport(
            report_type="PREDICTION_DRIFT",
            model_name=model_name,
            model_version=model_version,
            reference_window_start=reference_window[0],
            reference_window_end=reference_window[1],
            current_window_start=current_window[0],
            current_window_end=current_window[1],
            reference_sample_count=len(ref_clean),
            current_sample_count=len(curr_clean),
            detection_method="KS_TEST_AND_PSI",
            policy_version=policy_version,
            is_drift_detected=is_drift_detected,
            summary_json=summary,
            metrics_json={"prediction_metrics": summary},
            limitations=limitations_text,
            created_at=now,
        )
        self.db.add(report)
        self.db.flush()

        # Update Prometheus prediction drift status
        metrics.set_drift_status(model_name, model_version, "prediction_drift", is_drift_detected)

        if is_drift_detected:
            self.alert_service.create_alert(
                model_name=model_name,
                model_version=model_version,
                alert_type="PREDICTION_DRIFT",
                severity="WARNING",
                message=(
                    f"Prediction distribution shift detected for model {model_name}:v{model_version} "
                    f"(p_value={p_val:.4e}, PSI={psi_val:.4f})."
                ),
                triggering_evidence=summary,
                report_id=report.id,
            )

        self.db.commit()
        self.db.refresh(report)
        return report

    def list_reports(
        self,
        report_type: str | None = None,
        model_version: str | None = None,
        limit: int = 50,
    ) -> list[models.MonitoringReport]:
        query = self.db.query(models.MonitoringReport)
        if report_type:
            query = query.filter(models.MonitoringReport.report_type == report_type)
        if model_version:
            query = query.filter(models.MonitoringReport.model_version == model_version)
        return query.order_by(models.MonitoringReport.created_at.desc()).limit(limit).all()
