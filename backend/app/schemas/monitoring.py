from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    """Schema for submitting delayed ground-truth feedback."""

    inference_id: str = Field(..., description="Unique inference identifier returned by /api/v1/predict")
    actual_value: float = Field(..., description="Ground truth target value (e.g. actual traffic volume)")
    submitted_by: str = Field("operator", description="Identifier of the system or operator providing ground truth")
    notes: str | None = Field(None, description="Optional diagnostic notes")


class FeedbackResponse(BaseModel):
    """Schema for feedback confirmation response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    inference_id: str
    actual_value: float
    submitted_at: datetime
    submitted_by: str
    notes: str | None = None


class QualityReportResponse(BaseModel):
    """Schema for rolling model quality assessment."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    model_version: str
    window_start: datetime | None = None
    window_end: datetime | None = None
    sample_count: int
    labeled_sample_count: int
    label_coverage: float
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    status: str  # SUFFICIENT_LABELS, INSUFFICIENT_LABELS
    details_json: dict[str, Any] | None = None
    created_at: datetime


class DriftAnalysisRequest(BaseModel):
    """Parameters for triggering data or prediction drift analysis."""

    model_version: str | None = Field(
        None, description="Model version to analyze. Defaults to active production champion."
    )
    reference_window_start: datetime | None = None
    reference_window_end: datetime | None = None
    current_window_start: datetime | None = None
    current_window_end: datetime | None = None
    significance_level: float = Field(0.05, description="Significance level alpha for statistical tests (e.g. KS-test)")
    psi_threshold: float = Field(0.25, description="PSI threshold indicating significant drift")
    policy_version: str = Field("v1.0.0", description="Version of the drift policy rules")


class FeatureDriftResult(BaseModel):
    """Per-feature drift measurement."""

    feature_name: str
    feature_type: str  # numeric, categorical
    detection_method: str  # KS_TEST, PSI, CHI_SQUARE, TOTAL_VARIATION_DISTANCE
    test_statistic: float
    p_value: float | None = None
    threshold: float
    is_drift_detected: bool
    missing_rate_reference: float = 0.0
    missing_rate_current: float = 0.0


class DriftReportResponse(BaseModel):
    """Schema for persistent drift monitoring reports."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    report_type: str  # DATA_DRIFT, PREDICTION_DRIFT
    model_name: str
    model_version: str
    reference_sample_count: int
    current_sample_count: int
    detection_method: str
    policy_version: str
    is_drift_detected: bool
    summary_json: dict[str, Any]
    metrics_json: dict[str, Any] | None = None
    limitations: str | None = None
    created_at: datetime


class AlertResponse(BaseModel):
    """Schema for monitoring alerts."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    report_id: str | None = None
    model_name: str
    model_version: str
    alert_type: str  # DATA_DRIFT, PREDICTION_DRIFT, QUALITY_DEGRADATION, SERVING_FAILURE
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    triggering_evidence_json: dict[str, Any]
    status: str  # OPEN, ACKNOWLEDGED, RESOLVED
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    resolution_notes: str | None = None
    created_at: datetime


class AlertAcknowledgeRequest(BaseModel):
    actor: str = Field("operator", description="Identifier of the operator acknowledging the alert")


class AlertResolveRequest(BaseModel):
    actor: str = Field("operator", description="Identifier of the operator resolving the alert")
    notes: str = Field("", description="Resolution rationale or remediation actions taken")
