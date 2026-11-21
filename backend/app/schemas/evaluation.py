from datetime import UTC, datetime

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class RegressionMetrics(BaseModel):
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    r2: float = Field(..., description="R-squared coefficient of determination")
    sample_count: int = Field(..., ge=1, description="Number of evaluated samples")


class ResidualSummary(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    p95: float


class PartitionEvaluation(BaseModel):
    model_name: str
    partition_name: str
    metrics: RegressionMetrics
    residuals: ResidualSummary
    evaluated_at: datetime = Field(default_factory=utc_now)


class ModelComparison(BaseModel):
    metric_target: str = "validation"
    baseline_metrics: RegressionMetrics
    candidate_metrics: RegressionMetrics
    delta_mae: float = Field(..., description="candidate.mae - baseline.mae (negative is better)")
    delta_rmse: float = Field(..., description="candidate.rmse - baseline.rmse (negative is better)")
    delta_r2: float = Field(..., description="candidate.r2 - baseline.r2 (positive is better)")
    candidate_superior: bool = Field(..., description="True if candidate improves on MAE and RMSE over baseline")


class FeatureMetadata(BaseModel):
    numerical_features: list[str]
    categorical_features: list[str]
    target_column: str
    excluded_columns: list[str]
    derived_features: list[str]
