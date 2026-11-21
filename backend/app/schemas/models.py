from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReleaseGatePolicy(BaseModel):
    """Configurable release gate criteria and thresholds."""

    version: str = "v1.0.0"
    max_val_mae: float = 400.0
    max_val_rmse: float = 600.0
    min_val_r2: float = 0.85
    min_mae_improvement_pct: float = 20.0
    max_regression_pct_vs_champion: float = 5.0
    require_dataset_validation: bool = True
    require_training_succeeded: bool = True


class GateEvaluationResult(BaseModel):
    """Auditable result of evaluating a model candidate against a ReleaseGatePolicy."""

    passed: bool
    policy_version: str
    evaluated_metrics: dict[str, Any] = Field(default_factory=dict)
    criteria_results: dict[str, bool] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)


class ModelCandidateCreate(BaseModel):
    training_run_id: str
    model_name: str | None = None


class ModelCandidateResponse(BaseModel):
    id: str
    training_run_id: str
    dataset_version_id: str
    mlflow_model_name: str
    mlflow_model_version: str
    model_uri: str
    artifact_uri: str
    signature_json: dict[str, Any] | None = None
    input_contract_json: dict[str, Any] | None = None
    metrics_json: dict[str, Any] | None = None
    gate_evaluation_json: dict[str, Any] | None = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
