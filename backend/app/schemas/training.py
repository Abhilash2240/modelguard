from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class SplitConfig(BaseModel):
    train_ratio: float = Field(0.70, ge=0.05, le=0.90, description="Proportion for training partition")
    val_ratio: float = Field(0.15, ge=0.05, le=0.50, description="Proportion for validation partition")
    test_ratio: float = Field(0.15, ge=0.05, le=0.50, description="Proportion for test partition")
    timestamp_column: str = Field("date_time", description="Name of chronological timestamp column")
    target_column: str = Field("traffic_volume", description="Name of prediction target column")
    seed: int = Field(42, description="Random seed for reproducibility")
    min_observations: int = Field(
        10,
        ge=3,
        description="Configurable minimum row count required for valid chronological splitting",
    )

    @model_validator(mode="after")
    def validate_ratios_sum(self) -> "SplitConfig":
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total})")
        return self


class PartitionManifest(BaseModel):
    name: Literal["train", "validation", "test"]
    row_count: int
    min_timestamp: str
    max_timestamp: str
    fingerprint_sha256: str


class BoundaryAdjustment(BaseModel):
    boundary: str  # e.g. "train_val" or "val_test"
    adjusted: bool
    boundary_timestamp: str | None = None
    shifted_records: int = 0
    action: str


class SplitManifest(BaseModel):
    dataset_version_id: str
    dataset_sha256: str
    split_config: SplitConfig
    total_rows: int
    partitions: dict[str, PartitionManifest]
    boundary_adjustments: list[BoundaryAdjustment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class TrainingJobConfig(BaseModel):
    model_candidate: str = "random_forest"
    split: SplitConfig = Field(default_factory=SplitConfig)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainingJobCreate(BaseModel):
    dataset_version_id: str
    validation_run_id: str | None = None
    config: TrainingJobConfig = Field(default_factory=TrainingJobConfig)


class TrainingJobResponse(BaseModel):
    id: str
    dataset_version_id: str
    validation_run_id: str | None
    config_json: dict[str, Any]
    status: str
    queue_job_id: str | None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class TrainingRunResponse(BaseModel):
    id: str
    training_job_id: str
    mlflow_run_id: str | None
    dataset_version_id: str
    git_sha: str | None
    dependency_lock_hash: str | None
    python_version: str
    random_seed: int
    split_manifest_uri: str | None
    metrics_json: dict[str, Any] | None
    artifact_uri: str | None
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MLflowRunInfo(BaseModel):
    run_id: str
    experiment_id: str
    status: str
    artifact_uri: str
    lifecycle_stage: str | None = None
    start_time: int | None = None
    end_time: int | None = None


class ReconcileReport(BaseModel):
    reconciled_runs: list[str] = Field(default_factory=list)
    orphaned_runs_failed: list[str] = Field(default_factory=list)
    ambiguous_runs: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
