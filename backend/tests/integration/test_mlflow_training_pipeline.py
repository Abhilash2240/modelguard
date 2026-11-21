import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from app.core.errors import InvalidOperationError
from app.db import models
from app.domain.hashing import calculate_schema_hash, calculate_sha256
from app.ml.features import prepare_features
from app.ml.tracking import MLflowTracker, NullTracker
from app.schemas.training import SplitConfig, TrainingJobConfig, TrainingJobCreate
from app.services.training_job_service import TrainingJobService


@pytest.fixture
def test_tracking_env():
    """Create a temporary directory for isolated SQLite MLflow tracking store and artifacts."""
    tmp = Path(tempfile.mkdtemp())
    db_file = tmp / "test_mlflow.db"
    art_dir = tmp / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    artifact_root = art_dir.as_uri()
    experiment_name = "test-metro-traffic-exp"

    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_root=artifact_root,
    )

    yield {
        "tracker": tracker,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "artifact_root": artifact_root,
        "tmp_dir": tmp,
    }

    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def dataset_setup(db: Session, valid_csv_path: Path):
    """Seed prerequisite Dataset, DatasetVersion, and ValidationRun."""
    dataset = models.Dataset(
        name="metro_traffic_integration",
        task_type="regression",
        source_uri="https://archive.ics.uci.edu/dataset/492/urban+traffic",
        licence="CC BY 4.0",
    )
    db.add(dataset)
    db.commit()

    df = pd.read_csv(valid_csv_path)
    sha256 = calculate_sha256(valid_csv_path)
    cols = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
    schema_hash = calculate_schema_hash(cols)

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.1",
        storage_uri=valid_csv_path.as_posix(),
        sha256_hash=sha256,
        byte_size=valid_csv_path.stat().st_size,
        row_count=len(df),
        columns_json=cols,
        schema_hash=schema_hash,
    )
    db.add(version)
    db.commit()

    val_run = models.ValidationRun(
        dataset_version_id=version.id,
        status="PASSED",
        error_count=0,
        warning_count=0,
        details_json={"findings": []},
    )
    db.add(val_run)
    db.commit()

    return {"dataset": dataset, "version": version, "validation_run": val_run}


def test_end_to_end_job_execution_one_to_one_mapping(
    db: Session, dataset_setup, test_tracking_env, valid_csv_path: Path
):
    version = dataset_setup["version"]
    val_run = dataset_setup["validation_run"]
    tracker = test_tracking_env["tracker"]

    service = TrainingJobService(db=db, tracker=tracker)

    # 1. Create TrainingJob
    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job_create = TrainingJobCreate(
        dataset_version_id=version.id,
        validation_run_id=val_run.id,
        config=TrainingJobConfig(
            model_candidate="random_forest",
            split=split_cfg,
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        ),
    )
    job = service.create_training_job(job_create)
    assert job.id is not None
    assert job.status == "QUEUED"

    # 2. Execute TrainingJob
    run = service.execute_training_job(job.id)
    assert run.status == "SUCCEEDED"
    assert job.status == "SUCCEEDED"
    assert run.training_job_id == job.id
    assert run.mlflow_run_id is not None

    # 3. Verify MLflow Tracking Store
    client = MlflowClient(test_tracking_env["tracking_uri"])
    m_run = client.get_run(run.mlflow_run_id)
    assert m_run.info.status == "FINISHED"

    # Verify Tags
    assert m_run.data.tags["modelguard.training_job_id"] == job.id
    assert m_run.data.tags["modelguard.training_run_id"] == run.id
    assert m_run.data.tags["modelguard.dataset_version_id"] == version.id

    # Verify Parameters
    assert "candidate_n_estimators" in m_run.data.params
    assert m_run.data.params["candidate_n_estimators"] == "10"
    assert "baseline_strategy" in m_run.data.params
    assert m_run.data.params["baseline_strategy"] == "mean"

    # Verify Metrics
    assert "candidate_val_mae" in m_run.data.metrics
    assert "candidate_test_mae" in m_run.data.metrics
    assert "baseline_val_mae" in m_run.data.metrics
    assert "comparison_delta_mae" in m_run.data.metrics


def test_separate_model_artifacts_in_single_run(db: Session, dataset_setup, test_tracking_env):
    version = dataset_setup["version"]
    tracker = test_tracking_env["tracker"]
    service = TrainingJobService(db=db, tracker=tracker)

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg, hyperparameters={"n_estimators": 5}),
        )
    )
    run = service.execute_training_job(job.id)

    client = MlflowClient(test_tracking_env["tracking_uri"])
    artifacts = [a.path for a in client.list_artifacts(run.mlflow_run_id)]

    # Both candidate_model and baseline_model must be logged in the same run
    assert "candidate_model" in artifacts
    assert "baseline_model" in artifacts
    assert "manifests" in artifacts
    assert "predictions" in artifacts
    assert "evaluations" in artifacts


def test_model_reloading_and_prediction_parity(db: Session, dataset_setup, test_tracking_env, valid_csv_path: Path):
    version = dataset_setup["version"]
    tracker = test_tracking_env["tracker"]
    service = TrainingJobService(db=db, tracker=tracker)

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg, hyperparameters={"n_estimators": 10, "max_depth": 4}),
        )
    )
    run = service.execute_training_job(job.id)

    # Load logged candidate model via MLflow
    mlflow.set_tracking_uri(test_tracking_env["tracking_uri"])
    reloaded_candidate = mlflow.sklearn.load_model(f"runs:/{run.mlflow_run_id}/candidate_model")
    reloaded_baseline = mlflow.sklearn.load_model(f"runs:/{run.mlflow_run_id}/baseline_model")

    df = pd.read_csv(valid_csv_path)
    X, _ = prepare_features(df, target_col="traffic_volume")

    preds_candidate = reloaded_candidate.predict(X)
    preds_baseline = reloaded_baseline.predict(X)

    assert len(preds_candidate) == len(df)
    assert len(preds_baseline) == len(df)
    assert np.all(np.isfinite(preds_candidate))
    assert np.all(np.isfinite(preds_baseline))


def test_database_to_mlflow_metric_reconciliation(db: Session, dataset_setup, test_tracking_env):
    version = dataset_setup["version"]
    tracker = test_tracking_env["tracker"]
    service = TrainingJobService(db=db, tracker=tracker)

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg, hyperparameters={"n_estimators": 10}),
        )
    )
    run = service.execute_training_job(job.id)

    client = MlflowClient(test_tracking_env["tracking_uri"])
    m_run = client.get_run(run.mlflow_run_id)

    db_metrics = run.metrics_json
    mlflow_metrics = m_run.data.metrics

    for metric_name in ("candidate_val_mae", "candidate_test_mae", "baseline_val_mae", "comparison_delta_mae"):
        assert metric_name in db_metrics
        assert metric_name in mlflow_metrics
        assert db_metrics[metric_name] == pytest.approx(mlflow_metrics[metric_name], rel=1e-5)


def test_duplicate_execution_rejected(db: Session, dataset_setup, test_tracking_env):
    version = dataset_setup["version"]
    tracker = test_tracking_env["tracker"]
    service = TrainingJobService(db=db, tracker=tracker)

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg),
        )
    )

    # 1. First execution succeeds
    service.execute_training_job(job.id)
    assert job.status == "SUCCEEDED"

    # 2. Second execution attempt must raise InvalidOperationError
    with pytest.raises(InvalidOperationError) as exc_info:
        service.execute_training_job(job.id)
    assert "expected 'QUEUED'" in str(exc_info.value)


def test_retry_creates_distinct_job_and_run(db: Session, dataset_setup, test_tracking_env):
    version = dataset_setup["version"]
    tracker = test_tracking_env["tracker"]
    service = TrainingJobService(db=db, tracker=tracker)

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg),
        )
    )

    # Simulate job failure
    job.status = "FAILED"
    db.commit()

    # Retry job
    new_job = service.retry_training_job(job.id)
    assert new_job.id != job.id
    assert new_job.status == "QUEUED"
    assert new_job.dataset_version_id == job.dataset_version_id

    # Verify audit trail records retry lineage
    retry_audit = (
        db.query(models.AuditEvent)
        .filter(models.AuditEvent.resource_id == new_job.id, models.AuditEvent.action == "RETRY_TRAINING_JOB")
        .first()
    )
    assert retry_audit is not None
    assert retry_audit.before_state_json["original_job_id"] == job.id

    # Execute new job
    new_run = service.execute_training_job(new_job.id)
    assert new_run.status == "SUCCEEDED"
    assert new_run.training_job_id == new_job.id


def test_reconciliation_unlinked_run_recovery(db: Session, dataset_setup, test_tracking_env):
    version = dataset_setup["version"]
    tracker = test_tracking_env["tracker"]
    service = TrainingJobService(db=db, tracker=tracker)

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg),
        )
    )

    run = service.execute_training_job(job.id)
    saved_mlflow_id = run.mlflow_run_id

    # Simulate unlinked state by wiping mlflow_run_id and setting status RUNNING
    run.mlflow_run_id = None
    run.status = "RUNNING"
    db.commit()

    # Reconcile
    report = service.reconcile_orphaned_runs()
    db.refresh(run)

    # Must discover and re-link the MLflow run via tags
    assert run.mlflow_run_id == saved_mlflow_id
    assert run.id in report.reconciled_runs


def test_reconciliation_finalization_recovery(db: Session, dataset_setup, test_tracking_env):
    version = dataset_setup["version"]
    tracker = test_tracking_env["tracker"]
    service = TrainingJobService(db=db, tracker=tracker)

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg),
        )
    )
    run = service.execute_training_job(job.id)

    # Manually transition MLflow run back to RUNNING
    client = MlflowClient(test_tracking_env["tracking_uri"])
    client.update_run(run.mlflow_run_id, status="RUNNING")
    assert client.get_run(run.mlflow_run_id).info.status == "RUNNING"

    # Reconcile
    report = service.reconcile_orphaned_runs()
    m_run = client.get_run(run.mlflow_run_id)

    # MLflow run must be idempotently finalized to FINISHED
    assert m_run.info.status == "FINISHED"
    assert m_run.data.tags.get("modelguard.reconciliation") == "finalized_post_execution"
    assert run.id in report.reconciled_runs


def test_tracking_disabled_mode_in_service(db: Session, dataset_setup):
    version = dataset_setup["version"]
    service = TrainingJobService(db=db, tracker=NullTracker())

    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            config=TrainingJobConfig(split=split_cfg),
        )
    )
    run = service.execute_training_job(job.id)

    # In disabled mode, execution succeeds with mlflow_run_id = None
    assert run.status == "SUCCEEDED"
    assert run.mlflow_run_id is None
    assert run.metrics_json is not None
