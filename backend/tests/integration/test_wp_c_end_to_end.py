import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from app.db import models
from app.domain.hashing import calculate_schema_hash, calculate_sha256
from app.ml.tracking import MLflowTracker
from app.schemas.inference import PredictionRequest
from app.schemas.models import ReleaseGatePolicy
from app.schemas.training import SplitConfig, TrainingJobConfig, TrainingJobCreate
from app.services.deployment_service import DeploymentService
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService
from app.services.training_job_service import TrainingJobService


@pytest.fixture
def tracking_env():
    tmp = Path(tempfile.mkdtemp())
    db_file = tmp / "test_mlflow.db"
    art_dir = tmp / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    artifact_root = art_dir.as_uri()
    experiment_name = "wp-c-integration-exp"

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
def trained_run(db: Session, valid_csv_path: Path, tracking_env):
    """Execute a real TrainingJob with MLflowTracker to produce a valid TrainingRun."""
    dataset = models.Dataset(
        name="metro_traffic_wpc",
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
        details_json={"checks_passed": 10},
    )
    db.add(val_run)
    db.commit()

    # Train model via TrainingJobService
    job_service = TrainingJobService(db=db, tracker=tracking_env["tracker"])
    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job = job_service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            validation_run_id=val_run.id,
            config=TrainingJobConfig(split=split_cfg, hyperparameters={"n_estimators": 10, "max_depth": 4}),
        )
    )
    run = job_service.execute_training_job(job.id)
    assert run.status == "SUCCEEDED"

    return {"dataset": dataset, "version": version, "val_run": val_run, "job": job, "run": run}


def test_full_wp_c_lifecycle_end_to_end(db: Session, trained_run, tracking_env, client: TestClient):
    """Complete Work Package C lifecycle:

    Register -> Release Gate -> Staging -> Smoke Check -> Approve -> Promote -> Predict -> Telemetry.
    """
    run = trained_run["run"]
    version = trained_run["version"]
    tracking_uri = tracking_env["tracking_uri"]

    reg_service = ModelRegistryService(db=db, tracking_uri=tracking_uri)
    dep_service = DeploymentService(db=db, tracking_uri=tracking_uri)
    inf_service = InferenceService(db=db, tracking_uri=tracking_uri)

    # 1. Register candidate model in MLflow Model Registry and ModelGuard DB
    candidate = reg_service.register_candidate(
        training_run_id=run.id,
        model_name="metro-traffic-volume",
    )
    assert candidate.id is not None
    assert candidate.status == "REGISTERED"
    assert candidate.mlflow_model_version == "1"
    assert candidate.training_run_id == run.id
    assert candidate.dataset_version_id == version.id

    # Verify MLflow Model Registry holds version 1
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    mv = mlflow_client.get_model_version("metro-traffic-volume", "1")
    assert str(mv.version) == "1"

    # 2. Evaluate Release Gate
    test_policy = ReleaseGatePolicy(
        version="integration-test-policy",
        max_val_mae=5000.0,
        max_val_rmse=5000.0,
        min_val_r2=-100.0,
        min_mae_improvement_pct=-100.0,
    )
    gate_result = reg_service.evaluate_release_gate(candidate.id, policy=test_policy)
    assert gate_result.passed is True
    assert len(gate_result.rejection_reasons) == 0
    assert candidate.status == "GATE_PASSED"

    # 3. Stage candidate and execute real smoke checks
    staging_dep = dep_service.stage_candidate(candidate.id)
    assert staging_dep.status == "AWAITING_APPROVAL"
    assert staging_dep.smoke_test_results_json["passed"] is True
    assert staging_dep.smoke_test_results_json["model_loaded"] is True
    assert staging_dep.smoke_test_results_json["sample_output_finite"] is True
    assert candidate.status == "AWAITING_APPROVAL"

    # 4. Explicit Approval
    approved_dep = dep_service.approve_deployment(
        deployment_id=staging_dep.id,
        actor="senior-mlops-engineer",
        rationale="Meets all operational thresholds and passed staging smoke test",
    )
    assert approved_dep.status == "APPROVED"
    assert approved_dep.approved_by == "senior-mlops-engineer"
    assert candidate.status == "APPROVED"

    # 5. Promote to Production Champion
    prod_dep = dep_service.promote_deployment(
        deployment_id=staging_dep.id,
        actor="release-manager",
    )
    assert prod_dep.status == "PROMOTED"
    assert candidate.status == "PROMOTED"

    # Verify MLflow alias 'champion' points to version 1
    champion_mv = mlflow_client.get_model_version_by_alias("metro-traffic-volume", "champion")
    assert str(champion_mv.version) == "1"

    # 6. Version-Aware Prediction
    pred_payload = [
        {
            "holiday": "None",
            "temp": 288.28,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "weather_main": "Clear",
            "weather_description": "sky is clear",
            "date_time": "2026-11-21 09:00:00",
        },
        {
            "holiday": "None",
            "temp": 290.15,
            "rain_1h": 0.5,
            "snow_1h": 0.0,
            "clouds_all": 75,
            "weather_main": "Rain",
            "weather_description": "light rain",
            "date_time": "2026-11-21 17:00:00",
        },
    ]

    pred_res = inf_service.predict(PredictionRequest(inputs=pred_payload))
    assert pred_res.model_name == "metro-traffic-volume"
    assert pred_res.model_version == "1"
    assert len(pred_res.predictions) == 2
    assert all(p > 0 for p in pred_res.predictions)
    assert pred_res.latency_ms > 0

    # 7. Privacy-Safe Telemetry Verification
    event = (
        db.query(models.PredictionEvent).filter(models.PredictionEvent.inference_id == pred_res.inference_id).first()
    )
    assert event is not None
    assert event.status == "SUCCESS"
    assert event.model_version == "1"
    assert event.input_summary_json["row_count"] == 2
    # Ensure raw input fields are not present in telemetry
    assert "temp" not in event.input_summary_json
    assert "date_time" not in event.input_summary_json


def test_rejection_leaves_production_model_unchanged(db: Session, trained_run, tracking_env):
    """Verify that rejecting a candidate leaves the active production model completely unchanged."""
    run = trained_run["run"]
    tracking_uri = tracking_env["tracking_uri"]

    reg_service = ModelRegistryService(db=db, tracking_uri=tracking_uri)
    dep_service = DeploymentService(db=db, tracking_uri=tracking_uri)

    # 1. Promote v1 as champion
    c1 = reg_service.register_candidate(run.id, "metro-traffic-volume")
    test_policy = ReleaseGatePolicy(
        version="integration-test-policy",
        max_val_mae=5000.0,
        max_val_rmse=5000.0,
        min_val_r2=-100.0,
        min_mae_improvement_pct=-100.0,
    )
    reg_service.evaluate_release_gate(c1.id, policy=test_policy)
    d1 = dep_service.stage_candidate(c1.id)
    dep_service.approve_deployment(d1.id, "approver-1", "Approve v1")
    dep_service.promote_deployment(d1.id, "promoter-1")

    prod_env = db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "production").first()
    assert prod_env.active_candidate_id == c1.id

    # 2. Stage a simulated second candidate v2
    c2 = models.ModelCandidate(
        training_run_id=run.id,
        dataset_version_id=run.dataset_version_id,
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="2",
        model_uri=c1.model_uri,
        artifact_uri=c1.artifact_uri,
        metrics_json=c1.metrics_json,
        status="GATE_PASSED",
    )
    db.add(c2)
    db.commit()

    d2 = dep_service.stage_candidate(c2.id)
    assert d2.status == "AWAITING_APPROVAL"

    # 3. Reject v2
    rejected_d2 = dep_service.reject_deployment(d2.id, actor="reviewer", reason="Failed traffic burst test")
    assert rejected_d2.status == "REJECTED"

    # 4. Production MUST remain pointing to c1
    db.refresh(prod_env)
    assert prod_env.active_candidate_id == c1.id
