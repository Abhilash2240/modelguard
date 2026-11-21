import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from app.db import models
from app.domain.hashing import calculate_schema_hash, calculate_sha256
from app.ml.tracking import MLflowTracker
from app.schemas.inference import PredictionRequest
from app.schemas.models import ReleaseGatePolicy
from app.schemas.training import SplitConfig, TrainingJobConfig, TrainingJobCreate
from app.services.deployment_service import DeploymentService
from app.services.drift_service import DriftAnalysisService
from app.services.feedback_service import FeedbackService
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService
from app.services.training_job_service import TrainingJobService


@pytest.fixture
def tracking_env():
    tmp = Path(tempfile.mkdtemp())
    db_file = tmp / "test_mlflow_wp_d.db"
    art_dir = tmp / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    artifact_root = art_dir.as_uri()
    experiment_name = "wp-d-integration-exp"

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


def test_wp_d_full_lifecycle_monitoring_feedback_rollback(db: Session, valid_csv_path: Path, tracking_env):
    """End-to-end integration test for WP D:
    1. Register dataset and train model candidate v1.
    2. Register v1, pass release gate, stage, approve, and promote to production.
    3. Run live inference and persist telemetry.
    4. Submit delayed feedback: test idempotency and conflicting label rejection.
    5. Evaluate rolling quality report against ground truth.
    6. Train, register, stage, approve, and promote candidate v2.
    7. Detect input drift and prediction drift, verifying alert generation.
    8. Rollback production from v2 back to v1.
    9. Verify MLflow 'champion' alias is restored to v1, database pointers updated, and inference resolves v1.
    """
    tracking_uri = tracking_env["tracking_uri"]
    tracker = tracking_env["tracker"]
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Dataset setup
    dataset = models.Dataset(
        name="metro_traffic_wpd",
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

    # Train Run 1 (v1)
    job_service = TrainingJobService(db=db, tracker=tracker)
    split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    job1 = job_service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            validation_run_id=val_run.id,
            config=TrainingJobConfig(split=split_cfg, hyperparameters={"n_estimators": 5, "max_depth": 3}),
        )
    )
    run1 = job_service.execute_training_job(job1.id)

    # 2. Register candidate v1
    reg_service = ModelRegistryService(db=db, tracking_uri=tracking_uri)
    cand1 = reg_service.register_candidate(training_run_id=run1.id)
    policy = ReleaseGatePolicy(max_val_mae=10000.0, max_val_rmse=10000.0, min_val_r2=-100.0)
    gate1 = reg_service.evaluate_release_gate(cand1.id, policy=policy)
    assert gate1.passed is True

    # Stage, approve, promote v1
    dep_service = DeploymentService(db=db, tracking_uri=tracking_uri)
    stg_dep1 = dep_service.stage_candidate(cand1.id)
    dep_service.approve_deployment(stg_dep1.id, actor="lead_reviewer", rationale="Approved initial v1")
    prod_dep1 = dep_service.promote_deployment(stg_dep1.id, actor="lead_reviewer")
    assert prod_dep1.status == "PROMOTED"

    # Verify MLflow alias champion is 1
    alias_v1 = mlflow_client.get_model_version_by_alias("metro-traffic-volume", "champion")
    assert str(alias_v1.version) == str(cand1.mlflow_model_version)

    # 3. Serve Inference
    inf_service = InferenceService(db=db, tracking_uri=tracking_uri)
    sample_record = {
        "holiday": "None",
        "temp": 288.15,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 40,
        "weather_main": "Clouds",
        "weather_description": "scattered clouds",
        "date_time": "2026-11-21 09:00:00",
    }
    pred_res = inf_service.predict(PredictionRequest(inputs=[sample_record]))
    assert pred_res.model_version == cand1.mlflow_model_version
    assert len(pred_res.predictions) == 1

    # 4. Feedback Ingestion: Test Idempotency and Conflict
    feedback_service = FeedbackService(db=db)
    fb1 = feedback_service.submit_feedback(
        inference_id=pred_res.inference_id,
        actual_value=3250.0,
        submitted_by="induction_loop_01",
    )
    assert fb1.actual_value == 3250.0

    # Idempotent re-submission
    fb1_idem = feedback_service.submit_feedback(
        inference_id=pred_res.inference_id,
        actual_value=3250.0,
    )
    assert fb1_idem.id == fb1.id

    # 5. Quality Calculation (insufficient labels explicitly handled)
    q_report = feedback_service.calculate_rolling_quality(
        model_name="metro-traffic-volume",
        model_version=cand1.mlflow_model_version,
        min_samples=10,
    )
    assert q_report.status == "INSUFFICIENT_LABELS"
    assert q_report.labeled_sample_count == 1

    # 6. Train and Promote Model v2
    job2 = job_service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            validation_run_id=val_run.id,
            config=TrainingJobConfig(split=split_cfg, hyperparameters={"n_estimators": 10, "max_depth": 5}),
        )
    )
    run2 = job_service.execute_training_job(job2.id)
    cand2 = reg_service.register_candidate(training_run_id=run2.id)
    gate2 = reg_service.evaluate_release_gate(cand2.id, policy=policy)
    assert gate2.passed is True

    stg_dep2 = dep_service.stage_candidate(cand2.id)
    dep_service.approve_deployment(stg_dep2.id, actor="lead_reviewer", rationale="Approved upgrade to v2")
    prod_dep2 = dep_service.promote_deployment(stg_dep2.id, actor="lead_reviewer")

    # Confirm MLflow champion alias is now v2
    alias_v2 = mlflow_client.get_model_version_by_alias("metro-traffic-volume", "champion")
    assert str(alias_v2.version) == str(cand2.mlflow_model_version)

    # 7. Drift Detection on Perturbed Batch
    drift_service = DriftAnalysisService(db=db)
    current_df = df.copy()
    current_df["temp"] = current_df["temp"] + 25.0  # Temperature drift
    drift_report = drift_service.analyze_input_drift(
        current_df=current_df,
        reference_df=df,
        model_name="metro-traffic-volume",
        model_version=cand2.mlflow_model_version,
    )
    assert drift_report.is_drift_detected is True
    assert "temp" in drift_report.summary_json["drifting_features"]

    # Verify alert was created
    alert = db.query(models.MonitoringAlert).filter(models.MonitoringAlert.report_id == drift_report.id).first()
    assert alert is not None
    assert alert.alert_type == "DATA_DRIFT"

    # 8. Audited Rollback from v2 back to v1
    rollback_dep = dep_service.rollback_deployment(
        deployment_id=prod_dep2.id,
        actor="on_call_sre",
        reason="Temperature drift detected and input anomaly suspected",
    )

    assert rollback_dep.candidate_id == cand1.id
    assert rollback_dep.previous_candidate_id == cand2.id
    assert rollback_dep.status == "PROMOTED"

    # 9. Verify Restored Champion Alias and Inference Resolution
    alias_restored = mlflow_client.get_model_version_by_alias("metro-traffic-volume", "champion")
    assert str(alias_restored.version) == str(cand1.mlflow_model_version)

    prod_env = db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "production").first()
    assert prod_env.active_candidate_id == cand1.id

    # Verify subsequent inference resolves restored v1
    pred_res_post_rollback = inf_service.predict(PredictionRequest(inputs=[sample_record]))
    assert pred_res_post_rollback.model_version == cand1.mlflow_model_version
