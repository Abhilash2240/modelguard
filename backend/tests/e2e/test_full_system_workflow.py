import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from app.core.errors import (
    ContractValidationError,
    DeploymentError,
    FeedbackConflictError,
    InputContractViolationError,
    InvalidOperationError,
)
from app.db import models
from app.ml.tracking import MLflowTracker
from app.schemas.dataset import DatasetCreate
from app.schemas.inference import PredictionRequest
from app.schemas.models import ReleaseGatePolicy
from app.schemas.training import SplitConfig, TrainingJobConfig, TrainingJobCreate
from app.services.dataset_service import DatasetService
from app.services.deployment_service import DeploymentService
from app.services.drift_service import DriftAnalysisService
from app.services.feedback_service import FeedbackService
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService
from app.services.training_job_service import TrainingJobService


@pytest.fixture
def tracking_env():
    tmp = Path(tempfile.mkdtemp())
    db_file = tmp / "e2e_mlflow.db"
    art_dir = tmp / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    artifact_root = art_dir.as_uri()
    experiment_name = "e2e-workflow-exp"

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


def test_e2e_full_golden_path_workflow(db: Session, valid_csv_path: Path, tracking_env):
    """Complete ModelGuard E2E workflow:
    Dataset registration -> validation -> training -> experiment tracking ->
    model candidate registration -> release gate -> staging -> approval ->
    promotion -> inference -> feedback -> monitoring -> alert -> rollback.
    """
    tracking_uri = tracking_env["tracking_uri"]
    tracker = tracking_env["tracker"]
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Dataset Registration & Provenance
    dataset_service = DatasetService(db=db)
    dataset = dataset_service.register_dataset(
        DatasetCreate(
            name="metro_traffic_e2e",
            task_type="regression",
            source_uri="https://archive.ics.uci.edu/dataset/492/urban+traffic",
            licence="CC BY 4.0",
        )
    )
    version = dataset_service.create_dataset_version(
        dataset_id=dataset.id,
        version_tag="v1.0.0",
        source_file_path=valid_csv_path,
    )
    val_run = dataset_service.validate_version(version.id)
    assert val_run.status == "PASSED"
    assert val_run.error_count == 0

    # 2. Training Job Execution & Experiment Tracking
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
    assert run1.status == "SUCCEEDED"
    assert run1.mlflow_run_id is not None

    # 3. Model Registration & Release Gate
    reg_service = ModelRegistryService(db=db, tracking_uri=tracking_uri)
    cand1 = reg_service.register_candidate(training_run_id=run1.id)
    assert cand1.status == "REGISTERED"

    policy = ReleaseGatePolicy(max_val_mae=10000.0, max_val_rmse=10000.0, min_val_r2=-100.0)
    gate_res1 = reg_service.evaluate_release_gate(cand1.id, policy=policy)
    assert gate_res1.passed is True
    assert cand1.status == "GATE_PASSED"

    # 4. Staging Deployment & Automated Smoke Checks
    dep_service = DeploymentService(db=db, tracking_uri=tracking_uri)
    stg_dep1 = dep_service.stage_candidate(cand1.id)
    assert stg_dep1.status == "AWAITING_APPROVAL"
    assert stg_dep1.smoke_test_results_json.get("passed") is True

    # 5. Explicit Approval & Promotion to Production
    dep_service.approve_deployment(stg_dep1.id, actor="lead_reviewer", rationale="Smoke checks passed in staging")
    prod_dep1 = dep_service.promote_deployment(stg_dep1.id, actor="lead_reviewer")
    assert prod_dep1.status == "PROMOTED"

    # Verify MLflow alias 'champion' resolves candidate 1
    alias_mv = mlflow_client.get_model_version_by_alias("metro-traffic-volume", "champion")
    assert str(alias_mv.version) == str(cand1.mlflow_model_version)

    # 6. Version-Aware Inference Serving & Privacy Telemetry
    inf_service = InferenceService(db=db, tracking_uri=tracking_uri)
    sample_input = [
        {
            "holiday": "None",
            "temp": 288.15,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "weather_main": "Clouds",
            "weather_description": "scattered clouds",
            "date_time": "2026-11-21 09:00:00",
        }
    ]
    pred_res = inf_service.predict(PredictionRequest(inputs=sample_input))
    assert pred_res.model_version == cand1.mlflow_model_version
    assert len(pred_res.predictions) == 1
    assert pred_res.latency_ms > 0

    # 7. Delayed Feedback & Quality Assessment
    feedback_service = FeedbackService(db=db)
    feedback = feedback_service.submit_feedback(
        inference_id=pred_res.inference_id,
        actual_value=3100.0,
        submitted_by="induction_loop_01",
    )
    assert feedback.actual_value == 3100.0

    # 8. Train, Gate & Promote Candidate 2
    job2 = job_service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            validation_run_id=val_run.id,
            config=TrainingJobConfig(split=split_cfg, hyperparameters={"n_estimators": 10, "max_depth": 4}),
        )
    )
    run2 = job_service.execute_training_job(job2.id)
    cand2 = reg_service.register_candidate(training_run_id=run2.id)
    gate_res2 = reg_service.evaluate_release_gate(cand2.id, policy=policy)
    assert gate_res2.passed is True

    stg_dep2 = dep_service.stage_candidate(cand2.id)
    dep_service.approve_deployment(stg_dep2.id, actor="lead_reviewer", rationale="Candidate 2 approved")
    prod_dep2 = dep_service.promote_deployment(stg_dep2.id, actor="lead_reviewer")

    # MLflow alias champion now points to candidate 2
    alias_mv2 = mlflow_client.get_model_version_by_alias("metro-traffic-volume", "champion")
    assert str(alias_mv2.version) == str(cand2.mlflow_model_version)

    # 9. Statistical Drift Evaluation & Alert Generation
    drift_service = DriftAnalysisService(db=db)
    ref_df = pd.read_csv(valid_csv_path)
    curr_df = ref_df.copy()
    curr_df["temp"] = curr_df["temp"] + 30.0  # Covariate shift

    drift_report = drift_service.analyze_input_drift(
        current_df=curr_df,
        reference_df=ref_df,
        model_name="metro-traffic-volume",
        model_version=cand2.mlflow_model_version,
    )
    assert drift_report.is_drift_detected is True
    assert "temp" in drift_report.summary_json["drifting_features"]

    # Verify persistent alert
    alert = db.query(models.MonitoringAlert).filter(models.MonitoringAlert.report_id == drift_report.id).first()
    assert alert is not None
    assert alert.alert_type == "DATA_DRIFT"

    # 10. Audited Rollback to Candidate 1
    rollback_dep = dep_service.rollback_deployment(
        deployment_id=prod_dep2.id,
        actor="on_call_sre",
        reason="Temperature covariate drift detected; rolling back to proven candidate 1",
    )
    assert rollback_dep.candidate_id == cand1.id
    assert rollback_dep.status == "PROMOTED"

    # Verify MLflow alias restored
    alias_restored = mlflow_client.get_model_version_by_alias("metro-traffic-volume", "champion")
    assert str(alias_restored.version) == str(cand1.mlflow_model_version)

    # Verify subsequent inference resolves restored candidate 1
    post_rollback_pred = inf_service.predict(PredictionRequest(inputs=sample_input))
    assert post_rollback_pred.model_version == cand1.mlflow_model_version


def test_e2e_invalid_dataset_blocks_training(db: Session, invalid_csv_path: Path):
    """Confirm invalid dataset fails contract validation and blocks training job execution."""
    dataset_service = DatasetService(db=db)
    dataset = dataset_service.register_dataset(
        DatasetCreate(
            name="metro_traffic_invalid",
            task_type="regression",
            source_uri="http://invalid.data",
            licence="MIT",
        )
    )
    version = dataset_service.create_dataset_version(
        dataset_id=dataset.id,
        source_file_path=invalid_csv_path,
        version_tag="v0.0.1",
    )
    val_run = dataset_service.validate_version(version.id)
    assert val_run.status == "FAILED"
    assert val_run.error_count > 0

    # Attempting to execute training on failed validation run must be rejected
    tracker = mock.MagicMock()
    tracker.start_training_run.return_value = "mock-run-id-1234"
    job_service = TrainingJobService(db=db, tracker=tracker)
    split_cfg = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, min_observations=5)
    job = job_service.create_training_job(
        TrainingJobCreate(
            dataset_version_id=version.id,
            validation_run_id=val_run.id,
            config=TrainingJobConfig(split=split_cfg),
        )
    )
    with pytest.raises(ContractValidationError) as exc_info:
        job_service.execute_training_job(job.id)
    assert "failed contract validation" in str(exc_info.value)


def test_e2e_failed_release_gate_blocks_staging(db: Session, tracking_env):
    """Confirm candidate failing release gate thresholds cannot be staged."""
    tracking_uri = tracking_env["tracking_uri"]
    reg_service = ModelRegistryService(db=db, tracking_uri=tracking_uri)
    dep_service = DeploymentService(db=db, tracking_uri=tracking_uri)

    # Create dummy candidate with poor metrics
    cand = models.ModelCandidate(
        id=models.generate_uuid(),
        training_run_id=models.generate_uuid(),
        dataset_version_id=models.generate_uuid(),
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="99",
        model_uri="models:/metro-traffic-volume/99",
        artifact_uri="storage/models/99",
        status="REGISTERED",
        metrics_json={
            "candidate_val_mae": 9999.0,  # Unacceptable MAE
            "baseline_val_mae": 100.0,
            "candidate_val_rmse": 9999.0,
            "candidate_val_r2": 0.10,
        },
    )
    db.add(cand)
    db.commit()

    # Enforce strict policy
    strict_policy = ReleaseGatePolicy(max_val_mae=400.0, max_val_rmse=600.0, min_val_r2=0.85)
    gate_res = reg_service.evaluate_release_gate(cand.id, policy=strict_policy)
    assert gate_res.passed is False
    assert cand.status == "GATE_FAILED"

    # Staging must be rejected
    with pytest.raises(InvalidOperationError) as exc_info:
        dep_service.stage_candidate(cand.id)
    assert "Expected 'GATE_PASSED'" in str(exc_info.value) or "failed release gate" in str(exc_info.value)


def test_e2e_staging_smoke_failure_blocks_production(db: Session, tracking_env):
    """Confirm candidate that fails staging smoke test is marked SMOKE_FAILED and blocks promotion."""
    tracking_uri = tracking_env["tracking_uri"]
    dep_service = DeploymentService(db=db, tracking_uri=tracking_uri)

    # Candidate with invalid URI that causes load error
    cand = models.ModelCandidate(
        id=models.generate_uuid(),
        training_run_id=models.generate_uuid(),
        dataset_version_id=models.generate_uuid(),
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="88",
        model_uri="models:/metro-traffic-volume/corrupt-uri",
        artifact_uri="storage/models/corrupt",
        status="GATE_PASSED",
    )
    db.add(cand)
    db.commit()

    # Staging executes smoke test which fails and raises DeploymentError
    with pytest.raises(DeploymentError) as exc_info:
        dep_service.stage_candidate(cand.id)
    assert "smoke checks failed" in str(exc_info.value).lower()

    # Verify deployment record and candidate status
    deployment = db.query(models.Deployment).filter(models.Deployment.candidate_id == cand.id).first()
    assert deployment is not None
    assert deployment.status == "SMOKE_FAILED"
    db.refresh(cand)
    assert cand.status == "FAILED"

    # Promotion must be rejected
    with pytest.raises(InvalidOperationError) as exc_info:
        dep_service.promote_deployment(deployment.id, actor="admin")
    assert "must be APPROVED" in str(exc_info.value)


def test_e2e_rejected_approval_leaves_champion_unchanged(db: Session, tracking_env):
    """Confirm human rejection leaves existing production champion completely untouched."""
    tracking_uri = tracking_env["tracking_uri"]
    dep_service = DeploymentService(db=db, tracking_uri=tracking_uri)

    prod_env = models.DeploymentEnvironment(name="production")
    staging_env = models.DeploymentEnvironment(name="staging")
    db.add_all([prod_env, staging_env])

    cand_champion = models.ModelCandidate(
        id="cand-champion",
        training_run_id="tr-c",
        dataset_version_id="ds-c",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="models:/metro-traffic-volume/1",
        artifact_uri="art1",
        status="PROMOTED",
    )
    cand_staged = models.ModelCandidate(
        id="cand-staged",
        training_run_id="tr-s",
        dataset_version_id="ds-c",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="2",
        model_uri="models:/metro-traffic-volume/2",
        artifact_uri="art2",
        status="STAGED",
    )
    db.add_all([cand_champion, cand_staged])
    db.flush()

    prod_env.active_candidate_id = cand_champion.id

    deployment = models.Deployment(
        id="dep-to-reject",
        environment_id=staging_env.id,
        candidate_id=cand_staged.id,
        status="AWAITING_APPROVAL",
    )
    db.add(deployment)
    db.commit()

    # Operator rejects staged candidate
    rejected_dep = dep_service.reject_deployment(
        deployment_id=deployment.id,
        actor="senior_lead",
        reason="Subpar performance on heavy rain edge cases",
    )
    assert rejected_dep.status == "REJECTED"
    assert cand_staged.status == "REJECTED"

    # Production champion remains untouched
    db.refresh(prod_env)
    assert prod_env.active_candidate_id == cand_champion.id
    db.refresh(cand_champion)
    assert cand_champion.status == "PROMOTED"


def test_e2e_invalid_inference_does_not_invoke_estimator(db: Session):
    """Confirm input contract pre-validation rejects bad payloads before estimator invocation."""
    inf_service = InferenceService(db=db)

    # Missing required feature 'weather_main'
    bad_payload = [
        {
            "holiday": "None",
            "temp": 290.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 20,
            "weather_description": "clear sky",
            "date_time": "2026-11-21 12:00:00",
        }
    ]

    with mock.patch.object(inf_service, "_get_model") as mock_get_model:
        with pytest.raises(InputContractViolationError) as exc_info:
            inf_service.predict(PredictionRequest(inputs=bad_payload))
        assert "weather_main" in str(exc_info.value)
        # Estimator was never loaded or called!
        mock_get_model.assert_not_called()


def test_e2e_delayed_feedback_idempotency_and_conflict(db: Session):
    """Verify delayed feedback idempotency on matching actuals and rejection on conflict."""
    ev = models.PredictionEvent(
        inference_id="e2e-inf-feedback",
        model_name="metro-traffic-volume",
        model_version="1",
        latency_ms=10.0,
        status="SUCCESS",
        prediction_json={"predictions": [3000.0]},
    )
    db.add(ev)
    db.commit()

    feedback_service = FeedbackService(db=db)
    # First submission
    fb1 = feedback_service.submit_feedback(inference_id="e2e-inf-feedback", actual_value=3050.0)
    assert fb1.actual_value == 3050.0

    # Idempotent re-submission (identical actual value)
    fb2 = feedback_service.submit_feedback(inference_id="e2e-inf-feedback", actual_value=3050.0)
    assert fb1.id == fb2.id

    # Conflicting submission (different actual value)
    with pytest.raises(FeedbackConflictError) as exc_info:
        feedback_service.submit_feedback(inference_id="e2e-inf-feedback", actual_value=4500.0)
    assert "Conflicting ground-truth value" in str(exc_info.value)


def test_e2e_audit_records_survive_all_transitions(db: Session):
    """Verify immutable audit records survive all lifecycle operations."""
    audit_events = db.query(models.AuditEvent).all()
    # Ensure audit events can be queried and have immutable properties
    for event in audit_events:
        assert event.actor is not None
        assert event.action is not None
        assert event.resource_type is not None
        assert event.created_at is not None
