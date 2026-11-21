from unittest import mock

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.core.errors import DeploymentError, InvalidOperationError
from app.db import models
from app.services.deployment_service import DeploymentService


@pytest.fixture
def candidate_setup(db: Session):
    dataset = models.Dataset(name="test_d", task_type="regression", source_uri="http://x", licence="MIT")
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.1",
        storage_uri="storage/t.csv",
        sha256_hash="c" * 64,
        byte_size=100,
        row_count=10,
        columns_json=[],
        schema_hash="d" * 64,
    )
    db.add(version)
    db.commit()

    val_run = models.ValidationRun(
        dataset_version_id=version.id, status="PASSED", error_count=0, warning_count=0, details_json={}
    )
    db.add(val_run)

    job = models.TrainingJob(
        dataset_version_id=version.id, validation_run_id=val_run.id, config_json={}, status="SUCCEEDED"
    )
    db.add(job)
    db.commit()

    train_run = models.TrainingRun(
        training_job_id=job.id,
        dataset_version_id=version.id,
        mlflow_run_id="run-123",
        python_version="3.13.7",
        random_seed=42,
        status="SUCCEEDED",
        metrics_json={
            "candidate_val_mae": 250.0,
            "candidate_val_rmse": 400.0,
            "candidate_val_r2": 0.92,
            "baseline_val_mae": 1700.0,
        },
    )
    db.add(train_run)
    db.commit()

    candidate = models.ModelCandidate(
        training_run_id=train_run.id,
        dataset_version_id=version.id,
        mlflow_model_name="traffic-reg",
        mlflow_model_version="1",
        model_uri="models:/traffic-reg/1",
        artifact_uri="runs:/run-123/candidate_model",
        metrics_json=train_run.metrics_json,
        status="GATE_PASSED",
    )
    db.add(candidate)
    db.commit()

    return {"candidate": candidate, "run": train_run, "version": version}


def test_staging_deployment_and_smoke_check_pass(db: Session, candidate_setup):
    candidate = candidate_setup["candidate"]
    service = DeploymentService(db=db, tracking_uri="sqlite:///:memory:")

    mock_estimator = mock.MagicMock()
    mock_estimator.predict.return_value = np.array([2450.5])

    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        deployment = service.stage_candidate(candidate.id)

    assert deployment.id is not None
    assert deployment.candidate_id == candidate.id
    assert deployment.status == "AWAITING_APPROVAL"
    assert deployment.smoke_test_results_json["passed"] is True
    assert deployment.smoke_test_results_json["sample_prediction"] == 2450.5

    # Candidate status updated
    db.refresh(candidate)
    assert candidate.status == "AWAITING_APPROVAL"

    # Staging environment pointer updated
    staging_env = db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "staging").first()
    assert staging_env is not None
    assert staging_env.active_candidate_id == candidate.id


def test_staging_rejected_if_gate_failed(db: Session, candidate_setup):
    candidate = candidate_setup["candidate"]
    candidate.status = "GATE_FAILED"
    db.commit()

    service = DeploymentService(db=db, tracking_uri="sqlite:///:memory:")
    with pytest.raises(InvalidOperationError) as exc_info:
        service.stage_candidate(candidate.id)
    assert "Expected 'GATE_PASSED'" in str(exc_info.value)


def test_staging_fails_if_artifact_unreachable(db: Session, candidate_setup):
    candidate = candidate_setup["candidate"]
    service = DeploymentService(db=db, tracking_uri="sqlite:///:memory:")

    with mock.patch("mlflow.sklearn.load_model", side_effect=Exception("Artifact corrupt")):
        with pytest.raises(DeploymentError) as exc_info:
            service.stage_candidate(candidate.id)
        assert "Staging smoke checks failed" in str(exc_info.value)

    db.refresh(candidate)
    assert candidate.status == "FAILED"


def test_explicit_approval_workflow(db: Session, candidate_setup):
    candidate = candidate_setup["candidate"]
    service = DeploymentService(db=db, tracking_uri="sqlite:///:memory:")

    mock_estimator = mock.MagicMock(predict=lambda x: np.array([1200.0]))
    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        deployment = service.stage_candidate(candidate.id)

    approved = service.approve_deployment(
        deployment_id=deployment.id,
        actor="lead-mlops",
        rationale="Validation and test metrics satisfy release criteria",
    )

    assert approved.status == "APPROVED"
    assert approved.approved_by == "lead-mlops"
    assert approved.approval_rationale == "Validation and test metrics satisfy release criteria"
    assert approved.approved_at is not None

    db.refresh(candidate)
    assert candidate.status == "APPROVED"


def test_explicit_rejection_workflow(db: Session, candidate_setup):
    candidate = candidate_setup["candidate"]
    service = DeploymentService(db=db, tracking_uri="sqlite:///:memory:")

    # Seed an existing production champion
    prod_champ = models.ModelCandidate(
        training_run_id=candidate.training_run_id,
        dataset_version_id=candidate.dataset_version_id,
        mlflow_model_name="traffic-reg",
        mlflow_model_version="0",
        model_uri="models:/traffic-reg/0",
        artifact_uri="runs:/fake/candidate_model",
        status="PROMOTED",
    )
    db.add(prod_champ)
    db.flush()
    prod_env = models.DeploymentEnvironment(name="production", active_candidate_id=prod_champ.id)
    db.add(prod_env)
    db.commit()

    mock_estimator = mock.MagicMock(predict=lambda x: np.array([1200.0]))
    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        deployment = service.stage_candidate(candidate.id)

    rejected = service.reject_deployment(
        deployment_id=deployment.id,
        actor="lead-mlops",
        reason="Performance regressed during peak rush hour window",
    )

    assert rejected.status == "REJECTED"
    assert rejected.approval_rationale == "Performance regressed during peak rush hour window"

    db.refresh(candidate)
    assert candidate.status == "REJECTED"

    # Production environment pointer MUST remain unchanged
    db.refresh(prod_env)
    assert prod_env.active_candidate_id == prod_champ.id


def test_promotion_workflow_exact_version_and_mlflow_alias(db: Session, candidate_setup):
    candidate = candidate_setup["candidate"]
    service = DeploymentService(db=db, tracking_uri="sqlite:///:memory:")

    mock_estimator = mock.MagicMock(predict=lambda x: np.array([1200.0]))
    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        staging_dep = service.stage_candidate(candidate.id)

    service.approve_deployment(staging_dep.id, actor="lead-mlops", rationale="Ready for prod")

    with mock.patch("app.services.deployment_service.MlflowClient") as mock_client_cls:
        client_inst = mock_client_cls.return_value

        prod_dep = service.promote_deployment(staging_dep.id, actor="lead-mlops")

        # Verify exact champion alias update
        client_inst.set_registered_model_alias.assert_called_once_with(
            name="traffic-reg",
            alias="champion",
            version="1",
        )

        assert prod_dep.status == "PROMOTED"
        assert prod_dep.candidate_id == candidate.id

        prod_env = (
            db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "production").first()
        )
        assert prod_env is not None
        assert prod_env.active_candidate_id == candidate.id

        db.refresh(candidate)
        assert candidate.status == "PROMOTED"


def test_promotion_rejected_without_prior_approval(db: Session, candidate_setup):
    candidate = candidate_setup["candidate"]
    service = DeploymentService(db=db, tracking_uri="sqlite:///:memory:")

    mock_estimator = mock.MagicMock(predict=lambda x: np.array([1200.0]))
    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        staging_dep = service.stage_candidate(candidate.id)

    # In AWAITING_APPROVAL state (unapproved)
    with pytest.raises(InvalidOperationError) as exc_info:
        service.promote_deployment(staging_dep.id, actor="lead-mlops")
    assert "must be APPROVED before promotion" in str(exc_info.value)
