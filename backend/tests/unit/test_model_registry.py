from unittest import mock

import pytest
from sqlalchemy.orm import Session

from app.core.errors import InvalidOperationError
from app.db import models
from app.services.model_registry_service import ModelRegistryService


@pytest.fixture
def run_and_dataset(db: Session):
    dataset = models.Dataset(
        name="test_metro",
        task_type="regression",
        source_uri="http://example.com",
        licence="CC BY 4.0",
    )
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.1",
        storage_uri="storage/test.csv",
        sha256_hash="a" * 64,
        byte_size=1000,
        row_count=100,
        columns_json=[],
        schema_hash="b" * 64,
    )
    db.add(version)
    db.commit()

    val_run = models.ValidationRun(
        dataset_version_id=version.id,
        status="PASSED",
        error_count=0,
        warning_count=0,
        details_json={},
    )
    db.add(val_run)

    job = models.TrainingJob(
        dataset_version_id=version.id,
        validation_run_id=val_run.id,
        config_json={},
        status="SUCCEEDED",
    )
    db.add(job)
    db.commit()

    train_run = models.TrainingRun(
        training_job_id=job.id,
        dataset_version_id=version.id,
        mlflow_run_id="mlflow-run-abc",
        python_version="3.13.7",
        random_seed=42,
        status="SUCCEEDED",
        metrics_json={
            "candidate_val_mae": 285.5,
            "candidate_val_rmse": 450.0,
            "candidate_val_r2": 0.91,
            "baseline_val_mae": 1720.0,
        },
        artifact_uri="runs:/mlflow-run-abc/candidate_model",
    )
    db.add(train_run)
    db.commit()

    return {"dataset": dataset, "version": version, "val_run": val_run, "job": job, "run": train_run}


def test_candidate_registration_from_successful_run(db: Session, run_and_dataset):
    train_run = run_and_dataset["run"]
    service = ModelRegistryService(db=db, tracking_uri="sqlite:///:memory:")

    mock_version = mock.MagicMock(version="1")
    with mock.patch("app.services.model_registry_service.MlflowClient") as mock_client_cls:
        client_inst = mock_client_cls.return_value
        client_inst.create_model_version.return_value = mock_version

        candidate = service.register_candidate(
            training_run_id=train_run.id,
            model_name="test-traffic-model",
        )

        assert candidate.id is not None
        assert candidate.training_run_id == train_run.id
        assert candidate.dataset_version_id == train_run.dataset_version_id
        assert candidate.mlflow_model_name == "test-traffic-model"
        assert candidate.mlflow_model_version == "1"
        assert candidate.model_uri == "models:/test-traffic-model/1"
        assert candidate.status == "REGISTERED"

        # Verify audit event persisted
        audit = (
            db.query(models.AuditEvent)
            .filter(
                models.AuditEvent.resource_id == candidate.id, models.AuditEvent.action == "REGISTER_MODEL_CANDIDATE"
            )
            .first()
        )
        assert audit is not None


def test_candidate_registration_from_failed_run_rejected(db: Session, run_and_dataset):
    train_run = run_and_dataset["run"]
    train_run.status = "FAILED"
    db.commit()

    service = ModelRegistryService(db=db, tracking_uri="sqlite:///:memory:")
    with pytest.raises(InvalidOperationError) as exc_info:
        service.register_candidate(training_run_id=train_run.id)
    assert "Expected 'SUCCEEDED'" in str(exc_info.value)


def test_release_gate_valid_candidate_passes(db: Session, run_and_dataset):
    train_run = run_and_dataset["run"]
    candidate = models.ModelCandidate(
        training_run_id=train_run.id,
        dataset_version_id=train_run.dataset_version_id,
        mlflow_model_name="test-traffic-model",
        mlflow_model_version="1",
        model_uri="models:/test-traffic-model/1",
        artifact_uri="runs:/fake/candidate_model",
        metrics_json=train_run.metrics_json,
        status="REGISTERED",
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db=db, tracking_uri="sqlite:///:memory:")
    with mock.patch("mlflow.sklearn.load_model", return_value=mock.MagicMock(predict=lambda x: [1.0])):
        result = service.evaluate_release_gate(candidate.id)

    assert result.passed is True
    assert len(result.rejection_reasons) == 0
    assert candidate.status == "GATE_PASSED"
    assert candidate.gate_evaluation_json["passed"] is True


def test_release_gate_poor_metric_fails(db: Session, run_and_dataset):
    train_run = run_and_dataset["run"]
    # MAE 550 exceeds default 400.0 limit
    poor_metrics = {
        "candidate_val_mae": 550.0,
        "candidate_val_rmse": 750.0,
        "candidate_val_r2": 0.60,
        "baseline_val_mae": 1720.0,
    }
    candidate = models.ModelCandidate(
        training_run_id=train_run.id,
        dataset_version_id=train_run.dataset_version_id,
        mlflow_model_name="test-traffic-model",
        mlflow_model_version="1",
        model_uri="models:/test-traffic-model/1",
        artifact_uri="runs:/fake/candidate_model",
        metrics_json=poor_metrics,
        status="REGISTERED",
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db=db, tracking_uri="sqlite:///:memory:")
    with mock.patch("mlflow.sklearn.load_model", return_value=mock.MagicMock(predict=lambda x: [1.0])):
        result = service.evaluate_release_gate(candidate.id)

    assert result.passed is False
    assert any("exceeds policy limit" in r for r in result.rejection_reasons)
    assert candidate.status == "GATE_FAILED"


def test_release_gate_missing_validation_fails_closed(db: Session, run_and_dataset):
    # Set validation run to FAILED
    val_run = run_and_dataset["val_run"]
    val_run.status = "FAILED"
    db.commit()

    train_run = run_and_dataset["run"]
    candidate = models.ModelCandidate(
        training_run_id=train_run.id,
        dataset_version_id=train_run.dataset_version_id,
        mlflow_model_name="test-traffic-model",
        mlflow_model_version="1",
        model_uri="models:/test-traffic-model/1",
        artifact_uri="runs:/fake/candidate_model",
        metrics_json=train_run.metrics_json,
        status="REGISTERED",
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db=db, tracking_uri="sqlite:///:memory:")
    with mock.patch("mlflow.sklearn.load_model", return_value=mock.MagicMock(predict=lambda x: [1.0])):
        result = service.evaluate_release_gate(candidate.id)

    assert result.passed is False
    assert any("no passing ValidationRun" in r for r in result.rejection_reasons)
    assert candidate.status == "GATE_FAILED"


def test_release_gate_fails_against_champion(db: Session, run_and_dataset):
    train_run = run_and_dataset["run"]

    # Seed an existing champion with excellent MAE 200.0
    champion = models.ModelCandidate(
        training_run_id=train_run.id,
        dataset_version_id=train_run.dataset_version_id,
        mlflow_model_name="test-traffic-model",
        mlflow_model_version="0",
        model_uri="models:/test-traffic-model/0",
        artifact_uri="runs:/champ/candidate_model",
        metrics_json={"candidate_val_mae": 200.0, "candidate_val_rmse": 300.0, "candidate_val_r2": 0.95},
        status="PROMOTED",
    )
    db.add(champion)
    db.flush()

    prod_env = models.DeploymentEnvironment(name="production", active_candidate_id=champion.id)
    db.add(prod_env)
    db.commit()

    # Candidate has MAE 285.5, which is > 5% worse than 200.0
    candidate = models.ModelCandidate(
        training_run_id=train_run.id,
        dataset_version_id=train_run.dataset_version_id,
        mlflow_model_name="test-traffic-model",
        mlflow_model_version="1",
        model_uri="models:/test-traffic-model/1",
        artifact_uri="runs:/fake/candidate_model",
        metrics_json=train_run.metrics_json,
        status="REGISTERED",
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db=db, tracking_uri="sqlite:///:memory:")
    with mock.patch("mlflow.sklearn.load_model", return_value=mock.MagicMock(predict=lambda x: [1.0])):
        result = service.evaluate_release_gate(candidate.id)

    assert result.passed is False
    assert any("regresses by more than" in r for r in result.rejection_reasons)
    assert candidate.status == "GATE_FAILED"
