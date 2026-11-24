from unittest.mock import patch

from sqlalchemy.orm import Session

from app.db import models
from app.services.model_registry_service import ModelRegistryService


@patch("mlflow.sklearn.load_model")
def test_missing_candidate_metrics_fails_without_unbound_local_error(mock_load, db: Session):
    """Missing required candidate metrics fail closed cleanly without UnboundLocalError."""
    # Seed candidate with incomplete metrics
    candidate = models.ModelCandidate(
        id="cand-missing-metrics",
        training_run_id="run-incomplete",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="runs:/fake/model",
        artifact_uri="runs:/fake/artifact",
        status="REGISTERED",
        metrics_json={"candidate_val_mae": 250.0},  # missing RMSE and R2
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db)
    result = service.evaluate_release_gate(candidate.id)

    assert result.passed is False
    assert any(
        "missing candidate_val_mae, candidate_val_rmse, or candidate_val_r2" in r for r in result.rejection_reasons
    )


@patch("mlflow.sklearn.load_model")
def test_nan_mae_rejected_fail_closed(mock_load, db: Session):
    """NaN candidate MAE is strictly rejected by release gate."""
    candidate = models.ModelCandidate(
        id="cand-nan-metrics",
        training_run_id="run-nan",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="runs:/fake/model",
        artifact_uri="runs:/fake/artifact",
        status="REGISTERED",
        metrics_json={
            "candidate_val_mae": float("nan"),
            "candidate_val_rmse": 350.0,
            "candidate_val_r2": 0.85,
            "baseline_val_mae": 500.0,
        },
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db)
    result = service.evaluate_release_gate(candidate.id)

    assert result.passed is False
    assert any("invalid, NaN, or non-finite" in r for r in result.rejection_reasons)


@patch("mlflow.sklearn.load_model")
def test_infinite_rmse_rejected(mock_load, db: Session):
    """Infinite RMSE is rejected by release gate."""
    candidate = models.ModelCandidate(
        id="cand-inf-metrics",
        training_run_id="run-inf",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="runs:/fake/model",
        artifact_uri="runs:/fake/artifact",
        status="REGISTERED",
        metrics_json={
            "candidate_val_mae": 250.0,
            "candidate_val_rmse": float("inf"),
            "candidate_val_r2": 0.85,
            "baseline_val_mae": 500.0,
        },
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db)
    result = service.evaluate_release_gate(candidate.id)

    assert result.passed is False
    assert any("invalid, NaN, or non-finite" in r for r in result.rejection_reasons)


@patch("mlflow.sklearn.load_model")
def test_missing_baseline_mae_rejected(mock_load, db: Session):
    """Missing baseline MAE fails the gate because improvement cannot be verified."""
    candidate = models.ModelCandidate(
        id="cand-no-baseline",
        training_run_id="run-nobase",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="runs:/fake/model",
        artifact_uri="runs:/fake/artifact",
        status="REGISTERED",
        metrics_json={
            "candidate_val_mae": 250.0,
            "candidate_val_rmse": 350.0,
            "candidate_val_r2": 0.85,
        },
    )
    db.add(candidate)
    db.commit()

    service = ModelRegistryService(db)
    result = service.evaluate_release_gate(candidate.id)

    assert result.passed is False
    assert any("Baseline validation MAE missing" in r for r in result.rejection_reasons)
