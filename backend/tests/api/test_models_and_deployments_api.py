from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db import models


@pytest.fixture
def seed_candidate(db: Session):
    dataset = models.Dataset(name="api_test_d", task_type="regression", source_uri="http://x", licence="MIT")
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.1",
        storage_uri="storage/t.csv",
        sha256_hash="1" * 64,
        byte_size=100,
        row_count=10,
        columns_json=[],
        schema_hash="2" * 64,
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
        mlflow_run_id="run-api-1",
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
        mlflow_model_name="traffic-api-model",
        mlflow_model_version="1",
        model_uri="models:/traffic-api-model/1",
        artifact_uri="runs:/run-api-1/candidate_model",
        metrics_json=train_run.metrics_json,
        status="GATE_PASSED",
    )
    db.add(candidate)
    db.commit()

    return {"candidate": candidate, "run": train_run}


def test_api_models_endpoints(client: TestClient, seed_candidate):
    candidate = seed_candidate["candidate"]

    # 1. List models
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    assert any(c["id"] == candidate.id for c in data)

    # 2. Get specific model
    res_get = client.get(f"/api/v1/models/{candidate.id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == candidate.id
    assert res_get.json()["mlflow_model_name"] == "traffic-api-model"

    # 3. 404 for missing model
    res_404 = client.get("/api/v1/models/non-existent-uuid")
    assert res_404.status_code == 404


def test_api_deployment_lifecycle(client: TestClient, seed_candidate):
    candidate = seed_candidate["candidate"]

    mock_estimator = mock.MagicMock(predict=lambda x: np.array([2500.0]))

    # 1. Stage candidate
    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        res_stage = client.post(f"/api/v1/models/{candidate.id}/stage")
    assert res_stage.status_code == 200
    stage_data = res_stage.json()
    deployment_id = stage_data["id"]
    assert stage_data["status"] == "AWAITING_APPROVAL"
    assert stage_data["smoke_test_results_json"]["passed"] is True

    # 2. Approve deployment
    res_app = client.post(
        f"/api/v1/deployments/{deployment_id}/approve",
        json={"actor": "lead-approver", "rationale": "Verified metrics on staging"},
    )
    assert res_app.status_code == 200
    assert res_app.json()["status"] == "APPROVED"
    assert res_app.json()["approved_by"] == "test_admin"

    # 3. Promote deployment
    with mock.patch("app.services.deployment_service.MlflowClient"):
        res_prom = client.post(
            f"/api/v1/deployments/{deployment_id}/promote",
            json={"actor": "release-mgr", "rationale": "Promoting to prod"},
        )
    assert res_prom.status_code == 200
    assert res_prom.json()["status"] == "PROMOTED"

    # 4. List deployments
    res_list = client.get("/api/v1/deployments")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1


def test_api_predict_endpoint(client: TestClient, seed_candidate, db: Session):
    candidate = seed_candidate["candidate"]
    candidate.status = "PROMOTED"
    db.commit()

    prod_env = models.DeploymentEnvironment(name="production", active_candidate_id=candidate.id)
    db.add(prod_env)
    db.commit()

    payload = {
        "inputs": [
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
    }

    mock_estimator = mock.MagicMock(predict=lambda x: np.array([3210.5]))
    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        res = client.post("/api/v1/predict", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["model_name"] == "traffic-api-model"
    assert body["model_version"] == "1"
    assert body["predictions"] == [3210.5]
    assert "inference_id" in body
    assert body["latency_ms"] > 0
