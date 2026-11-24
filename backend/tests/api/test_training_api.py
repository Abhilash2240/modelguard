import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db import models


@pytest.fixture
def seed_training_data(db: Session):
    dataset = models.Dataset(
        name="training_api_test_ds",
        task_type="regression",
        source_uri="http://example.com/data",
        licence="MIT",
    )
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.0",
        storage_uri="storage/test.csv",
        sha256_hash="a" * 64,
        byte_size=500,
        row_count=20,
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
    db.commit()

    job1 = models.TrainingJob(
        dataset_version_id=version.id,
        validation_run_id=val_run.id,
        config_json={"model": "random_forest", "n_estimators": 10},
        status="SUCCEEDED",
    )
    job2 = models.TrainingJob(
        dataset_version_id=version.id,
        validation_run_id=val_run.id,
        config_json={"model": "random_forest", "n_estimators": 20},
        status="FAILED",
    )
    db.add_all([job1, job2])
    db.commit()

    run1 = models.TrainingRun(
        training_job_id=job1.id,
        dataset_version_id=version.id,
        mlflow_run_id="mlflow-run-abc",
        python_version="3.12.0",
        random_seed=42,
        status="SUCCEEDED",
        metrics_json={"candidate_val_mae": 320.5, "baseline_val_mae": 850.0},
        artifact_uri="file:///mlflow/artifacts/run-abc",
    )
    db.add(run1)
    db.commit()

    return {
        "dataset": dataset,
        "version": version,
        "job1": job1,
        "job2": job2,
        "run1": run1,
    }


def test_list_training_jobs_empty(client: TestClient):
    response = client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_training_jobs_with_runs(client: TestClient, seed_training_data):
    response = client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # Verify job 1 has nested training_run populated
    job1_data = next(j for j in data if j["id"] == seed_training_data["job1"].id)
    assert job1_data["status"] == "SUCCEEDED"
    assert job1_data["training_run"] is not None
    assert job1_data["training_run"]["mlflow_run_id"] == "mlflow-run-abc"
    assert job1_data["training_run"]["metrics_json"]["candidate_val_mae"] == 320.5

    # Verify job 2 has no training_run
    job2_data = next(j for j in data if j["id"] == seed_training_data["job2"].id)
    assert job2_data["status"] == "FAILED"
    assert job2_data["training_run"] is None


def test_list_training_jobs_filtered_by_status(client: TestClient, seed_training_data):
    response = client.get("/api/v1/training/jobs?status=SUCCEEDED")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == seed_training_data["job1"].id


def test_get_training_job_success(client: TestClient, seed_training_data):
    job_id = seed_training_data["job1"].id
    response = client.get(f"/api/v1/training/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["training_run"]["id"] == seed_training_data["run1"].id


def test_get_training_job_not_found(client: TestClient):
    response = client.get("/api/v1/training/jobs/nonexistent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_list_training_runs(client: TestClient, seed_training_data):
    response = client.get("/api/v1/training/runs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == seed_training_data["run1"].id
    assert data[0]["random_seed"] == 42


def test_get_training_run_success(client: TestClient, seed_training_data):
    run_id = seed_training_data["run1"].id
    response = client.get(f"/api/v1/training/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == run_id
    assert data["mlflow_run_id"] == "mlflow-run-abc"


def test_get_training_run_not_found(client: TestClient):
    response = client.get("/api/v1/training/runs/nonexistent-id")
    assert response.status_code == 404
