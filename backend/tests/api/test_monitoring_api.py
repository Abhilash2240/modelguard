from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db import models


@pytest.fixture
def seeded_db(db: Session):
    # Ensure default environments exist
    staging = db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "staging").first()
    if not staging:
        db.add(models.DeploymentEnvironment(name="staging"))
    prod = db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "production").first()
    if not prod:
        db.add(models.DeploymentEnvironment(name="production"))
    db.commit()
    return db


def test_metrics_endpoint(client: TestClient):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "modelguard_http_requests_total" in resp.text


def test_feedback_api_lifecycle(client: TestClient, seeded_db: Session):
    # Create inference event
    ev = models.PredictionEvent(
        inference_id="api-inf-1",
        model_name="metro-traffic-volume",
        model_version="1",
        latency_ms=10.0,
        status="SUCCESS",
        prediction_json={"predictions": [2800.0], "count": 1},
    )
    seeded_db.add(ev)
    seeded_db.commit()

    # Submit feedback
    resp = client.post(
        "/api/v1/feedback",
        json={"inference_id": "api-inf-1", "actual_value": 2850.0, "notes": "Road sensor"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["inference_id"] == "api-inf-1"
    assert data["actual_value"] == 2850.0

    # Idempotent re-submission
    resp_idem = client.post(
        "/api/v1/feedback",
        json={"inference_id": "api-inf-1", "actual_value": 2850.0},
    )
    assert resp_idem.status_code == 200
    assert resp_idem.json()["id"] == data["id"]

    # Conflicting re-submission
    resp_conflict = client.post(
        "/api/v1/feedback",
        json={"inference_id": "api-inf-1", "actual_value": 3500.0},
    )
    assert resp_conflict.status_code == 409


def test_quality_api(client: TestClient, seeded_db: Session):
    resp = client.get("/api/v1/quality?model_name=metro-traffic-volume&min_samples=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("INSUFFICIENT_LABELS", "SUFFICIENT_LABELS")


def test_alerts_api_lifecycle(client: TestClient, seeded_db: Session):
    # Insert test alert
    alert = models.MonitoringAlert(
        model_name="metro-traffic-volume",
        model_version="1",
        alert_type="DATA_DRIFT",
        severity="WARNING",
        message="Test alert",
        triggering_evidence_json={"feature": "temp"},
        status="OPEN",
    )
    seeded_db.add(alert)
    seeded_db.commit()
    seeded_db.refresh(alert)

    # List alerts
    list_resp = client.get("/api/v1/alerts")
    assert list_resp.status_code == 200
    alerts = list_resp.json()
    assert len(alerts) >= 1

    # Acknowledge alert
    ack_resp = client.post(f"/api/v1/alerts/{alert.id}/acknowledge", json={"actor": "ops_user"})
    assert ack_resp.status_code == 200
    assert ack_resp.json()["status"] == "ACKNOWLEDGED"

    # Resolve alert
    res_resp = client.post(
        f"/api/v1/alerts/{alert.id}/resolve",
        json={"actor": "ops_user", "notes": "Sensor recalibrated"},
    )
    assert res_resp.status_code == 200
    assert res_resp.json()["status"] == "RESOLVED"
    assert res_resp.json()["resolution_notes"] == "Sensor recalibrated"


def test_rollback_api_endpoint(client: TestClient, seeded_db: Session):
    prod_env = (
        seeded_db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "production").first()
    )

    c1 = models.ModelCandidate(
        id="c1",
        training_run_id="tr1",
        dataset_version_id="ds1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="models:/metro-traffic-volume/1",
        artifact_uri="uri1",
        status="ARCHIVED",
    )
    c2 = models.ModelCandidate(
        id="c2",
        training_run_id="tr2",
        dataset_version_id="ds1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="2",
        model_uri="models:/metro-traffic-volume/2",
        artifact_uri="uri2",
        status="PROMOTED",
    )
    seeded_db.add_all([c1, c2])
    seeded_db.flush()

    prod_env.active_candidate_id = c2.id

    dep2 = models.Deployment(
        id="dep-api-2",
        environment_id=prod_env.id,
        candidate_id=c2.id,
        previous_candidate_id=c1.id,
        status="PROMOTED",
    )
    seeded_db.add(dep2)
    seeded_db.commit()

    mock_client = mock.MagicMock()
    mock_model = mock.MagicMock()
    mock_model.predict.return_value = [3200.0]

    with (
        mock.patch("app.services.deployment_service.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.sklearn.load_model", return_value=mock_model),
    ):
        resp = client.post(
            f"/api/v1/deployments/{dep2.id}/rollback",
            json={"actor": "lead_engineer", "reason": "Severe prediction error spike"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["candidate_id"] == c1.id
        assert data["previous_candidate_id"] == c2.id
        assert "ROLLBACK" in data["approval_rationale"]
