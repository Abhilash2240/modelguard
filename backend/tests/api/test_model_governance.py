from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token, get_password_hash
from app.db import models
from app.main import app


@pytest.fixture
def auth_operator_token(db: Session):
    user = models.User(
        id="user-operator-gov-uuid",
        username="operator_gov",
        hashed_password=get_password_hash("Password123!"),
        role=models.UserRole.OPERATOR.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    token = create_access_token("operator_gov", "OPERATOR")
    return token


@pytest.fixture
def sample_payload():
    return {
        "inputs": [
            {
                "holiday": "None",
                "temp": 288.15,
                "rain_1h": 0.0,
                "snow_1h": 0.0,
                "clouds_all": 40,
                "weather_main": "Clouds",
                "weather_description": "scattered clouds",
                "date_time": "2026-11-24 09:00:00",
            }
        ]
    }


def test_production_inference_bypassing_champion_rejected(db: Session, auth_operator_token, sample_payload):
    """Specifying an unpromoted or different model version in production /predict is strictly rejected."""
    # Seed production environment with champion v1
    prod_env = models.DeploymentEnvironment(name="production")
    db.add(prod_env)

    champ_candidate = models.ModelCandidate(
        id="cand-champion-v1",
        training_run_id="run-1",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="runs:/fake/model",
        artifact_uri="runs:/fake/artifact",
        status="PROMOTED",
    )
    db.add(champ_candidate)
    db.flush()
    prod_env.active_candidate_id = champ_candidate.id

    # Seed another candidate v2 that is NOT promoted
    staged_candidate = models.ModelCandidate(
        id="cand-staged-v2",
        training_run_id="run-2",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="2",
        model_uri="runs:/fake/model2",
        artifact_uri="runs:/fake/artifact",
        status="STAGED",
    )
    db.add(staged_candidate)
    db.commit()

    from app.db.session import get_db

    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        # Caller tries to bypass champion by passing model_version="2" in production endpoint
        sample_payload["model_version"] = "2"
        res = client.post(
            "/api/v1/predict",
            json=sample_payload,
            headers={"Authorization": f"Bearer {auth_operator_token}"},
        )
        assert res.status_code == 400
        assert "not the active production champion" in res.json()["detail"]


@patch("mlflow.sklearn.load_model")
def test_diagnostic_inference_allowed_for_operator(mock_load, db: Session, auth_operator_token, sample_payload):
    """Operators can evaluate specific candidate versions via isolated diagnostic inference."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [3500.0]
    mock_load.return_value = mock_model

    # Seed production champion v1
    prod_env = models.DeploymentEnvironment(name="production")
    db.add(prod_env)
    champ = models.ModelCandidate(
        id="cand-champ-1",
        training_run_id="run-1",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="runs:/fake/model1",
        artifact_uri="runs:/fake/artifact",
        status="PROMOTED",
    )
    db.add(champ)
    db.flush()
    prod_env.active_candidate_id = champ.id

    # Seed diagnostic candidate v2
    diagnostic_cand = models.ModelCandidate(
        id="cand-diag-2",
        training_run_id="run-2",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="2",
        model_uri="runs:/fake/model2",
        artifact_uri="runs:/fake/artifact",
        status="STAGED",
    )
    db.add(diagnostic_cand)
    db.commit()

    from app.db.session import get_db

    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        sample_payload["model_version"] = "2"
        res = client.post(
            "/api/v1/inference/diagnostic",
            json=sample_payload,
            headers={"Authorization": f"Bearer {auth_operator_token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["model_version"] == "2"
        assert data["predictions"] == [3500.0]

        # Verify champion in production environment was NOT modified
        db.refresh(prod_env)
        assert prod_env.active_candidate_id == champ.id
