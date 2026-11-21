from unittest import mock

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.core.errors import EntityNotFoundError, InputContractViolationError
from app.db import models
from app.schemas.inference import PredictionRequest
from app.services.inference_service import InferenceService


@pytest.fixture
def prod_model_setup(db: Session):
    dataset = models.Dataset(name="traffic_inf", task_type="regression", source_uri="http://x", licence="MIT")
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.1",
        storage_uri="storage/t.csv",
        sha256_hash="e" * 64,
        byte_size=100,
        row_count=10,
        columns_json=[],
        schema_hash="f" * 64,
    )
    db.add(version)
    db.commit()

    job = models.TrainingJob(dataset_version_id=version.id, config_json={}, status="SUCCEEDED")
    db.add(job)
    db.commit()

    train_run = models.TrainingRun(
        training_job_id=job.id,
        dataset_version_id=version.id,
        python_version="3.13.7",
        random_seed=42,
        status="SUCCEEDED",
    )
    db.add(train_run)
    db.commit()

    candidate = models.ModelCandidate(
        training_run_id=train_run.id,
        dataset_version_id=version.id,
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="models:/metro-traffic-volume/1",
        artifact_uri="runs:/fake/candidate_model",
        status="PROMOTED",
    )
    db.add(candidate)
    db.flush()

    prod_env = models.DeploymentEnvironment(name="production", active_candidate_id=candidate.id)
    db.add(prod_env)
    db.commit()

    return {"candidate": candidate, "env": prod_env}


SAMPLE_VALID_INPUT = [
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


def test_inference_resolves_active_production_champion(db: Session, prod_model_setup):
    service = InferenceService(db=db, tracking_uri="sqlite:///:memory:")

    mock_estimator = mock.MagicMock()
    mock_estimator.predict.return_value = np.array([3410.2])

    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        req = PredictionRequest(inputs=SAMPLE_VALID_INPUT)
        res = service.predict(req)

    assert res.inference_id is not None
    assert res.model_name == "metro-traffic-volume"
    assert res.model_version == "1"
    assert res.predictions == [3410.2]
    assert res.latency_ms > 0


def test_inference_input_validation_missing_feature_rejected(db: Session, prod_model_setup):
    service = InferenceService(db=db, tracking_uri="sqlite:///:memory:")

    # Missing 'temp' and 'clouds_all'
    invalid_input = [
        {
            "holiday": "None",
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "weather_main": "Clouds",
            "weather_description": "scattered clouds",
            "date_time": "2026-11-21 09:00:00",
        }
    ]

    with mock.patch("mlflow.sklearn.load_model") as mock_load:
        with pytest.raises(InputContractViolationError) as exc_info:
            service.predict(PredictionRequest(inputs=invalid_input))
        assert "Missing required feature(s)" in str(exc_info.value)
        # Estimator must NEVER be loaded or called on invalid input
        mock_load.assert_not_called()


def test_inference_input_validation_invalid_timestamp_rejected(db: Session, prod_model_setup):
    service = InferenceService(db=db, tracking_uri="sqlite:///:memory:")

    invalid_input = [
        {
            "holiday": "None",
            "temp": 288.15,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "weather_main": "Clouds",
            "weather_description": "scattered clouds",
            "date_time": "not-a-valid-datetime-value",
        }
    ]

    with pytest.raises(InputContractViolationError) as exc_info:
        service.predict(PredictionRequest(inputs=invalid_input))
    assert "Invalid timestamp" in str(exc_info.value)


def test_inference_telemetry_persisted_privacy_safe(db: Session, prod_model_setup):
    service = InferenceService(db=db, tracking_uri="sqlite:///:memory:")

    mock_estimator = mock.MagicMock(predict=lambda x: np.array([2150.0]))
    with mock.patch("mlflow.sklearn.load_model", return_value=mock_estimator):
        res = service.predict(PredictionRequest(inputs=SAMPLE_VALID_INPUT))

    event = db.query(models.PredictionEvent).filter(models.PredictionEvent.inference_id == res.inference_id).first()
    assert event is not None
    assert event.status == "SUCCESS"
    assert event.latency_ms > 0
    assert event.model_version == "1"

    # Confirm raw input payload is NOT stored in database
    summary = event.input_summary_json
    assert "features_received" in summary
    assert "null_counts" in summary
    assert "row_count" in summary
    assert "temp" not in summary  # Raw values omitted
    assert "date_time" not in summary


def test_inference_no_production_model_raises_controlled_error(db: Session):
    # Empty DB with no production model
    service = InferenceService(db=db, tracking_uri="sqlite:///:memory:")

    with pytest.raises(EntityNotFoundError) as exc_info:
        service.predict(PredictionRequest(inputs=SAMPLE_VALID_INPUT))
    assert "No active production model deployed" in str(exc_info.value)
