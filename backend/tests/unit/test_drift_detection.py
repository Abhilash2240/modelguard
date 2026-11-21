import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import models
from app.services.drift_service import DriftAnalysisService, compute_categorical_tvd, compute_psi


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_traffic_df():
    return pd.DataFrame(
        {
            "holiday": ["None"] * 50,
            "temp": np.linspace(280.0, 300.0, 50),
            "rain_1h": [0.0] * 50,
            "snow_1h": [0.0] * 50,
            "clouds_all": [20] * 25 + [80] * 25,
            "weather_main": ["Clear"] * 25 + ["Clouds"] * 25,
            "weather_description": ["sky is clear"] * 25 + ["few clouds"] * 25,
            "date_time": ["2026-11-21 09:00:00"] * 50,
            "traffic_volume": np.linspace(1000, 5000, 50),
        }
    )


def test_compute_psi_identical_distributions():
    arr = np.random.normal(loc=10.0, scale=2.0, size=500)
    psi = compute_psi(arr, arr)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_compute_psi_shifted_distributions():
    ref = np.random.normal(loc=10.0, scale=2.0, size=500)
    curr = np.random.normal(loc=18.0, scale=2.0, size=500)  # Significant shift
    psi = compute_psi(ref, curr)
    assert psi > 0.25


def test_compute_categorical_tvd():
    s_ref = pd.Series(["Clear"] * 50 + ["Clouds"] * 50)
    s_curr = pd.Series(["Clear"] * 10 + ["Rain"] * 90)
    tvd = compute_categorical_tvd(s_ref, s_curr)
    assert tvd > 0.30


def test_input_drift_detection_with_temperature_shock(db_session, sample_traffic_df):
    """Test input drift detection on continuous feature with statistical evidence."""
    service = DriftAnalysisService(db_session)

    # Current data has +30K temperature shift
    current_df = sample_traffic_df.copy()
    current_df["temp"] = current_df["temp"] + 30.0

    report = service.analyze_input_drift(
        current_df=current_df,
        reference_df=sample_traffic_df,
        model_name="metro-traffic-volume",
        model_version="1",
        alpha=0.05,
    )

    assert report.is_drift_detected is True
    assert report.report_type == "DATA_DRIFT"
    assert report.reference_sample_count == 50
    assert report.current_sample_count == 50
    assert "temp" in report.summary_json["drifting_features"]
    assert "Input drift measures feature-level covariate divergence" in report.limitations
    assert "do NOT prove model accuracy degradation" in report.limitations

    # Confirm alert was generated
    alert = db_session.query(models.MonitoringAlert).filter(models.MonitoringAlert.report_id == report.id).first()
    assert alert is not None
    assert alert.alert_type == "DATA_DRIFT"


def test_input_drift_detection_with_missing_value_spike(db_session, sample_traffic_df):
    """Test data quality drift on missing value rate increase."""
    service = DriftAnalysisService(db_session)

    current_df = sample_traffic_df.copy()
    # Mask 40% of clouds_all to NaN
    current_df.loc[0:20, "clouds_all"] = np.nan

    report = service.analyze_input_drift(
        current_df=current_df,
        reference_df=sample_traffic_df,
        model_name="metro-traffic-volume",
        model_version="1",
    )

    assert report.is_drift_detected is True
    assert "clouds_all" in report.summary_json["drifting_features"]


def test_prediction_drift_separated_from_input_drift(db_session):
    """Verify prediction drift is evaluated independently and does not claim accuracy loss."""
    service = DriftAnalysisService(db_session)

    ref_preds = np.random.normal(loc=3000, scale=400, size=100)
    curr_preds = np.random.normal(loc=5500, scale=400, size=100)  # Distinct shift

    report = service.analyze_prediction_drift(
        current_predictions=curr_preds,
        reference_predictions=ref_preds,
        model_name="metro-traffic-volume",
        model_version="1",
    )

    assert report.is_drift_detected is True
    assert report.report_type == "PREDICTION_DRIFT"
    assert "Prediction drift measures shifts in model output distribution" in report.limitations
    assert "must not be conflated with input drift or verified model degradation" in report.limitations

    # Confirm PREDICTION_DRIFT alert
    alert = (
        db_session.query(models.MonitoringAlert).filter(models.MonitoringAlert.alert_type == "PREDICTION_DRIFT").first()
    )
    assert alert is not None
    assert "Prediction distribution shift detected" in alert.message
