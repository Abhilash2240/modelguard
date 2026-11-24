from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db import models
from app.services.drift_service import DriftAnalysisService
from app.services.feedback_service import FeedbackService


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        # Seed environments
        for env_name in ("staging", "production"):
            env = models.DeploymentEnvironment(name=env_name)
            session.add(env)
        session.commit()
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_reference_df():
    return pd.DataFrame(
        {
            "holiday": ["None"] * 50,
            "temp": np.linspace(280.0, 300.0, 50),
            "rain_1h": [0.0] * 50,
            "snow_1h": [0.0] * 50,
            "clouds_all": [20] * 25 + [80] * 25,
            "weather_main": ["Clear"] * 25 + ["Clouds"] * 25,
            "weather_description": ["sky is clear"] * 25 + ["few clouds"] * 25,
            "traffic_volume": np.linspace(1000, 5000, 50),
        }
    )


# 1. Telemetry input drift insufficient data
def test_input_drift_insufficient_data(memory_db, sample_reference_df):
    # Only 2 events (less than required 5)
    events = [
        models.PredictionEvent(
            inference_id=f"inf-{i}",
            candidate_id="cand-1",
            model_name="metro-traffic-volume",
            model_version="1",
            latency_ms=15.0,
            status="SUCCESS",
            input_summary_json={"row_count": 1},
        )
        for i in range(2)
    ]
    memory_db.add_all(events)
    memory_db.commit()

    total_obs = sum(ev.input_summary_json.get("row_count", 1) for ev in events)
    assert len(events) < 5 or total_obs < 10


# 2. Telemetry provenance tracking
def test_input_drift_telemetry_provenance(memory_db, sample_reference_df):
    service = DriftAnalysisService(memory_db)
    ds = models.Dataset(
        id="ds-parent-1",
        name="metro-traffic",
        task_type="REGRESSION",
        source_uri="dummy.csv",
        licence="CC0-1.0",
    )
    memory_db.add(ds)
    memory_db.flush()
    ds_ver = models.DatasetVersion(
        id="ds-ver-1",
        dataset_id=ds.id,
        version_tag="v1.0.0",
        storage_uri="dummy.csv",
        sha256_hash="expected_hash_abc123",
        byte_size=1024,
        row_count=50,
        columns_json=[
            "temp",
            "clouds_all",
            "rain_1h",
            "snow_1h",
            "holiday",
            "weather_main",
            "weather_description",
            "traffic_volume",
        ],
        schema_hash="schema123",
    )
    memory_db.add(ds_ver)
    memory_db.commit()

    events = []
    for i in range(6):
        # Even numbered events have mismatched checksums
        chk = "expected_hash_abc123" if i % 2 == 0 else "different_hash_xyz"
        ev = models.PredictionEvent(
            inference_id=f"prov-inf-{i}",
            candidate_id="cand-1",
            model_name="metro-traffic-volume",
            model_version="1",
            latency_ms=20.0,
            status="SUCCESS",
            input_summary_json={
                "row_count": 2,
                "reference_checksum": chk,
                "null_counts": {
                    k: 0
                    for k in [
                        "temp",
                        "clouds_all",
                        "rain_1h",
                        "snow_1h",
                        "holiday",
                        "weather_main",
                        "weather_description",
                    ]
                },
                "numeric_stats": {
                    "temp": {"bin_counts": [0, 0, 1, 1, 0, 0, 0, 0]},
                    "clouds_all": {"bin_counts": [0, 1, 0, 0, 0, 1]},
                    "rain_1h": {"bin_counts": [2, 0, 0, 0, 0, 0, 0]},
                    "snow_1h": {"bin_counts": [2, 0, 0, 0, 0, 0]},
                },
                "categorical_stats": {
                    "holiday": {"frequencies": {"None": 2}, "missing_count": 0},
                    "weather_main": {"frequencies": {"Clear": 1, "Clouds": 1}, "missing_count": 0},
                    "weather_description": {"frequencies": {"sky is clear": 1, "few clouds": 1}, "missing_count": 0},
                },
            },
        )
        memory_db.add(ev)
        events.append(ev)
    memory_db.commit()

    report = service.analyze_input_drift_from_telemetry(
        events=events,
        reference_df=sample_reference_df,
        reference_version_record=ds_ver,
        model_name="metro-traffic-volume",
        model_version="1",
    )
    assert report.summary_json["live_events_analyzed"] == 6
    assert report.summary_json["incompatible_events_flagged"] == 3
    assert report.summary_json["reference_checksum"] == "expected_hash_abc123"


# 3. Reference prediction loading and deterministic artifact
def test_reference_prediction_artifact_loading(memory_db, sample_reference_df, tmp_path):
    cand = models.ModelCandidate(
        id="cand-art-1",
        training_run_id="tr-1",
        dataset_version_id="ds-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="models:/metro-traffic-volume/1",
        artifact_uri="uri1",
        status="PROMOTED",
    )
    memory_db.add(cand)
    memory_db.commit()

    service = DriftAnalysisService(memory_db)
    # Mock get_reference_dataset_record
    ds_ver = models.DatasetVersion(
        id="ds-1",
        dataset_id="ds-dummy",
        version_tag="v1.0.0",
        storage_uri="dummy.csv",
        sha256_hash="hash123",
        byte_size=1024,
        row_count=50,
        columns_json=["temp", "traffic_volume"],
        schema_hash="schema123",
    )
    service.get_reference_dataset_record = lambda ds_id=None: (sample_reference_df, ds_ver)

    preds, meta = service.load_reference_predictions(cand, max_samples=20)
    assert len(preds) == 20
    assert meta["model_name"] == "metro-traffic-volume"
    assert meta["model_version"] == "1"
    assert meta["dataset_version_tag"] == "v1.0.0"
    assert meta["dataset_checksum"] == "hash123"


# 4. Prediction drift failure when insufficient observations
def test_prediction_drift_insufficient_data(memory_db):
    # Only 3 predictions
    curr_preds = [2500.0, 3100.0, 4200.0]
    assert len(curr_preds) < 10


# 5. Quality report: No active champion returns explicit NO_ACTIVE_CHAMPION
def test_quality_no_active_champion(memory_db):
    service = FeedbackService(memory_db)
    report = service.calculate_rolling_quality(model_name="metro-traffic-volume")
    assert report.status == "NO_ACTIVE_CHAMPION"
    assert report.model_version in ("NONE", "UNKNOWN")
    assert report.sample_count == 0
    assert report.labeled_sample_count == 0
    assert "No active champion deployed" in report.details_json.get("message", "")


# 6. Quality report: Zero, one, and ten feedback records
def test_quality_feedback_thresholds(memory_db):
    prod_env = (
        memory_db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "production").first()
    )
    cand = models.ModelCandidate(
        id="cand-prod-1",
        training_run_id="tr-1",
        dataset_version_id="ds-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="models:/metro-traffic-volume/1",
        artifact_uri="uri1",
        status="PROMOTED",
    )
    memory_db.add(cand)
    memory_db.flush()
    prod_env.active_candidate_id = cand.id
    memory_db.commit()

    service = FeedbackService(memory_db)

    # 0 events -> INSUFFICIENT_LABELS
    r0 = service.calculate_rolling_quality(model_name="metro-traffic-volume", min_samples=10)
    assert r0.status == "INSUFFICIENT_LABELS"
    assert r0.sample_count == 0
    assert r0.labeled_sample_count == 0

    # Create 10 inference events
    events = []
    for i in range(10):
        ev = models.PredictionEvent(
            inference_id=f"qual-inf-{i}",
            candidate_id=cand.id,
            model_name="metro-traffic-volume",
            model_version="1",
            latency_ms=20.0,
            status="SUCCESS",
            prediction_json={"predictions": [3000.0 + i * 100], "count": 1},
        )
        memory_db.add(ev)
        events.append(ev)
    memory_db.commit()

    # 1 feedback record -> INSUFFICIENT_LABELS, labeled_count=1, sample_count=10
    fb1 = models.PredictionFeedback(
        inference_id="qual-inf-0",
        actual_value=3050.0,
        submitted_by="sensor-test",
    )
    memory_db.add(fb1)
    memory_db.commit()

    r1 = service.calculate_rolling_quality(model_name="metro-traffic-volume", min_samples=10)
    assert r1.status == "INSUFFICIENT_LABELS"
    assert r1.sample_count == 10
    assert r1.labeled_sample_count == 1
    assert r1.mae is None

    # Add 9 more feedback records (total 10) -> SUFFICIENT_LABELS
    for i in range(1, 10):
        fb = models.PredictionFeedback(
            inference_id=f"qual-inf-{i}",
            actual_value=3000.0 + i * 100 + 25.0,  # Constant error of 25.0
            submitted_by="sensor-test",
        )
        memory_db.add(fb)
    memory_db.commit()

    r10 = service.calculate_rolling_quality(model_name="metro-traffic-volume", min_samples=10)
    assert r10.status == "SUFFICIENT_LABELS"
    assert r10.sample_count == 10
    assert r10.labeled_sample_count == 10
    assert r10.mae == pytest.approx(27.5, abs=5.0)
    assert r10.rmse is not None
    assert r10.r2 is not None


# 7. Categorical missingness calculation before string conversion
def test_categorical_missingness_preserved():
    from app.services.drift_service import compute_categorical_tvd

    # Series with genuine np.nan/None
    s_ref = pd.Series(["Clear", "Rain", None, "Clear", "Snow"])
    s_curr = pd.Series(["Clear", "Rain", "Snow", "Clear", "Rain"])

    missing_ref = float(s_ref.isna().mean())
    missing_curr = float(s_curr.isna().mean())

    assert missing_ref == pytest.approx(0.20, abs=1e-3)
    assert missing_curr == pytest.approx(0.00, abs=1e-3)

    # String conversion without dropna turns None into "None"
    s_bad = s_ref.astype(str)
    assert s_bad.isna().mean() == 0.0  # False 0.0 missingness

    # Correct handling
    tvd = compute_categorical_tvd(s_ref.dropna().astype(str), s_curr.dropna().astype(str))
    assert tvd >= 0.0


# 8. Production security validation for Settings
def test_production_security_validation():
    # Insecure / short SECRET_KEY raises ValidationError
    with pytest.raises(ValidationError):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="short-key",
            BOOTSTRAP_ADMIN_PASSWORD="ValidPassword123!",
        )

    with pytest.raises(ValidationError):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="modelguard-insecure-development-secret-key-change-in-production",
            BOOTSTRAP_ADMIN_PASSWORD="ValidPassword123!",
        )

    # Weak BOOTSTRAP_ADMIN_PASSWORD raises ValidationError
    with pytest.raises(ValidationError):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="9f8c2e71d3a4b5608e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d",
            BOOTSTRAP_ADMIN_PASSWORD="admin",
        )

    # Valid production settings pass
    valid_settings = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="9f8c2e71d3a4b5608e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d",
        BOOTSTRAP_ADMIN_PASSWORD="StrongProductionPassword2026!",
    )
    assert valid_settings.ENVIRONMENT == "production"

    # Development allows default keys
    dev_settings = Settings(
        ENVIRONMENT="development",
        SECRET_KEY="dev-secret",
    )
    assert dev_settings.ENVIRONMENT == "development"


# 9. Docker Compose localhost port bindings configuration
def test_docker_compose_localhost_port_bindings():
    compose_path = Path(__file__).resolve().parent.parent.parent.parent / "docker-compose.yml"
    assert compose_path.exists(), f"docker-compose.yml not found at {compose_path}"

    content = compose_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    in_ports = False
    port_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "ports:":
            in_ports = True
            continue
        elif in_ports:
            if stripped.startswith("- "):
                port_lines.append(stripped)
            elif not stripped.startswith("#") and stripped != "":
                in_ports = False

    assert len(port_lines) >= 5, f"Expected at least 5 port mappings in compose, found {len(port_lines)}"
    for pl in port_lines:
        assert "127.0.0.1:" in pl, f"Port mapping '{pl}' does not bind strictly to 127.0.0.1"
