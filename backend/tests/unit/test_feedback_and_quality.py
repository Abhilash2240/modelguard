import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.errors import EntityNotFoundError, FeedbackConflictError
from app.db import models
from app.services.feedback_service import FeedbackService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # Seed default environments
    session.add(models.DeploymentEnvironment(name="staging"))
    session.add(models.DeploymentEnvironment(name="production"))
    session.commit()

    try:
        yield session
    finally:
        session.close()


def test_feedback_submission_success(db_session):
    """Test successful ground truth feedback ingestion for valid inference."""
    event = models.PredictionEvent(
        inference_id="test-inf-1",
        model_name="metro-traffic-volume",
        model_version="1",
        latency_ms=10.0,
        status="SUCCESS",
        prediction_json={"predictions": [3000.0], "count": 1},
        input_summary_json={"row_count": 1},
    )
    db_session.add(event)
    db_session.commit()

    service = FeedbackService(db_session)
    feedback = service.submit_feedback(
        inference_id="test-inf-1",
        actual_value=3100.0,
        submitted_by="sensor_1",
        notes="Traffic count verified",
    )

    assert feedback.inference_id == "test-inf-1"
    assert feedback.actual_value == 3100.0
    assert feedback.submitted_by == "sensor_1"
    assert feedback.notes == "Traffic count verified"

    # Confirm historical prediction was not modified
    db_session.refresh(event)
    assert event.prediction_json == {"predictions": [3000.0], "count": 1}


def test_feedback_non_existent_inference_rejected(db_session):
    """Verify submitting feedback for unknown inference ID raises EntityNotFoundError."""
    service = FeedbackService(db_session)
    with pytest.raises(EntityNotFoundError) as exc_info:
        service.submit_feedback(inference_id="unknown-id", actual_value=123.0)
    assert "unknown-id" in str(exc_info.value)


def test_feedback_idempotent_resubmission(db_session):
    """Verify re-submitting exact same ground-truth value is idempotent and returns existing record."""
    event = models.PredictionEvent(
        inference_id="test-inf-idem",
        model_name="metro-traffic-volume",
        model_version="1",
        latency_ms=8.0,
        status="SUCCESS",
        prediction_json={"predictions": [2500.0], "count": 1},
    )
    db_session.add(event)
    db_session.commit()

    service = FeedbackService(db_session)
    first = service.submit_feedback(inference_id="test-inf-idem", actual_value=2550.0)
    # Re-submit identical value
    second = service.submit_feedback(inference_id="test-inf-idem", actual_value=2550.0)

    assert first.id == second.id
    assert second.actual_value == 2550.0

    # Ensure only 1 record exists in database
    feedbacks = (
        db_session.query(models.PredictionFeedback)
        .filter(models.PredictionFeedback.inference_id == "test-inf-idem")
        .all()
    )
    assert len(feedbacks) == 1


def test_feedback_conflicting_submission_rejected(db_session):
    """Verify submitting conflicting ground truth for same inference ID raises FeedbackConflictError."""
    event = models.PredictionEvent(
        inference_id="test-inf-conflict",
        model_name="metro-traffic-volume",
        model_version="1",
        latency_ms=9.0,
        status="SUCCESS",
        prediction_json={"predictions": [4000.0], "count": 1},
    )
    db_session.add(event)
    db_session.commit()

    service = FeedbackService(db_session)
    service.submit_feedback(inference_id="test-inf-conflict", actual_value=4100.0)

    # Submitting different value must raise FeedbackConflictError
    with pytest.raises(FeedbackConflictError) as exc_info:
        service.submit_feedback(inference_id="test-inf-conflict", actual_value=5000.0)
    assert "Conflicting ground-truth value" in str(exc_info.value)


def test_quality_insufficient_labels_handled_explicitly(db_session):
    """Confirm quality evaluation handles insufficient labels without reporting misleading scores."""
    # Create 3 events with feedback (threshold is 10)
    service = FeedbackService(db_session)
    for i in range(3):
        inf_id = f"inf-insuff-{i}"
        ev = models.PredictionEvent(
            inference_id=inf_id,
            model_name="metro-traffic-volume",
            model_version="1",
            latency_ms=10.0,
            status="SUCCESS",
            prediction_json={"predictions": [3000.0 + i * 100], "count": 1},
        )
        db_session.add(ev)
        db_session.flush()
        service.submit_feedback(inference_id=inf_id, actual_value=3050.0 + i * 100)

    report = service.calculate_rolling_quality(
        model_name="metro-traffic-volume",
        model_version="1",
        min_samples=10,
    )

    assert report.status == "INSUFFICIENT_LABELS"
    assert report.sample_count == 3
    assert report.labeled_sample_count == 3
    assert report.mae is None
    assert report.rmse is None
    assert report.r2 is None
    assert "Insufficient labeled samples" in report.details_json["message"]


def test_quality_sufficient_labels_and_degradation_alert(db_session):
    """Verify rolling evaluation calculates regression metrics and raises alert when degraded."""
    service = FeedbackService(db_session)
    for i in range(12):
        inf_id = f"inf-deg-{i}"
        pred = 3000.0 + i * 50
        # High error: actuals diverge by 500 units (MAE = 500 > 450 threshold)
        actual = pred + 500.0
        ev = models.PredictionEvent(
            inference_id=inf_id,
            model_name="metro-traffic-volume",
            model_version="1",
            latency_ms=12.0,
            status="SUCCESS",
            prediction_json={"predictions": [pred], "count": 1},
        )
        db_session.add(ev)
        db_session.flush()
        service.submit_feedback(inference_id=inf_id, actual_value=actual)

    report = service.calculate_rolling_quality(
        model_name="metro-traffic-volume",
        model_version="1",
        min_samples=10,
        max_mae_threshold=450.0,
    )

    assert report.status == "SUFFICIENT_LABELS"
    assert report.sample_count == 12
    assert report.labeled_sample_count == 12
    assert report.label_coverage == 100.0
    assert report.mae == pytest.approx(500.0, rel=1e-2)
    assert report.rmse == pytest.approx(500.0, rel=1e-2)

    # Confirm alert was generated
    alert = (
        db_session.query(models.MonitoringAlert)
        .filter(models.MonitoringAlert.alert_type == "QUALITY_DEGRADATION")
        .first()
    )
    assert alert is not None
    assert alert.severity == "WARNING" or alert.severity == "CRITICAL"
    assert alert.report_id == report.id


def test_quality_no_active_champion_explicit_status(db_session):
    """Verify that when no champion is deployed to production, quality returns NO_ACTIVE_CHAMPION status."""
    service = FeedbackService(db_session)
    # Ensure no production environment exists or has no active candidate
    report = service.calculate_rolling_quality(model_name="metro-traffic-volume", model_version=None)
    assert report.status == "NO_ACTIVE_CHAMPION"
    assert report.model_version == "NONE"
    assert report.details_json["min_samples_required"] == 10
    assert "No active champion deployed" in report.details_json["message"]
