import unittest.mock as mock

import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from app.core.errors import (
    ArtifactLoggingError,
    TrackingInitializationError,
    TrackingLoggingError,
)
from app.ml.tracking import MLflowTracker, NullTracker


def test_null_tracker_semantics_returns_none():
    tracker = NullTracker()
    run_id = tracker.start_training_run(
        job_id="job-123",
        run_id="run-456",
        dataset_version_id="ver-789",
    )
    assert run_id is None
    assert tracker.get_run_id() is None
    assert tracker.get_artifact_uri() is None
    assert tracker.resolve_experiment("any-name") is None

    # Verify no-ops do not raise errors
    tracker.log_parameters({"key": "val"})
    tracker.log_metrics({"mae": 10.5})
    tracker.log_models(Pipeline([("dummy", DummyRegressor())]), Pipeline([("dummy", DummyRegressor())]))
    tracker.log_artifacts({"file.json": {"data": 1}})
    tracker.end_run(status="FINISHED")


def test_tracking_initialization_failure():
    with mock.patch("mlflow.start_run", side_effect=Exception("Connection refused")):
        with mock.patch(
            "mlflow.tracking.MlflowClient.get_experiment_by_name", return_value=mock.MagicMock(experiment_id="exp-1")
        ):
            tracker = MLflowTracker(tracking_uri="http://fake-host:5000")
            with pytest.raises(TrackingInitializationError) as exc_info:
                tracker.start_training_run("job-1", "run-1", "ver-1")
            assert "Failed to start MLflow run" in str(exc_info.value)


def test_tracking_failure_during_metric_logging():
    tracker = MLflowTracker(tracking_uri="http://fake-host:5000")
    tracker._active_run_id = "test-run-123"

    with mock.patch("mlflow.log_metrics", side_effect=Exception("Store unavailable")):
        with pytest.raises(TrackingLoggingError) as exc_info:
            tracker.log_metrics({"candidate_val_mae": 100.0})
        assert "Failed to log MLflow metrics" in str(exc_info.value)


def test_tracking_failure_during_artifact_logging():
    tracker = MLflowTracker(tracking_uri="http://fake-host:5000")
    tracker._active_run_id = "test-run-123"

    with mock.patch("mlflow.log_artifact", side_effect=Exception("Disk full")):
        with pytest.raises(ArtifactLoggingError) as exc_info:
            tracker.log_artifacts({"test.json": {"key": "value"}})
        assert "Failed to log artifacts to MLflow" in str(exc_info.value)


def test_unambiguous_parameter_and_metric_namespaces():
    tracker = MLflowTracker(tracking_uri="http://fake-host:5000")
    tracker._active_run_id = "test-run-123"

    logged_params = {}
    logged_metrics = {}

    def mock_log_params(p):
        logged_params.update(p)

    def mock_log_metrics(m, step=None):
        logged_metrics.update(m)

    with mock.patch("mlflow.log_params", side_effect=mock_log_params):
        with mock.patch("mlflow.log_metrics", side_effect=mock_log_metrics):
            tracker.log_parameters(
                {
                    "candidate_n_estimators": 100,
                    "baseline_strategy": "mean",
                    "split_train_ratio": 0.70,
                }
            )
            tracker.log_metrics(
                {
                    "candidate_val_mae": 245.12,
                    "baseline_val_mae": 1720.50,
                    "comparison_delta_mae": 1475.38,
                }
            )

    assert "candidate_n_estimators" in logged_params
    assert "baseline_strategy" in logged_params
    assert "split_train_ratio" in logged_params
    assert logged_params["candidate_n_estimators"] == 100

    assert "candidate_val_mae" in logged_metrics
    assert "baseline_val_mae" in logged_metrics
    assert "comparison_delta_mae" in logged_metrics
    assert logged_metrics["candidate_val_mae"] == 245.12


def test_mandatory_identity_tags_present():
    with mock.patch(
        "mlflow.tracking.MlflowClient.get_experiment_by_name", return_value=mock.MagicMock(experiment_id="exp-1")
    ):
        with mock.patch("mlflow.start_run") as mock_start_run:
            mock_run = mock.MagicMock()
            mock_run.info.run_id = "mlf-999"
            mock_run.info.artifact_uri = "file:///artifacts/mlf-999"
            mock_start_run.return_value = mock_run

            tracker = MLflowTracker(tracking_uri="http://fake-host:5000")
            tracker.start_training_run(
                job_id="job-uuid-111",
                run_id="run-uuid-222",
                dataset_version_id="ver-uuid-333",
                tags={"modelguard.custom": "custom-val"},
            )

            assert mock_start_run.called
            _, kwargs = mock_start_run.call_args
            passed_tags = kwargs["tags"]
            assert passed_tags["modelguard.training_job_id"] == "job-uuid-111"
            assert passed_tags["modelguard.training_run_id"] == "run-uuid-222"
            assert passed_tags["modelguard.dataset_version_id"] == "ver-uuid-333"
            assert passed_tags["modelguard.custom"] == "custom-val"
