"""Unit and regression tests for Colab exported training artifacts and reproducibility."""

import json
from collections import UserList
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn.compose._column_transformer as _ct
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure backwards-compatibility for models serialized with scikit-learn <= 1.6
if not hasattr(_ct, "_RemainderColsList"):

    class _RemainderColsList(UserList):
        def __init__(self, columns=None, *args, **kwargs):
            super().__init__(columns or [])
            self.future_dtype = kwargs.get("future_dtype")
            self.warning_was_emitted = kwargs.get("warning_was_emitted", False)
            self.warning_enabled = kwargs.get("warning_enabled", True)

    _ct._RemainderColsList = _RemainderColsList


@pytest.fixture
def artifacts_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    colab_dir = repo_root / "experiments" / "outputs" / "colab_run"
    assert colab_dir.is_dir(), f"Colab outputs directory not found at: {colab_dir}"
    return colab_dir


@pytest.fixture
def reproducibility_bundle(artifacts_dir: Path) -> dict:
    bundle_path = artifacts_dir / "reproducibility_bundle.json"
    assert bundle_path.is_file(), f"Reproducibility bundle not found at: {bundle_path}"
    with bundle_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_artifact_inventory_and_execution_status(artifacts_dir: Path, reproducibility_bundle: dict):
    expected_files = [
        "reproducibility_bundle.json",
        "candidate_random_forest.joblib",
        "baseline_dummy.joblib",
        "validation_predictions.csv",
        "test_predictions.csv",
    ]
    for fname in expected_files:
        p = artifacts_dir / fname
        assert p.is_file(), f"Expected artifact '{fname}' missing from {artifacts_dir}"
        assert p.stat().st_size > 0, f"Artifact '{fname}' is empty"

    assert reproducibility_bundle.get("execution_status") == "COMPLETED"


def test_dataset_provenance_in_bundle(reproducibility_bundle: dict):
    expected_uuid = "8da56ac6-8614-4789-943c-5e612008e77b"
    expected_tag = "v1.0.1"
    expected_sha256 = "dad82c80c03d6cd881d60e1acc8e1237a75e241727fe367b997aa62f83c32201"

    assert reproducibility_bundle["dataset_version_id"] == expected_uuid
    assert reproducibility_bundle["dataset_version_tag"] == expected_tag
    assert reproducibility_bundle["dataset_sha256"] == expected_sha256

    split_manifest = reproducibility_bundle["split_manifest"]
    assert split_manifest["dataset_version_id"] == expected_uuid
    assert split_manifest["dataset_sha256"] == expected_sha256
    assert split_manifest["total_rows"] == 48204


def test_metrics_and_prediction_recomputation(artifacts_dir: Path, reproducibility_bundle: dict):
    val_df = pd.read_csv(artifacts_dir / "validation_predictions.csv")
    test_df = pd.read_csv(artifacts_dir / "test_predictions.csv")

    assert len(val_df) == 7230
    assert len(test_df) == 7231

    # Candidate validation metrics
    c_val_mae = mean_absolute_error(val_df["actual"], val_df["candidate_predicted"])
    c_val_rmse = np.sqrt(mean_squared_error(val_df["actual"], val_df["candidate_predicted"]))
    c_val_r2 = r2_score(val_df["actual"], val_df["candidate_predicted"])

    bundle_c_val = reproducibility_bundle["candidate_validation_metrics"]
    assert abs(c_val_mae - bundle_c_val["mae"]) < 1e-3
    assert abs(c_val_rmse - bundle_c_val["rmse"]) < 1e-3
    assert abs(c_val_r2 - bundle_c_val["r2"]) < 1e-3

    # Candidate test metrics
    test_mae = mean_absolute_error(test_df["actual"], test_df["predicted"])
    test_rmse = np.sqrt(mean_squared_error(test_df["actual"], test_df["predicted"]))
    test_r2 = r2_score(test_df["actual"], test_df["predicted"])

    bundle_test = reproducibility_bundle["final_test_metrics"]
    assert abs(test_mae - bundle_test["mae"]) < 1e-3
    assert abs(test_rmse - bundle_test["rmse"]) < 1e-3
    assert abs(test_r2 - bundle_test["r2"]) < 1e-3

    # Residuals
    assert np.allclose(val_df["actual"] - val_df["candidate_predicted"], val_df["candidate_residual"])
    assert np.allclose(test_df["actual"] - test_df["predicted"], test_df["residual"])


def test_model_artifact_loading_and_prediction(artifacts_dir: Path):
    candidate_pipe = joblib.load(artifacts_dir / "candidate_random_forest.joblib")
    baseline_pipe = joblib.load(artifacts_dir / "baseline_dummy.joblib")

    assert "preprocessor" in candidate_pipe.named_steps
    assert "regressor" in candidate_pipe.named_steps
    assert "preprocessor" in baseline_pipe.named_steps
    assert "regressor" in baseline_pipe.named_steps

    sample_df = pd.DataFrame(
        {
            "holiday": [None, "None"],
            "temp": [288.0, 290.0],
            "rain_1h": [0.0, 5.0],
            "snow_1h": [0.0, 0.0],
            "clouds_all": [40, 90],
            "weather_main": ["Clouds", "Rain"],
            "weather_description": ["scattered clouds", "light rain"],
            "hour": [9, 10],
            "day_of_week": [1, 1],
            "month": [10, 10],
            "is_weekend": [0, 0],
        }
    )

    c_preds = candidate_pipe.predict(sample_df)
    b_preds = baseline_pipe.predict(sample_df)

    assert len(c_preds) == 2
    assert len(b_preds) == 2
    assert np.all(np.isfinite(c_preds))
    assert np.all(np.isfinite(b_preds))
