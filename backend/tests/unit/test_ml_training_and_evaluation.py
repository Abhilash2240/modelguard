from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.evaluation import calculate_metrics, calculate_residuals, compare_models
from app.ml.training import TrainingService
from app.schemas.evaluation import PartitionEvaluation, RegressionMetrics, ResidualSummary
from app.schemas.training import SplitConfig


def test_calculate_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 19.0, 29.0, 42.0, 51.0])

    metrics = calculate_metrics(y_true, y_pred)
    # MAE = (2 + 1 + 1 + 2 + 1) / 5 = 7 / 5 = 1.4
    assert metrics.mae == 1.4
    # RMSE = sqrt((4 + 1 + 1 + 4 + 1)/5) = sqrt(11/5) = sqrt(2.2) approx 1.4832
    assert metrics.rmse == pytest.approx(1.4832, abs=0.001)
    assert metrics.r2 > 0.95
    assert metrics.sample_count == 5


def test_calculate_residuals():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([8.0, 20.0, 35.0])
    # Residuals = [2.0, 0.0, -5.0]

    residuals = calculate_residuals(y_true, y_pred)
    assert residuals.min == -5.0
    assert residuals.max == 2.0
    assert residuals.mean == pytest.approx(-1.0, abs=0.01)


def test_compare_models():
    base_eval = PartitionEvaluation(
        model_name="baseline",
        partition_name="validation",
        metrics=RegressionMetrics(mae=100.0, rmse=150.0, r2=0.10, sample_count=50),
        residuals=ResidualSummary(mean=0, std=150, min=-300, max=300, p25=-50, p50=0, p75=50, p95=100),
    )
    cand_eval = PartitionEvaluation(
        model_name="candidate",
        partition_name="validation",
        metrics=RegressionMetrics(mae=50.0, rmse=70.0, r2=0.85, sample_count=50),
        residuals=ResidualSummary(mean=0, std=70, min=-150, max=150, p25=-20, p50=0, p75=20, p95=50),
    )

    comp = compare_models(base_eval, cand_eval)
    assert comp.delta_mae == -50.0
    assert comp.delta_rmse == -80.0
    assert comp.delta_r2 == 0.75
    assert comp.candidate_superior is True


def test_training_service_end_to_end(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    # valid_csv has 10 rows; set min_observations=5 for testing
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
    service = TrainingService(split_config=config)

    result, test_df = service.train_and_evaluate_validation(
        df=df,
        dataset_version_id="ver-test",
        dataset_sha256="fake-sha-256",
        candidate_hyperparameters={"n_estimators": 10, "max_depth": 4},
    )

    # 1. Validation results must be populated
    assert result.baseline_val_eval.metrics.sample_count > 0
    assert result.candidate_val_eval.metrics.sample_count > 0
    assert result.comparison is not None
    assert len(result.val_predictions) > 0

    # 2. Test holdout is untouched and returned separately
    assert len(test_df) > 0
    test_eval, test_preds = service.evaluate_test_holdout(
        pipeline=result.candidate_pipeline,
        test_df=test_df,
    )
    assert test_eval.partition_name == "test"
    assert test_eval.metrics.sample_count == len(test_df)
    assert len(test_preds) == len(test_df)


def test_training_determinism(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)

    service1 = TrainingService(split_config=config)
    res1, _ = service1.train_and_evaluate_validation(df, "ver-test", "fake-sha", {"n_estimators": 10})

    service2 = TrainingService(split_config=config)
    res2, _ = service2.train_and_evaluate_validation(df, "ver-test", "fake-sha", {"n_estimators": 10})

    # Exact deterministic metric reproducibility
    assert res1.candidate_val_eval.metrics.mae == res2.candidate_val_eval.metrics.mae
    assert res1.candidate_val_eval.metrics.rmse == res2.candidate_val_eval.metrics.rmse
    assert np.allclose(res1.val_predictions["candidate_predicted"], res2.val_predictions["candidate_predicted"])
