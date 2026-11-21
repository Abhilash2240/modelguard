import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from app.schemas.evaluation import ModelComparison, PartitionEvaluation, RegressionMetrics, ResidualSummary


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate MAE, RMSE, and R² regression metrics."""
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot calculate metrics on empty targets.")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # For tiny samples or zero variance, r2_score may be NaN or negative; handle safely
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    return RegressionMetrics(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        r2=round(r2, 4),
        sample_count=n,
    )


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> ResidualSummary:
    """Calculate summary statistics and percentiles for prediction residuals (y_true - y_pred)."""
    residuals = y_true - y_pred
    return ResidualSummary(
        mean=round(float(np.mean(residuals)), 4),
        std=round(float(np.std(residuals)), 4),
        min=round(float(np.min(residuals)), 4),
        max=round(float(np.max(residuals)), 4),
        p25=round(float(np.percentile(residuals, 25)), 4),
        p50=round(float(np.percentile(residuals, 50)), 4),
        p75=round(float(np.percentile(residuals, 75)), 4),
        p95=round(float(np.percentile(residuals, 95)), 4),
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    partition_name: str,
) -> tuple[PartitionEvaluation, np.ndarray]:
    """Evaluate a trained scikit-learn pipeline on a given partition."""
    y_true = np.asarray(y)
    y_pred = pipeline.predict(X)

    metrics = calculate_metrics(y_true, y_pred)
    residuals = calculate_residuals(y_true, y_pred)

    evaluation = PartitionEvaluation(
        model_name=model_name,
        partition_name=partition_name,
        metrics=metrics,
        residuals=residuals,
    )

    return evaluation, y_pred


def compare_models(
    baseline_eval: PartitionEvaluation,
    candidate_eval: PartitionEvaluation,
    partition_name: str = "validation",
) -> ModelComparison:
    """Compare candidate model against baseline on the same evaluation partition.

    delta_mae: candidate - baseline (negative indicates candidate improvement)
    delta_rmse: candidate - baseline (negative indicates candidate improvement)
    delta_r2: candidate - baseline (positive indicates candidate improvement)
    """
    b_metrics = baseline_eval.metrics
    c_metrics = candidate_eval.metrics

    d_mae = round(c_metrics.mae - b_metrics.mae, 4)
    d_rmse = round(c_metrics.rmse - b_metrics.rmse, 4)
    d_r2 = round(c_metrics.r2 - b_metrics.r2, 4)

    # Candidate is considered superior only if it achieves strictly lower MAE and RMSE
    candidate_superior = (d_mae < 0) and (d_rmse < 0)

    return ModelComparison(
        metric_target=partition_name,
        baseline_metrics=b_metrics,
        candidate_metrics=c_metrics,
        delta_mae=d_mae,
        delta_rmse=d_rmse,
        delta_r2=d_r2,
        candidate_superior=candidate_superior,
    )
