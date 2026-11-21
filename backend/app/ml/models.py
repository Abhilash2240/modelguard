from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import build_preprocessor


def build_baseline_pipeline() -> Pipeline:
    """Construct the baseline ModelGuard pipeline using DummyRegressor (mean strategy).

    Ensures preprocessor is chained together inside the pipeline so that
    fitting occurs exclusively on training data.
    """
    preprocessor = build_preprocessor()
    model = DummyRegressor(strategy="mean")
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def build_candidate_pipeline(
    random_state: int = 42,
    n_estimators: int = 50,
    max_depth: int | None = 12,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    **kwargs: Any,
) -> Pipeline:
    """Construct the candidate ModelGuard pipeline using RandomForestRegressor.

    CPU-compatible, deterministic configuration with bounded trees for fast, reliable local training.
    """
    preprocessor = build_preprocessor()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
