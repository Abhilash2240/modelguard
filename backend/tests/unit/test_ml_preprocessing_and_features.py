from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    extract_temporal_features,
    get_feature_metadata,
    prepare_features,
)
from app.ml.models import build_baseline_pipeline, build_candidate_pipeline
from app.ml.preprocessing import build_preprocessor


def test_feature_metadata():
    metadata = get_feature_metadata()
    assert "temp" in metadata.numerical_features
    assert "hour" in metadata.derived_features
    assert "date_time" in metadata.excluded_columns
    assert metadata.target_column == "traffic_volume"


def test_extract_temporal_features():
    df = pd.DataFrame(
        {
            "date_time": ["2012-10-02 09:00:00", "2012-10-06 18:00:00"],
            "temp": [288.0, 280.0],
        }
    )
    feat_df = extract_temporal_features(df)

    assert feat_df.loc[0, "hour"] == 9
    assert feat_df.loc[0, "day_of_week"] == 1  # Tuesday
    assert feat_df.loc[0, "month"] == 10
    assert feat_df.loc[0, "is_weekend"] == 0

    assert feat_df.loc[1, "hour"] == 18
    assert feat_df.loc[1, "day_of_week"] == 5  # Saturday
    assert feat_df.loc[1, "is_weekend"] == 1


def test_prepare_features_target_isolation(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    X, y = prepare_features(df)

    assert y is not None
    assert len(y) == len(df)
    assert y.name == "traffic_volume"
    # Target column must be excluded from feature matrix X
    assert "traffic_volume" not in X.columns
    # Raw timestamp string must be excluded
    assert "date_time" not in X.columns
    # Derived and raw features must be present
    for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        assert col in X.columns


def test_preprocessing_no_leakage():
    # 1. Create artificial train and test sets with different distributions
    train_data = pd.DataFrame(
        {
            "temp": [200.0, 220.0, 240.0],  # mean = 220.0
            "rain_1h": [0.0, 0.0, 0.0],
            "snow_1h": [0.0, 0.0, 0.0],
            "clouds_all": [10, 20, 30],
            "hour": [8, 9, 10],
            "day_of_week": [1, 2, 3],
            "month": [5, 5, 5],
            "is_weekend": [0, 0, 0],
            "holiday": ["None", "None", "None"],
            "weather_main": ["Clear", "Clear", "Clear"],
            "weather_description": ["sky is clear", "sky is clear", "sky is clear"],
        }
    )
    test_data = pd.DataFrame(
        {
            "temp": [300.0, 320.0, 340.0],  # mean = 320.0 (substantially different)
            "rain_1h": [5.0, 5.0, 5.0],
            "snow_1h": [0.0, 0.0, 0.0],
            "clouds_all": [80, 80, 80],
            "hour": [18, 19, 20],
            "day_of_week": [4, 5, 6],
            "month": [7, 7, 7],
            "is_weekend": [0, 1, 1],
            "holiday": ["None", "None", "None"],
            "weather_main": ["Rain", "Rain", "Rain"],
            "weather_description": ["heavy rain", "heavy rain", "heavy rain"],
        }
    )

    preprocessor = build_preprocessor()
    # Fit strictly on train_data
    preprocessor.fit(train_data)

    # Inspect the fitted StandardScaler inside the numerical transformer pipeline
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    # The fitted mean for 'temp' (index 0) must exactly equal train mean (220.0), NOT test mean (320.0)
    assert scaler.mean_[0] == pytest.approx(220.0)

    # Transform test_data: scaling must use the training mean and standard deviation
    transformed_test = preprocessor.transform(test_data)
    assert transformed_test.shape[0] == 3
    # Check that scaled temp in test reflects (300 - 220) / std, not centered around 0
    assert transformed_test[0, 0] > 0.0


def test_baseline_and_candidate_pipeline_fit():
    X = pd.DataFrame(
        {
            "temp": [280.0, 285.0, 290.0, 295.0, 300.0],
            "rain_1h": [0.0, 0.0, 1.0, 0.0, 0.0],
            "snow_1h": [0.0, 0.0, 0.0, 0.0, 0.0],
            "clouds_all": [10, 20, 50, 80, 90],
            "hour": [8, 9, 10, 11, 12],
            "day_of_week": [0, 1, 2, 3, 4],
            "month": [1, 2, 3, 4, 5],
            "is_weekend": [0, 0, 0, 0, 0],
            "holiday": ["None", "None", "None", "None", "None"],
            "weather_main": ["Clear", "Clear", "Rain", "Clouds", "Clouds"],
            "weather_description": ["clear", "clear", "rain", "clouds", "clouds"],
        }
    )
    y = pd.Series([1000, 2000, 3000, 4000, 5000])

    # 1. Baseline Pipeline
    base_pipe = build_baseline_pipeline()
    base_pipe.fit(X, y)
    base_preds = base_pipe.predict(X)
    # DummyRegressor(strategy='mean') predicts training mean for all rows
    assert np.allclose(base_preds, 3000.0)

    # 2. Candidate Pipeline
    cand_pipe = build_candidate_pipeline(random_state=42, n_estimators=10)
    cand_pipe.fit(X, y)
    cand_preds = cand_pipe.predict(X)
    assert len(cand_preds) == 5
    # Predictions should be reasonable numbers
    assert np.all(cand_preds > 0)
