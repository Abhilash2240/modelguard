import pandas as pd

from app.schemas.evaluation import FeatureMetadata

# Canonical feature lists for UCI Metro Interstate Traffic Volume
NUMERICAL_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "holiday",
    "weather_main",
    "weather_description",
]

TARGET_COLUMN = "traffic_volume"
TIMESTAMP_COLUMN = "date_time"
EXCLUDED_COLUMNS = ["date_time"]  # Raw timestamp string is replaced with extracted components


def get_feature_metadata() -> FeatureMetadata:
    """Return explicit metadata regarding feature selection and exclusions."""
    return FeatureMetadata(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target_column=TARGET_COLUMN,
        excluded_columns=EXCLUDED_COLUMNS,
        derived_features=["hour", "day_of_week", "month", "is_weekend"],
    )


def extract_temporal_features(df: pd.DataFrame, ts_col: str = TIMESTAMP_COLUMN) -> pd.DataFrame:
    """Extract causal calendar and time features available at prediction time from the timestamp.

    Derived features:
    - hour (0-23): Capture diurnal rush hour peaks
    - day_of_week (0-6): Monday=0 through Sunday=6
    - month (1-12): Seasonal patterns
    - is_weekend (0 or 1): Traffic pattern shifts on weekends
    """
    df_out = df.copy()
    if ts_col in df_out.columns:
        dt_series = pd.to_datetime(df_out[ts_col])
        df_out["hour"] = dt_series.dt.hour
        df_out["day_of_week"] = dt_series.dt.day_of_week
        df_out["month"] = dt_series.dt.month
        df_out["is_weekend"] = (dt_series.dt.day_of_week >= 5).astype(int)
    return df_out


def prepare_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    ts_col: str = TIMESTAMP_COLUMN,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Prepare features for model training or inference with strict target leakage prevention.

    1. Extracts causal datetime features (hour, day_of_week, month, is_weekend).
    2. Drops raw timestamp string to prevent categorical explosion.
    3. Isolates target variable into y if present.
    4. Guarantees column order matches canonical feature definitions.
    """
    df_feat = extract_temporal_features(df, ts_col=ts_col)

    y: pd.Series | None = None
    if target_col in df_feat.columns:
        y = df_feat[target_col].copy()
        df_feat = df_feat.drop(columns=[target_col])

    # Drop raw timestamp string if present
    if ts_col in df_feat.columns:
        df_feat = df_feat.drop(columns=[ts_col])

    all_expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    # Ensure all expected columns exist (fill missing with None/NaN if needed)
    for col in all_expected_features:
        if col not in df_feat.columns:
            df_feat[col] = None

    # Reorder columns strictly according to feature definitions
    X = df_feat[all_expected_features].copy()
    return X, y
