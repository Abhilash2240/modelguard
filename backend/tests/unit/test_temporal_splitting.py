from pathlib import Path

import pandas as pd
import pytest
from app.ml.splitting import ChronologicalSplitter, SplitError
from app.schemas.training import SplitConfig
from pydantic import ValidationError


def test_split_config_validation():
    # Valid config
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=10)
    assert config.train_ratio == 0.70

    # Ratios that do not sum to 1.0
    with pytest.raises(ValidationError, match="Split ratios must sum to 1.0"):
        SplitConfig(train_ratio=0.80, val_ratio=0.15, test_ratio=0.15)


def test_temporal_split_clean_data(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    splitter = ChronologicalSplitter(SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5))

    train_df, val_df, test_df, manifest = splitter.split(
        df=df,
        dataset_version_id="dv-1234",
        dataset_sha256="abcde12345",
    )

    # 1. Total row count preservation
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert manifest.total_rows == len(df)
    assert manifest.dataset_sha256 == "abcde12345"

    # 2. Partitions must not be empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # 3. Strict chronological non-overlap
    train_max = pd.to_datetime(train_df["date_time"]).max()
    val_min = pd.to_datetime(val_df["date_time"]).min()
    val_max = pd.to_datetime(val_df["date_time"]).max()
    test_min = pd.to_datetime(test_df["date_time"]).min()

    assert train_max < val_min
    assert val_max < test_min

    # 4. Fingerprint presence
    assert len(manifest.partitions["train"].fingerprint_sha256) == 64
    assert len(manifest.partitions["validation"].fingerprint_sha256) == 64
    assert len(manifest.partitions["test"].fingerprint_sha256) == 64


def test_temporal_split_determinism(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    # Shuffle the input dataframe randomly to ensure sorting is completely deterministic
    df_shuffled = df.sample(frac=1.0, random_state=999).reset_index(drop=True)

    splitter = ChronologicalSplitter(SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5))

    t1, v1, te1, m1 = splitter.split(df, "ver-1", "sha-1")
    t2, v2, te2, m2 = splitter.split(df_shuffled, "ver-1", "sha-1")

    # Outputs must be 100% identical regardless of raw input row order
    assert m1.partitions["train"].fingerprint_sha256 == m2.partitions["train"].fingerprint_sha256
    assert m1.partitions["validation"].fingerprint_sha256 == m2.partitions["validation"].fingerprint_sha256
    assert m1.partitions["test"].fingerprint_sha256 == m2.partitions["test"].fingerprint_sha256
    assert t1.equals(t2)
    assert v1.equals(v2)
    assert te1.equals(te2)


def test_configurable_min_observations():
    df = pd.DataFrame(
        {
            "date_time": ["2026-01-01 10:00:00", "2026-01-01 11:00:00", "2026-01-01 12:00:00"],
            "traffic_volume": [100, 200, 300],
        }
    )
    # Configure min_observations to 10; splitting 3 rows must fail
    splitter = ChronologicalSplitter(SplitConfig(min_observations=10))

    with pytest.raises(SplitError, match="fewer than the configured minimum of 10"):
        splitter.split(df, "ver-1", "sha-1")

    # Lowering min_observations allows splitting small frames
    splitter_low = ChronologicalSplitter(SplitConfig(min_observations=3))
    t, v, te, m = splitter_low.split(df, "ver-1", "sha-1")
    assert len(t) == 1
    assert len(v) == 1
    assert len(te) == 1
    assert len(t) + len(v) + len(te) == 3


def test_null_timestamps_rejected():
    df = pd.DataFrame(
        {
            "date_time": ["2026-01-01 10:00:00", None, "2026-01-01 12:00:00", "2026-01-01 13:00:00"],
            "traffic_volume": [100, 200, 300, 400],
        }
    )
    splitter = ChronologicalSplitter(SplitConfig(min_observations=4))

    with pytest.raises(SplitError, match="Found 1 null or unparseable timestamps"):
        splitter.split(df, "ver-1", "sha-1")


def test_boundary_collision_explicit_handling():
    # Create dataset where rows at cut point share identical timestamp
    # 10 rows: cut at 70% is index 7. Let index 6 and 7 have identical timestamps
    data = {
        "date_time": [
            "2026-01-01 01:00:00",
            "2026-01-01 02:00:00",
            "2026-01-01 03:00:00",
            "2026-01-01 04:00:00",
            "2026-01-01 05:00:00",
            "2026-01-01 06:00:00",
            "2026-01-01 07:00:00",  # index 6
            "2026-01-01 07:00:00",  # index 7 (collision!)
            "2026-01-01 08:00:00",
            "2026-01-01 09:00:00",
        ],
        "traffic_volume": [10, 20, 30, 40, 50, 60, 70, 75, 80, 90],
    }
    df = pd.DataFrame(data)
    splitter = ChronologicalSplitter(SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5))

    train_df, val_df, test_df, manifest = splitter.split(df, "ver-1", "sha-1")

    # Assert collision was resolved and recorded in manifest
    train_adj = next(a for a in manifest.boundary_adjustments if a.boundary == "train_val")
    assert train_adj.adjusted is True
    assert train_adj.boundary_timestamp == "2026-01-01 07:00:00"

    # Both rows with 07:00:00 must be in train; validation must start strictly after 07:00:00
    assert len(train_df[train_df["date_time"] == "2026-01-01 07:00:00"]) == 2
    assert len(val_df[val_df["date_time"] == "2026-01-01 07:00:00"]) == 0

    # Strict temporal boundary check must pass
    train_max = pd.to_datetime(train_df["date_time"]).max()
    val_min = pd.to_datetime(val_df["date_time"]).min()
    assert train_max < val_min
