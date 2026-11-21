import hashlib

import pandas as pd

from app.core.errors import ModelGuardError
from app.schemas.training import BoundaryAdjustment, PartitionManifest, SplitConfig, SplitManifest


class SplitError(ModelGuardError):
    """Raised when dataset temporal splitting fails or violates boundary rules."""

    pass


class ChronologicalSplitter:
    """Deterministic chronological train/validation/test split generator.

    Guarantees temporal ordering, prevents data leakage across partitions,
    explicitly resolves equal timestamps at boundaries, and generates auditable manifests.
    """

    def __init__(self, config: SplitConfig | None = None):
        self.config = config or SplitConfig()

    @staticmethod
    def compute_partition_fingerprint(df_partition: pd.DataFrame) -> str:
        """Deterministic SHA-256 fingerprint of a partition's exact content."""
        csv_bytes = df_partition.to_csv(index=False, lineterminator="\n").encode("utf-8")
        return hashlib.sha256(csv_bytes).hexdigest()

    def _resolve_boundary_collision(
        self,
        df_sorted: pd.DataFrame,
        cut_idx: int,
        boundary_name: str,
        min_prior: int,
        min_posterior: int,
    ) -> tuple[int, BoundaryAdjustment]:
        """Explicitly resolve equal timestamps at partition cut boundaries.

        If df_sorted[cut_idx - 1] shares the exact same timestamp as df_sorted[cut_idx],
        adjusts the cut point so that the timestamp cluster is not bifurcated across splits.
        """
        n = len(df_sorted)
        if cut_idx <= 0 or cut_idx >= n:
            return cut_idx, BoundaryAdjustment(
                boundary=boundary_name,
                adjusted=False,
                boundary_timestamp=None,
                shifted_records=0,
                action="boundary_at_edge",
            )

        ts_before = df_sorted.iloc[cut_idx - 1]["__dt_sort__"]
        ts_after = df_sorted.iloc[cut_idx]["__dt_sort__"]

        if ts_before != ts_after:
            return cut_idx, BoundaryAdjustment(
                boundary=boundary_name,
                adjusted=False,
                boundary_timestamp=None,
                shifted_records=0,
                action="no_collision_clean_boundary",
            )

        collision_ts = ts_before
        collision_ts_str = str(df_sorted.iloc[cut_idx - 1][self.config.timestamp_column])

        # Find contiguous range of rows with this identical timestamp
        cluster_mask = df_sorted["__dt_sort__"] == collision_ts
        cluster_indices = df_sorted.index[cluster_mask].tolist()
        cluster_start = cluster_indices[0]
        cluster_end = cluster_indices[-1]

        # Strategy: Prefer keeping the entire cluster in the prior partition (shift forward to cluster_end + 1),
        # provided posterior partition has at least min_posterior rows remaining.
        forward_cut = cluster_end + 1
        if (n - forward_cut) >= min_posterior and forward_cut >= min_prior:
            shift = forward_cut - cut_idx
            return forward_cut, BoundaryAdjustment(
                boundary=boundary_name,
                adjusted=True,
                boundary_timestamp=collision_ts_str,
                shifted_records=shift,
                action=f"shifted_cut_forward_by_{shift}_to_keep_cluster_in_prior_partition",
            )

        # Fallback: Shift backward to cluster_start, keeping cluster in posterior partition,
        # provided prior partition has at least min_prior rows.
        backward_cut = cluster_start
        if backward_cut >= min_prior and (n - backward_cut) >= min_posterior:
            shift = cut_idx - backward_cut
            return backward_cut, BoundaryAdjustment(
                boundary=boundary_name,
                adjusted=True,
                boundary_timestamp=collision_ts_str,
                shifted_records=shift,
                action=f"shifted_cut_backward_by_{shift}_to_keep_cluster_in_posterior_partition",
            )

        raise SplitError(
            f"Cannot resolve equal timestamp collision on boundary '{boundary_name}' at timestamp '{collision_ts_str}': "
            f"collision spans {len(cluster_indices)} records, making it impossible to satisfy minimum partition sizes."
        )

    def split(
        self,
        df: pd.DataFrame,
        dataset_version_id: str,
        dataset_sha256: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
        """Perform deterministic chronological split into (train_df, val_df, test_df) and SplitManifest."""
        ts_col = self.config.timestamp_column
        target_col = self.config.target_column

        if ts_col not in df.columns:
            raise SplitError(f"Timestamp column '{ts_col}' not found in dataframe.")
        if target_col not in df.columns:
            raise SplitError(f"Target column '{target_col}' not found in dataframe.")

        total_rows = len(df)
        if total_rows < self.config.min_observations:
            raise SplitError(
                f"Dataframe contains {total_rows} rows, which is fewer than the configured minimum of "
                f"{self.config.min_observations} required observations."
            )

        # Validate null/NaT timestamps
        if df[ts_col].isna().any():
            null_count = int(df[ts_col].isna().sum())
            raise SplitError(f"Found {null_count} null or unparseable timestamps in column '{ts_col}'.")

        # Parse datetime for strict temporal sorting
        try:
            parsed_dt = pd.to_datetime(df[ts_col])
        except Exception as e:
            raise SplitError(f"Failed to parse timestamp column '{ts_col}' as datetime: {e}")

        # Deterministic chronological sort using stable mergesort
        df_sorted = df.copy()
        df_sorted["__dt_sort__"] = parsed_dt
        df_sorted = df_sorted.sort_values(by="__dt_sort__", ascending=True, kind="mergesort").reset_index(drop=True)

        # Initial index calculations (guaranteeing >= 1 row per partition for total_rows >= 3)
        nominal_train_idx = int(total_rows * self.config.train_ratio)
        initial_train_idx = max(1, min(nominal_train_idx, total_rows - 2))

        boundary_adjustments: list[BoundaryAdjustment] = []

        # 1. Resolve train/val boundary
        train_idx, train_adj = self._resolve_boundary_collision(
            df_sorted,
            cut_idx=initial_train_idx,
            boundary_name="train_val",
            min_prior=1,
            min_posterior=2,  # val and test must share remaining
        )
        boundary_adjustments.append(train_adj)

        # 2. Compute and resolve val/test boundary
        nominal_val_idx = int(total_rows * (self.config.train_ratio + self.config.val_ratio))
        initial_val_idx = max(train_idx + 1, min(nominal_val_idx, total_rows - 1))

        val_idx, val_adj = self._resolve_boundary_collision(
            df_sorted,
            cut_idx=initial_val_idx,
            boundary_name="val_test",
            min_prior=train_idx + 1,
            min_posterior=1,
        )
        boundary_adjustments.append(val_adj)

        # Verify sanity of cut indices
        if train_idx <= 0 or train_idx >= val_idx or val_idx >= total_rows:
            raise SplitError(
                f"Invalid partition boundaries after collision resolution: train_idx={train_idx}, "
                f"val_idx={val_idx}, total_rows={total_rows}."
            )

        # Create partitions (drop internal helper column)
        train_df = df_sorted.iloc[:train_idx].drop(columns=["__dt_sort__"]).reset_index(drop=True)
        val_df = df_sorted.iloc[train_idx:val_idx].drop(columns=["__dt_sort__"]).reset_index(drop=True)
        test_df = df_sorted.iloc[val_idx:].drop(columns=["__dt_sort__"]).reset_index(drop=True)

        # Strict temporal non-overlap assertion
        train_max_ts = pd.to_datetime(train_df[ts_col]).max()
        val_min_ts = pd.to_datetime(val_df[ts_col]).min()
        val_max_ts = pd.to_datetime(val_df[ts_col]).max()
        test_min_ts = pd.to_datetime(test_df[ts_col]).min()

        if train_max_ts >= val_min_ts:
            raise SplitError(
                f"Temporal leakage detected between train and validation: train_max ({train_max_ts}) >= val_min ({val_min_ts})"
            )
        if val_max_ts >= test_min_ts:
            raise SplitError(
                f"Temporal leakage detected between validation and test: val_max ({val_max_ts}) >= test_min ({test_min_ts})"
            )

        # Compute partition manifests and fingerprints
        partitions = {
            "train": PartitionManifest(
                name="train",
                row_count=len(train_df),
                min_timestamp=str(train_df[ts_col].min()),
                max_timestamp=str(train_df[ts_col].max()),
                fingerprint_sha256=self.compute_partition_fingerprint(train_df),
            ),
            "validation": PartitionManifest(
                name="validation",
                row_count=len(val_df),
                min_timestamp=str(val_df[ts_col].min()),
                max_timestamp=str(val_df[ts_col].max()),
                fingerprint_sha256=self.compute_partition_fingerprint(val_df),
            ),
            "test": PartitionManifest(
                name="test",
                row_count=len(test_df),
                min_timestamp=str(test_df[ts_col].min()),
                max_timestamp=str(test_df[ts_col].max()),
                fingerprint_sha256=self.compute_partition_fingerprint(test_df),
            ),
        }

        manifest = SplitManifest(
            dataset_version_id=dataset_version_id,
            dataset_sha256=dataset_sha256,
            split_config=self.config,
            total_rows=total_rows,
            partitions=partitions,
            boundary_adjustments=boundary_adjustments,
        )

        return train_df, val_df, test_df, manifest
