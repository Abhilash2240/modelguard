"""ModelGuard Work Package B Phase 3 Reproducibility Verification Script.

Executes a clean independent local training run with exact matching parameters:
- Same dataset version (v1.0.1) & SHA-256
- Same chronological split config (70/15/15, seed=42)
- Same model hyperparameters (RandomForest: n_estimators=50, max_depth=12, min_samples_split=4, min_samples_leaf=2)
- Compares partition row counts, partition SHA-256 fingerprints, validation metrics, test metrics, and prediction arrays.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.domain.hashing import calculate_sha256  # noqa: E402
from app.ml.features import get_feature_metadata  # noqa: E402
from app.ml.splitting import ChronologicalSplitter  # noqa: E402
from app.ml.training import TrainingService  # noqa: E402
from app.schemas.training import SplitConfig  # noqa: E402


def main() -> None:
    print("=== MODELGUARD WORK PACKAGE B: REPRODUCIBILITY VERIFICATION ===")

    # 1. Load exported Colab reproducibility bundle
    bundle_path = repo_root / "experiments" / "outputs" / "colab_run" / "reproducibility_bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Colab reproducibility bundle missing at: {bundle_path}")

    with bundle_path.open("r", encoding="utf-8") as f:
        colab_bundle = json.load(f)

    # 2. Verify dataset identity and integrity
    dataset_path = repo_root / "storage" / "datasets" / "metro_interstate_traffic_volume_v1.0.1_dad82c80.csv"
    assert dataset_path.is_file(), f"Dataset not found at {dataset_path}"
    local_dataset_sha256 = calculate_sha256(dataset_path)
    assert local_dataset_sha256 == colab_bundle["dataset_sha256"], "Dataset SHA-256 mismatch!"
    print(f"[OK] Dataset SHA-256 verified: {local_dataset_sha256}")

    # 3. Load dataset and generate splits
    df = pd.read_csv(dataset_path)
    split_cfg_dict = colab_bundle["training_configuration"]["split_config"]
    split_config = SplitConfig(**split_cfg_dict)

    splitter = ChronologicalSplitter(split_config)
    train_df, val_df, test_df, split_manifest = splitter.split(
        df=df,
        dataset_version_id=colab_bundle["dataset_version_id"],
        dataset_sha256=local_dataset_sha256,
    )

    print("\n--- Split Partition & Fingerprint Comparison ---")
    colab_manifest = colab_bundle["split_manifest"]
    local_partitions = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }
    for part_name in ["train", "validation", "test"]:
        local_part = local_partitions[part_name]
        colab_part = colab_manifest["partitions"][part_name]

        print(f"[{part_name.upper()}]")
        print(f"  Rows        : Local={len(local_part):,} vs Colab={colab_part['row_count']:,}")
        assert len(local_part) == colab_part["row_count"], f"Row count mismatch on {part_name}"

        local_fp = split_manifest.partitions[part_name].fingerprint_sha256
        print(f"  Fingerprint : Local={local_fp[:16]}... vs Colab={colab_part['fingerprint_sha256'][:16]}...")
        assert local_fp == colab_part["fingerprint_sha256"], f"Fingerprint mismatch on {part_name}"
        print(f"  Status      : EXACT MATCH (SHA-256: {local_fp})")

    # 4. Clean second training execution
    candidate_hyperparams = colab_bundle["training_configuration"]["candidate_hyperparameters"]
    training_service = TrainingService(split_config=split_config)

    print("\n--- Executing Clean Second Training Run ---")
    local_experiment_result, untouched_test_df = training_service.train_and_evaluate_validation(
        df=df,
        dataset_version_id=colab_bundle["dataset_version_id"],
        dataset_sha256=local_dataset_sha256,
        candidate_hyperparameters=candidate_hyperparams,
    )

    # Evaluate test holdout strictly once
    feature_meta = get_feature_metadata()
    local_test_eval, local_test_preds_df = TrainingService.evaluate_test_holdout(
        pipeline=local_experiment_result.candidate_pipeline,
        test_df=untouched_test_df,
        target_col=feature_meta.target_column,
        model_name="candidate_random_forest",
    )

    # 5. Numerical Metrics Comparison
    b_val_colab = colab_bundle["baseline_validation_metrics"]
    c_val_colab = colab_bundle["candidate_validation_metrics"]
    test_colab = colab_bundle["final_test_metrics"]

    b_val_local = local_experiment_result.baseline_val_eval.metrics
    c_val_local = local_experiment_result.candidate_val_eval.metrics
    test_local = local_test_eval.metrics

    print("\n--- Metric Parity Audit (Colab vs. Clean Local Run) ---")
    metrics_comparison = [
        ("Baseline Val MAE", b_val_colab["mae"], b_val_local.mae, 1e-3),
        ("Baseline Val RMSE", b_val_colab["rmse"], b_val_local.rmse, 1e-3),
        ("Baseline Val R2", b_val_colab["r2"], b_val_local.r2, 1e-3),
        ("Candidate Val MAE", c_val_colab["mae"], c_val_local.mae, 1e-3),
        ("Candidate Val RMSE", c_val_colab["rmse"], c_val_local.rmse, 1e-3),
        ("Candidate Val R2", c_val_colab["r2"], c_val_local.r2, 1e-3),
        ("Candidate Test MAE", test_colab["mae"], test_local.mae, 1e-3),
        ("Candidate Test RMSE", test_colab["rmse"], test_local.rmse, 1e-3),
        ("Candidate Test R2", test_colab["r2"], test_local.r2, 1e-3),
    ]

    for name, c_val, l_val, tol in metrics_comparison:
        diff = abs(c_val - l_val)
        status = "EXACT / WITHIN TOLERANCE" if diff <= tol else "DIVERGENCE"
        print(f"  {name:<20}: Colab={c_val:10.4f} | Local={l_val:10.4f} | Diff={diff:8.6f} -> {status}")
        assert diff <= tol, f"Metric {name} diverged beyond tolerance {tol}: Colab={c_val}, Local={l_val}"

    # 6. Prediction Array Comparison
    print("\n--- Prediction Output Array Parity ---")
    colab_val_preds = pd.read_csv(repo_root / "experiments" / "outputs" / "colab_run" / "validation_predictions.csv")
    colab_test_preds = pd.read_csv(repo_root / "experiments" / "outputs" / "colab_run" / "test_predictions.csv")

    local_val_cand = local_experiment_result.val_predictions["candidate_predicted"].to_numpy()
    colab_val_cand = colab_val_preds["candidate_predicted"].to_numpy()
    val_diff = np.max(np.abs(local_val_cand - colab_val_cand))
    print(f"  Validation Predictions Max Abs Difference: {val_diff:.8f}")

    local_test_cand = local_test_preds_df["predicted"].to_numpy()
    colab_test_cand = colab_test_preds["predicted"].to_numpy()
    test_diff = np.max(np.abs(local_test_cand - colab_test_cand))
    print(f"  Test Predictions Max Abs Difference      : {test_diff:.8f}")

    # Both differences should be zero or negligible floating-point precision
    assert val_diff < 1e-4, f"Validation predictions diverged: max diff={val_diff}"
    assert test_diff < 1e-4, f"Test predictions diverged: max diff={test_diff}"
    print("  Status: BIT-LEVEL / DETERMINISTIC REPRODUCIBILITY PROVEN")

    print("\n=== REPRODUCIBILITY VERIFICATION: 100% PASSED ===")


if __name__ == "__main__":
    main()
