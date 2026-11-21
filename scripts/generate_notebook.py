"""Script to generate the official Google Colab Jupyter Notebook for ModelGuard Work Package B."""

from pathlib import Path

import nbformat as nbf


def generate_colab_notebook(output_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {
            "name": "metro_traffic_training.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
        },
        "language_info": {
            "name": "python",
        },
    }

    cells = []

    # =========================================================================
    # CELL 1 — Project Introduction (Markdown)
    # =========================================================================
    cell_1_md = """# ModelGuard — Production ML Reliability Control Plane
## Work Package B: Reproducible Training & Experiment Tracking

### 1. Overview
**ModelGuard** is an MLOps reliability control plane that bridges data provenance, contract validation, reproducible model training, release governance, and observable production behavior.

This notebook executes **Work Package B**:
- Evaluates the **UCI Metro Interstate Traffic Volume** benchmark (Dataset ID: 492, DOI: `10.24432/C5X60B`, CC BY 4.0).
- **Task:** Hourly westbound Interstate 94 traffic volume regression.
- **Target:** `traffic_volume` (number of vehicles).
- **Goal:** Compare a simple `DummyRegressor` baseline against a `RandomForestRegressor` candidate inside leak-free scikit-learn Pipelines using strict chronological temporal holdouts.
- **Protocol:** Preprocessing is fitted exclusively on training rows. Model selection is based solely on the validation split. The test partition is evaluated strictly once as the final holdout.
- **Orchestration:** Leverages ModelGuard's reusable core modules (`TrainingService`, `validate_dataset_file`, `calculate_sha256`) to guarantee architectural parity between Colab and production backend environments.

---
"""
    cells.append(nbf.v4.new_markdown_cell(cell_1_md))

    # =========================================================================
    # CELL 2 — Runtime Inspection (Code)
    # =========================================================================
    cell_2_code = """# Cell 2: Runtime & Environment Inspection
import os
import platform
import sys

import psutil

print("=== ModelGuard Runtime Inspection ===")
print(f"Python Version : {sys.version}")
print(f"Platform       : {platform.platform()}")
print(f"Processor      : {platform.processor() or 'Standard CPU'}")
print(f"CPU Cores      : {os.cpu_count()}")

mem = psutil.virtual_memory()
print(f"Total RAM      : {mem.total / (1024**3):.2f} GB")
print(f"Available RAM  : {mem.available / (1024**3):.2f} GB")

# Check GPU availability (optional; CPU is primary for tabular baseline)
try:
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
except ImportError:
    gpu_available = False
    gpu_name = "Torch not installed (CPU execution)"

print(f"GPU Available  : {gpu_available} ({gpu_name})")
print("Environment inspection complete. Pipeline is CPU-first and deterministic.")
"""
    cells.append(nbf.v4.new_code_cell(cell_2_code))

    # =========================================================================
    # CELL 3 — Dependency Installation (Code)
    # =========================================================================
    cell_3_code = """# Cell 3: Dependency Installation
# Install exact compatible versions matching ModelGuard backend specification
# Note: MLflow tracking and notebook serialization are decoupled to Phase 4
!pip install --quiet "pandas>=2.2.0" "numpy>=1.26.0" "scikit-learn>=1.4.0" "pandera[pandas]>=0.20.0" "pydantic>=2.7.0" "pydantic-settings>=2.2.0" "sqlalchemy>=2.0.30" "joblib>=1.3.0"

import joblib
import numpy as np
import pandas as pd
import pandera as pa
import pydantic
import sklearn

print(f"scikit-learn : {sklearn.__version__}")
print(f"pandas       : {pd.__version__}")
print(f"numpy        : {np.__version__}")
print(f"pandera      : {pa.__version__}")
print(f"pydantic     : {pydantic.__version__}")
print(f"joblib       : {joblib.__version__}")
print("Dependencies installed and verified successfully.")
"""
    cells.append(nbf.v4.new_code_cell(cell_3_code))

    # =========================================================================
    # CELL 4 — Repository Setup & Code Access (Code)
    # =========================================================================
    cell_4_code = """# Cell 4: Repository Setup & Code Access
# Configures access to ModelGuard core Python packages in Google Colab.
import sys
from pathlib import Path

# Option A: If package or repository is extracted in Colab runtime
repo_candidates = [
    Path("/content/ModelGuard_Colab_Package"),
    Path("./ModelGuard_Colab_Package"),
    Path("./modelguard_build_pack"),
    Path("."),
]
repo_root = Path(".").resolve()
for cand in repo_candidates:
    if (cand / "backend").exists():
        repo_root = cand.resolve()
        break

backend_path = repo_root / "backend"
if str(backend_path) not in sys.path and backend_path.exists():
    sys.path.insert(0, str(backend_path))
    print(f"Added '{backend_path}' to sys.path.")
elif str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Verify core ModelGuard imports
try:
    from app.domain.contracts.metro_traffic_v1 import MetroTrafficContractV1
    from app.domain.hashing import calculate_sha256
    from app.domain.validation_service import validate_dataset_file
    from app.ml.data_loader import load_validated_dataset
    from app.ml.evaluation import compare_models, evaluate_pipeline
    from app.ml.features import get_feature_metadata, prepare_features
    from app.ml.models import build_baseline_pipeline, build_candidate_pipeline
    from app.ml.preprocessing import build_preprocessor
    from app.ml.splitting import ChronologicalSplitter
    from app.ml.training import TrainingService
    from app.schemas.training import SplitConfig

    verified_primitives = [
        MetroTrafficContractV1,
        calculate_sha256,
        validate_dataset_file,
        load_validated_dataset,
        compare_models,
        evaluate_pipeline,
        get_feature_metadata,
        prepare_features,
        build_baseline_pipeline,
        build_candidate_pipeline,
        build_preprocessor,
        ChronologicalSplitter,
        TrainingService,
        SplitConfig,
    ]
    print(f"Successfully verified {len(verified_primitives)} ModelGuard ML & domain primitives.")
except ImportError as e:
    raise ImportError(
        f"Could not import ModelGuard modules: {e}.\\n"
        "Please ensure the 'modelguard_build_pack' repository folder or 'backend' folder is in Colab working directory."
    )
"""
    cells.append(nbf.v4.new_code_cell(cell_4_code))

    # =========================================================================
    # CELL 5 — Dataset Configuration & Manifest Loading (Code)
    # =========================================================================
    cell_5_code = """# Cell 5: Dataset Manifest & Registration Verification
import json

# Target snapshot path in Colab runtime
dataset_candidates = [
    repo_root / "storage" / "datasets" / "metro_interstate_traffic_volume_v1.0.1_dad82c80.csv",
    Path("/content/ModelGuard_Colab_Package/storage/datasets/metro_interstate_traffic_volume_v1.0.1_dad82c80.csv"),
    Path("/content/metro_interstate_traffic_volume_v1.0.1_dad82c80.csv"),
    Path("/content/Metro_Interstate_Traffic_Volume_v1.0.1.csv"),
    Path("/content/Metro_Interstate_Traffic_Volume.csv"),
]

DATASET_PATH = None
for candidate in dataset_candidates:
    if candidate.is_file():
        DATASET_PATH = str(candidate.resolve())
        break

if not DATASET_PATH:
    DATASET_PATH = str(repo_root / "storage" / "datasets" / "metro_interstate_traffic_volume_v1.0.1_dad82c80.csv")

# Resolve official ModelGuard registration manifest
manifest_candidates = [
    repo_root / "datasets" / "manifests" / "metro_traffic_manifest.json",
    Path("/content/ModelGuard_Colab_Package/datasets/manifests/metro_traffic_manifest.json"),
    Path("/content/metro_traffic_manifest.json"),
]

manifest_path = manifest_candidates[0]
for m_cand in manifest_candidates:
    if m_cand.is_file():
        manifest_path = m_cand
        break

print("=== Dataset Configuration & Manifest Loading ===")
print(f"Target Snapshot Path : {DATASET_PATH}")
print(f"Registration Manifest: {manifest_path}")

if not manifest_path.is_file():
    raise FileNotFoundError(
        f"CRITICAL: Dataset registration manifest not found at '{manifest_path}'.\\n"
        "A registered dataset manifest is mandatory for reproducible training."
    )

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

print("\\n[MANIFEST ATTRIBUTION & REGISTERED IDENTITY]")
print(f"Dataset Name        : {manifest.get('name')}")
print(f"UCI Dataset ID      : {manifest.get('dataset_id')}")
print(f"Task Type           : {manifest.get('task')}")
print(f"Target Column       : {manifest.get('target')}")
print(f"Expected Rows       : {manifest.get('total_expected_rows')}")
print(f"Version Tag         : {manifest.get('version_tag')}")
print(f"Registered UUID     : {manifest.get('dataset_version_id')}")
print(f"Registered SHA-256  : {manifest.get('sha256')}")

dataset_file = Path(DATASET_PATH)
if not dataset_file.is_file():
    print(f"\\n[NOTICE] Dataset file '{DATASET_PATH}' not found in runtime.")
    print("Please upload 'metro_interstate_traffic_volume_v1.0.1_dad82c80.csv' to Colab to continue.")
"""
    cells.append(nbf.v4.new_code_cell(cell_5_code))

    # =========================================================================
    # CELL 6 — Cryptographic Dataset Integrity Verification & Identity Audit (Code)
    # =========================================================================
    cell_6_code = """# Cell 6: Cryptographic Dataset Integrity Verification & Identity Audit
from app.domain.hashing import calculate_sha256
from app.ml.data_loader import load_validated_dataset

print(\"\"\"[ARCHITECTURAL BOUNDARY: STANDALONE COLAB VS. DATABASE-BACKED EXECUTION]
In full ModelGuard backend deployments:
  `load_validated_dataset(db: Session, dataset_version_id: str)`
  queries the database for the registered `DatasetVersion` and `ValidationRun` records,
  verifies the physical file matches `version.sha256_hash`, and asserts `latest_val_run.status`
  is PASSED or WARNING with zero blocking errors before loading the DataFrame.

In standalone Google Colab execution:
  No live PostgreSQL / SQLite database session is running. To preserve cryptographic integrity
  without mocking or inventing fake DB records, Colab directly executes the identical domain
  verification logic:
  1. Verifies that the manifest contains a genuine registered dataset version UUID (not a version tag).
  2. `calculate_sha256()` from `app.domain.hashing` ensures the physical file matches `manifest['sha256']`.
  3. Verifies expected row count matches `manifest['total_expected_rows']`.
  4. `validate_dataset_file()` in Cell 7 runs the canonical Work Package A contract validation.
\"\"\")
print(f"Backend loader reference: {load_validated_dataset.__name__}(db, dataset_version_id)")

# --- INTEGRITY CHECKS ---
# 1. Verify registered dataset version UUID is present and distinct from human-readable tag
dataset_version_id = manifest.get("dataset_version_id")
version_tag = manifest.get("version_tag")

if not dataset_version_id:
    raise ValueError(
        "CRITICAL INTEGRITY FAILURE: 'dataset_version_id' is missing from registration manifest!\\n"
        "Training cannot proceed without an immutable database-registered dataset version UUID.\\n"
        "Ensure the dataset version has been registered in ModelGuard before Colab execution."
    )

if dataset_version_id == version_tag or dataset_version_id in ["v1.0.0", "v1.0.1"]:
    raise ValueError(
        f"CRITICAL INTEGRITY FAILURE: 'dataset_version_id' ('{dataset_version_id}') equals version tag!\\n"
        "The database version ID must be a distinct registered UUID, not the human-readable version tag."
    )

# 2. Verify dataset identity metadata
if manifest.get("name") != "metro_interstate_traffic_volume" or manifest.get("dataset_id") != 492:
    raise ValueError(
        f"CRITICAL INTEGRITY FAILURE: Dataset name/ID mismatch! "
        f"Expected 'metro_interstate_traffic_volume' (492), found '{manifest.get('name')}' ({manifest.get('dataset_id')})."
    )

if version_tag != "v1.0.1":
    raise ValueError(f"CRITICAL INTEGRITY FAILURE: Expected version tag 'v1.0.1', found '{version_tag}'.")

expected_sha256 = manifest.get("sha256")
if not expected_sha256:
    raise ValueError("CRITICAL INTEGRITY FAILURE: 'sha256' checksum is missing from registration manifest!")

if not dataset_file.is_file():
    raise FileNotFoundError(
        f"Dataset snapshot file '{DATASET_PATH}' does not exist. Cannot verify physical integrity.\\n"
        "Upload the official registered CSV snapshot to proceed."
    )

# 3. Calculate streaming cryptographic hash
actual_sha256 = calculate_sha256(dataset_file)
file_size_bytes = dataset_file.stat().st_size

print("\\n=== Physical Integrity Verification Report ===")
print(f"File Path        : {dataset_file.resolve()}")
print(f"File Size        : {file_size_bytes:,} bytes ({file_size_bytes / (1024**2):.2f} MB)")
print(f"Expected SHA-256 : {expected_sha256}")
print(f"Actual SHA-256   : {actual_sha256}")

# Anti-tampering check: verify checksum before proceeding
if actual_sha256.lower() != expected_sha256.lower():
    raise ValueError(
        f"CRITICAL: Dataset SHA-256 integrity mismatch!\\n"
        f"Expected: {expected_sha256}\\nActual:   {actual_sha256}\\n"
        "Training aborted. Never train on unverified or tampered data."
    )

# 4. Verify row count against registered manifest
raw_df = pd.read_csv(dataset_file)
actual_rows = len(raw_df)
expected_rows = manifest.get("total_expected_rows", 48204)
print(f"Expected Rows    : {expected_rows:,}")
print(f"Actual Rows      : {actual_rows:,}")

if actual_rows != expected_rows:
    raise ValueError(
        f"CRITICAL INTEGRITY FAILURE: Row count mismatch! Expected {expected_rows:,} rows, found {actual_rows:,}."
    )

integrity_verified = True
print("\\n[PASSED] Dataset snapshot checksum, registered UUID, and row count verified. Provenance is intact.")
"""
    cells.append(nbf.v4.new_code_cell(cell_6_code))

    # =========================================================================
    # CELL 7 — Work Package A Validation Workflow (Code)
    # =========================================================================
    cell_7_code = """# Cell 7: Work Package A Contract Validation Workflow
# Executes the official ModelGuard domain validation service
from app.domain.validation_service import validate_dataset_file

print(\"\"\"[VALIDATION WORKFLOW: DOMAIN SERVICE INTEGRATION]
In the ModelGuard backend:
  `DatasetService.validate_dataset_version()` calls `validate_dataset_file()`,
  persisting the resulting `ValidationSummary` and findings to the database.
In standalone Colab:
  `validate_dataset_file()` runs the exact same domain contract logic directly on disk,
  producing the identical `ValidationSummary` schema without requiring database persistence.
\"\"\")

# Execute domain validation service
val_summary = validate_dataset_file(
    file_path=dataset_file,
    dataset_name=manifest["name"],
    dataset_version_id=dataset_version_id,
    contract_version="v1",
)

print("=== ModelGuard Data Contract Validation Report ===")
print(f"Contract Version: {val_summary.contract_version}")
print(f"Status          : {val_summary.status}")
print(f"Total Records   : {val_summary.total_records:,}")
print(f"Blocking Errors : {val_summary.error_count}")
print(f"Warnings        : {val_summary.warning_count}")

# Categorize findings: BLOCKING vs WARNING
blocking_findings = [f for f in val_summary.findings if f.severity == "BLOCKING"]
warning_findings = [f for f in val_summary.findings if f.severity == "WARNING"]

if warning_findings:
    print(f"\\n[WARNING] {len(warning_findings)} non-blocking findings observed (informative only):")
    for w in warning_findings:
        print(f" - [{w.severity}] Column '{w.column}': check '{w.check}' ({w.failure_count} failures)")

if blocking_findings:
    print(f"\\n[CRITICAL] {len(blocking_findings)} BLOCKING findings encountered:")
    for b in blocking_findings:
        print(f" - [{b.severity}] Column '{b.column}': check '{b.check}' ({b.failure_count} failures)")

# Halt execution only if there are blocking errors or status is FAILED
if val_summary.status == "FAILED" or val_summary.error_count > 0 or len(blocking_findings) > 0:
    raise ValueError(
        f"CRITICAL: Dataset contract validation FAILED with {val_summary.error_count} blocking errors. "
        "Training is strictly prohibited on invalid data."
    )

print("\\n[PASSED] Dataset passed contract validation. Authorized to proceed to training.")
"""
    cells.append(nbf.v4.new_code_cell(cell_7_code))

    # =========================================================================
    # CELL 8 — Dataset Exploration (Code)
    # =========================================================================
    cell_8_code = """# Cell 8: Dataset Exploration & Profiling
print("=== Dataset Schema & Profiling ===")
print(raw_df.info())

print("\\n=== Missing Values ===")
print(raw_df.isna().sum())

print("\\n=== Temporal Span ===")
dt_parsed = pd.to_datetime(raw_df["date_time"])
print(f"Earliest Timestamp : {dt_parsed.min()}")
print(f"Latest Timestamp   : {dt_parsed.max()}")
print(f"Total Time Horizon : {dt_parsed.max() - dt_parsed.min()}")
duplicates = raw_df.duplicated(subset=["date_time"]).sum()
print(f"Duplicate Hours    : {duplicates:,}")

print("\\n=== Target Variable (traffic_volume) Summary ===")
print(raw_df["traffic_volume"].describe().round(2))

print("\\n=== Categorical Distribution: weather_main ===")
print(raw_df["weather_main"].value_counts())
"""
    cells.append(nbf.v4.new_code_cell(cell_8_code))

    # =========================================================================
    # CELL 9 — Feature Engineering & Leakage Review (Code)
    # =========================================================================
    cell_9_code = """# Cell 9: Feature Engineering & Target Leakage Prevention
# Uses ModelGuard's canonical feature preparation module.
metadata = get_feature_metadata()

print("=== Feature Selection & Engineering ===")
print(f"Target Column       : {metadata.target_column}")
print(f"Numerical Features  : {metadata.numerical_features}")
print(f"Categorical Features: {metadata.categorical_features}")
print(f"Derived Temporal    : {metadata.derived_features}")
print(f"Excluded Columns    : {metadata.excluded_columns}")

print(\"\"\"
[FEATURE AVAILABILITY & LEAKAGE AUDIT]
- 'hour', 'day_of_week', 'month', 'is_weekend': Extracted causally from 'date_time'. Available at prediction time.
- 'date_time': Dropped to prevent high-cardinality string memorization and temporal index leakage.
- 'traffic_volume': Strictly isolated as prediction target y. Never included in feature matrix X.
- 'weather_main', 'weather_description', 'temp', 'rain_1h', 'snow_1h': Weather attributes. In operational
  inference, these represent hourly forecasts rather than post-event measurements.
\"\"\")

X, y = prepare_features(raw_df, target_col=metadata.target_column)
print(f"Feature matrix shape X: {X.shape}")
print(f"Target vector shape   y: {y.shape}")
"""
    cells.append(nbf.v4.new_code_cell(cell_9_code))

    # =========================================================================
    # CELL 10 — Chronological Split Configuration & TrainingService (Code)
    # =========================================================================
    cell_10_code = """# Cell 10: Chronological Split Configuration & TrainingService Initialization
# ModelGuard mandates deterministic temporal holdouts to prevent future-to-past data leakage.
from app.ml.training import TrainingService
from app.schemas.training import SplitConfig

split_config = SplitConfig(
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    min_observations=100,
    timestamp_column="date_time",
    target_column="traffic_volume",
)

# Initialize TrainingService with configured temporal splitting parameters
training_service = TrainingService(split_config=split_config)

print("=== Chronological Split Configuration ===")
print(f"Train Ratio         : {split_config.train_ratio:.0%}")
print(f"Validation Ratio    : {split_config.val_ratio:.0%}")
print(f"Test Ratio          : {split_config.test_ratio:.0%}")
print(f"Random State / Seed : {split_config.seed}")
print(f"Timestamp Column    : {split_config.timestamp_column}")
print(f"Target Column       : {split_config.target_column}")
print(f"Min Observations    : {split_config.min_observations}")
print("\\nTrainingService initialized. Preprocessing and models will be fitted exclusively on Training.")
"""
    cells.append(nbf.v4.new_code_cell(cell_10_code))

    # =========================================================================
    # CELL 11 — Preprocessing Pipeline Construction (Code)
    # =========================================================================
    cell_11_code = """# Cell 11: Preprocessing Pipeline Construction & Leakage Audit
# Inspect the scikit-learn ColumnTransformer architecture embedded inside TrainingService.
preprocessor = build_preprocessor()
print("=== ColumnTransformer Architecture ===")
for name, trans, cols in preprocessor.transformers:
    print(f" - Transformer [{name.upper()}]: {type(trans).__name__} applied to {cols}")

print(\"\"\"
[PREPROCESSING HYGIENE & ANTI-LEAKAGE CONTROLS]
1. Numerical Features: SimpleImputer(strategy='median') + StandardScaler()
2. Categorical Features: SimpleImputer(strategy='constant') + OneHotEncoder(handle_unknown='ignore')
3. Inside TrainingService, this ColumnTransformer is encapsulated within a complete scikit-learn Pipeline.
4. .fit() is called strictly on the Training partition (X_train, y_train).
5. Validation and Test partitions are only passed to .transform() during evaluation.
6. Zero statistics (e.g. median, mean, variance) are learned from validation or test holdouts.
\"\"\")
"""
    cells.append(nbf.v4.new_code_cell(cell_11_code))

    # =========================================================================
    # CELL 12 — Baseline Model Architecture (Code)
    # =========================================================================
    cell_12_code = """# Cell 12: Baseline Model Architecture (DummyRegressor)
# Inspect the empirical baseline pipeline used to establish a performance floor.
baseline_pipe = build_baseline_pipeline()
print("=== Baseline Pipeline Architecture ===")
for step_name, step_obj in baseline_pipe.named_steps.items():
    print(f" - Step '{step_name}': {type(step_obj).__name__} ({step_obj})")

print(\"\"\"
[BASELINE DESIGN & GOVERNANCE ROLE]
- Model: DummyRegressor(strategy='mean')
- Role: Represents the simplest naive predictor (predicts training target mean for all inputs).
- Governance rule: A candidate ML model must statistically outperform this baseline on validation
  metrics (MAE, RMSE, R²) to qualify for promotion and test holdout evaluation.
- TrainingService orchestrates baseline training alongside candidate training automatically.
\"\"\")
"""
    cells.append(nbf.v4.new_code_cell(cell_12_code))

    # =========================================================================
    # CELL 13 — Candidate Configuration & Training via TrainingService (Code)
    # =========================================================================
    cell_13_code = """# Cell 13: Candidate Configuration & Training Execution via TrainingService
# Executes chronological splitting, feature preparation, baseline fitting, candidate fitting,
# and validation evaluation in a single reproducible, leak-free workflow.

candidate_hyperparams = {
    "n_estimators": 50,
    "max_depth": 12,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

print("=== Candidate Model Configuration ===")
for k, v in candidate_hyperparams.items():
    print(f" - {k:20s}: {v}")

print("\\nExecuting TrainingService.train_and_evaluate_validation()...")
print(f"Using registered dataset_version_id: '{dataset_version_id}'")

experiment_result, test_df = training_service.train_and_evaluate_validation(
    df=raw_df,
    dataset_version_id=dataset_version_id,
    dataset_sha256=actual_sha256,
    candidate_hyperparameters=candidate_hyperparams,
)

print("\\n=== Chronological Split Manifest ===")
manifest_split = experiment_result.split_manifest
print(f"Total Observations  : {manifest_split.total_rows:,}")
for part_name, part in manifest_split.partitions.items():
    print(f" - [{part_name.upper():10s}] Rows: {part.row_count:6,} | Bounds: {part.min_timestamp} -> {part.max_timestamp}")
    print(f"              Fingerprint: {part.fingerprint_sha256[:16]}...")

print("\\nBoundary collision adjustments:")
for adj in manifest_split.boundary_adjustments:
    print(f" - Boundary '{adj.boundary}': adjusted={adj.adjusted}, shifted={adj.shifted_records} ({adj.action})")

print(f"\\nUntouched test holdout preserved: {len(test_df):,} observations.")
print("[SUCCESS] Baseline and Candidate pipelines successfully trained and evaluated on validation.")
"""
    cells.append(nbf.v4.new_code_cell(cell_13_code))

    # =========================================================================
    # CELL 14 — Validation Evaluation & Model Comparison (Code)
    # =========================================================================
    cell_14_code = """# Cell 14: Validation Evaluation & Model Comparison
# Inspects validation metrics generated by TrainingService
b_val_eval = experiment_result.baseline_val_eval
c_val_eval = experiment_result.candidate_val_eval
comparison = experiment_result.comparison

print("=== Validation Set Metric Comparison ===")
print(f"Sample Count     : {b_val_eval.metrics.sample_count:,} observations\\n")
print(f"{'Metric':<18} | {'Baseline (Dummy)':<18} | {'Candidate (RF)':<18} | {'Delta':<12}")
print("-" * 75)
print(f"{'MAE (vehicles)':<18} | {b_val_eval.metrics.mae:<18.2f} | {c_val_eval.metrics.mae:<18.2f} | {comparison.delta_mae:<12.2f}")
print(f"{'RMSE (vehicles)':<18} | {b_val_eval.metrics.rmse:<18.2f} | {c_val_eval.metrics.rmse:<18.2f} | {comparison.delta_rmse:<12.2f}")
print(f"{'R² Coefficient':<18} | {b_val_eval.metrics.r2:<18.4f} | {c_val_eval.metrics.r2:<18.4f} | {comparison.delta_r2:<+12.4f}")

print("\\n=== Validation Residual Diagnostics (Candidate) ===")
res = c_val_eval.residuals
print(f"Mean Residual    : {res.mean:.2f}")
print(f"Std Residual     : {res.std:.2f}")
print(f"Min / Max        : [{res.min:.2f}, {res.max:.2f}]")
print(f"Percentiles (25, 50, 75, 95): [{res.p25:.1f}, {res.p50:.1f}, {res.p75:.1f}, {res.p95:.1f}]")
"""
    cells.append(nbf.v4.new_code_cell(cell_14_code))

    # =========================================================================
    # CELL 15 — Model Selection Decision (Code)
    # =========================================================================
    cell_15_code = """# Cell 15: Model Selection Decision Gate
# Enforces ModelGuard governance: candidate must be strictly superior on validation
print("=== Model Selection Gate ===")
if comparison.candidate_superior:
    selected_model_name = "candidate_random_forest"
    selected_pipeline = experiment_result.candidate_pipeline
    print("[SELECTED] Candidate Random Forest improves upon Baseline!")
    print(f"  MAE reduction  : {abs(comparison.delta_mae):.2f} vehicles")
    print(f"  RMSE reduction : {abs(comparison.delta_rmse):.2f} vehicles")
    print(f"  R² gain        : {comparison.delta_r2:+.4f}")
else:
    selected_model_name = "baseline_dummy"
    selected_pipeline = experiment_result.baseline_pipeline
    print("[WARNING] Candidate failed to outperform Baseline. Selecting Baseline to prevent regression.")

print(f"Active Selected Model for Staging / Test Evaluation: '{selected_model_name}'")
"""
    cells.append(nbf.v4.new_code_cell(cell_15_code))

    # =========================================================================
    # CELL 16 — Final Test Evaluation via TrainingService (Code)
    # =========================================================================
    cell_16_code = """# Cell 16: Final Test Evaluation via TrainingService (Untouched Holdout)
# Uses TrainingService.evaluate_test_holdout() to evaluate the selected model on the final holdout.
# The test partition was NOT used for training, feature selection, or model selection.
test_eval, test_predictions_df = TrainingService.evaluate_test_holdout(
    pipeline=selected_pipeline,
    test_df=test_df,
    target_col=metadata.target_column,
    model_name=selected_model_name,
)

print("=== Final Untouched Test Holdout Evaluation ===")
print(f"Model Evaluated : {test_eval.model_name}")
print(f"Holdout Samples : {test_eval.metrics.sample_count:,} observations")
print(f"Test MAE        : {test_eval.metrics.mae:.2f} vehicles")
print(f"Test RMSE       : {test_eval.metrics.rmse:.2f} vehicles")
print(f"Test R²         : {test_eval.metrics.r2:.4f}")

print("\\n=== Test Residual Distribution Summary (y_test - y_pred) ===")
res = test_eval.residuals
print(f"Mean Residual   : {res.mean:.2f}")
print(f"Std Residual    : {res.std:.2f}")
print(f"Min / Max       : [{res.min:.2f}, {res.max:.2f}]")
print(f"Percentiles (25, 50, 75, 95): [{res.p25:.1f}, {res.p50:.1f}, {res.p75:.1f}, {res.p95:.1f}]")
"""
    cells.append(nbf.v4.new_code_cell(cell_16_code))

    # =========================================================================
    # CELL 17 — Results Summary Table (Code)
    # =========================================================================
    cell_17_code = """# Cell 17: Results Summary Table
summary_data = {
    "Partition": ["Validation", "Validation", "Test (Holdout)"],
    "Model": ["Baseline (Dummy)", "Candidate (Random Forest)", f"Selected ({selected_model_name})"],
    "Observations": [b_val_eval.metrics.sample_count, c_val_eval.metrics.sample_count, test_eval.metrics.sample_count],
    "MAE": [b_val_eval.metrics.mae, c_val_eval.metrics.mae, test_eval.metrics.mae],
    "RMSE": [b_val_eval.metrics.rmse, c_val_eval.metrics.rmse, test_eval.metrics.rmse],
    "R²": [b_val_eval.metrics.r2, c_val_eval.metrics.r2, test_eval.metrics.r2],
}

summary_df = pd.DataFrame(summary_data)
print("=== Experiment Summary Table ===")
print(summary_df.to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(cell_17_code))

    # =========================================================================
    # CELL 18 — Export & Reproducibility Bundle (Code)
    # =========================================================================
    cell_18_code = """# Cell 18: Export & Complete Reproducibility Bundle
# Compiles all 18 provenance and artifact items required by ModelGuard governance.
import json
from datetime import UTC, datetime

output_dir = repo_root / "experiments" / "outputs" / "colab_run"
output_dir.mkdir(parents=True, exist_ok=True)

# 1 & 2. Serialize model artifacts
selected_model_path = output_dir / f"{selected_model_name}.joblib"
baseline_model_path = output_dir / "baseline_dummy.joblib"
joblib.dump(selected_pipeline, selected_model_path)
joblib.dump(experiment_result.baseline_pipeline, baseline_model_path)

# 14 & 15. Export predictions
val_preds_path = output_dir / "validation_predictions.csv"
test_preds_path = output_dir / "test_predictions.csv"
experiment_result.val_predictions.to_csv(val_preds_path, index=False)
test_predictions_df.to_csv(test_preds_path, index=False)

# Compile complete 18-part reproducibility bundle
reproducibility_bundle = {
    "selected_model_artifact": str(selected_model_path.name),
    "baseline_model_artifact": str(baseline_model_path.name),
    "training_configuration": {
        "split_config": split_config.model_dump(),
        "candidate_hyperparameters": candidate_hyperparams,
    },
    "dataset_manifest": manifest,
    "dataset_version_id": dataset_version_id,
    "dataset_version_tag": version_tag,
    "dataset_sha256": actual_sha256,
    "split_manifest": manifest_split.model_dump(),
    "validation_summary": val_summary.model_dump(),
    "baseline_validation_metrics": b_val_eval.metrics.model_dump(),
    "candidate_validation_metrics": c_val_eval.metrics.model_dump(),
    "model_comparison": comparison.model_dump(),
    "final_test_metrics": test_eval.metrics.model_dump(),
    "validation_predictions_summary": {
        "record_count": len(experiment_result.val_predictions),
        "file_name": val_preds_path.name,
    },
    "test_predictions_summary": {
        "record_count": len(test_predictions_df),
        "file_name": test_preds_path.name,
    },
    "feature_metadata": metadata.model_dump(),
    "python_and_dependency_provenance": {
        "python_version": sys.version,
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "pandera_version": pa.__version__,
        "pydantic_version": pydantic.__version__,
    },
    "execution_status": "COMPLETED",
    "timestamp_utc": datetime.now(UTC).isoformat(),
    "provenance_boundaries": {
        "registered_dataset_identity": dataset_version_id,
        "actual_dataset_checksum": actual_sha256,
        "colab_local_validation_status": val_summary.status,
        "persisted_database_validation_evidence": "STANDALONE_EVALUATED (DB session decoupled in Colab runtime)",
        "integrity_verified": integrity_verified,
    },
}

bundle_path = output_dir / "reproducibility_bundle.json"
with bundle_path.open("w", encoding="utf-8") as f:
    json.dump(reproducibility_bundle, f, indent=2, default=str)

print("=== REPRODUCIBILITY BUNDLE EXPORTED ===")
print(f"Selected Model   : {selected_model_path}")
print(f"Baseline Model   : {baseline_model_path}")
print(f"Val Predictions  : {val_preds_path} ({len(experiment_result.val_predictions):,} rows)")
print(f"Test Predictions : {test_preds_path} ({len(test_predictions_df):,} rows)")
print(f"Bundle Manifest  : {bundle_path}")
print(f"Integrity Status : VERIFIED (integrity_verified={integrity_verified})")
print("\\n[SUCCESS] ModelGuard Work Package B Phase 3 execution and export completed successfully.")
"""
    cells.append(nbf.v4.new_code_cell(cell_18_code))

    nb.cells = cells

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook with {len(cells)} cells at {output_path}")


if __name__ == "__main__":
    out = Path("experiments/notebooks/metro_traffic_training.ipynb")
    generate_colab_notebook(out)
