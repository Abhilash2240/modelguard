# ModelGuard — MLflow Experiment Tracking & Reproduction Guide

This guide describes how experiment tracking is integrated into the ModelGuard Production ML Reliability Control Plane (Work Package B, Phase 4).

---

## 1. Architecture & Design Principles

ModelGuard separates governance and control plane authority from experiment telemetry:

1. **Authoritative Control Plane**: The ModelGuard relational database (SQLAlchemy) is the single source of truth for dataset snapshots, data contracts, validation status, job queues, and immutable audit trails.
2. **Observability & Experiment Telemetry**: MLflow serves as the experiment tracking store, capturing model artifacts, hyperparameters, evaluation metrics, and prediction tables.
3. **1:1 Strict Mapping**:
   - Exactly one `TrainingJob` maps to exactly one `TrainingRun` in the database.
   - Exactly one `TrainingRun` maps to exactly one MLflow run.
   - Retries (under Option A) create a **new `TrainingJob`**, a **new `TrainingRun`**, and a **new MLflow run**, linked via the `audit_events` table.
4. **Decoupled Dual-Model Artifacts**:
   - Within the single MLflow run, both the `candidate_model` (Random Forest Regressor) and `baseline_model` (Dummy Regressor) scikit-learn pipelines are logged as separate named artifacts.
   - This enables direct comparative loading and evaluation from a single run context without nested runs.

---

## 2. Local Storage Layout

All tracking databases and artifacts reside inside the dedicated, git-ignored `storage/mlflow/` directory:

```text
storage/
├── datasets/                           # Immutable raw & cleaned CSV snapshots
│   ├── metro_interstate_traffic_volume_v1.0.0_e9c3e98b.csv
│   └── metro_interstate_traffic_volume_v1.0.1_dad82c80.csv
└── mlflow/                             # Dedicated MLflow root
    ├── mlflow.db                       # Local SQLite metadata store
    └── artifacts/                      # File-based artifact repository
        └── <experiment-id>/
            └── <mlflow-run-id>/
                └── artifacts/
                    ├── candidate_model/      # MLmodel, model.joblib, conda.yaml
                    ├── baseline_model/       # MLmodel, model.joblib, conda.yaml
                    ├── manifests/            # split_manifest.json, feature_metadata.json
                    ├── predictions/          # validation_predictions.csv, test_predictions.csv
                    └── evaluations/          # comparison_summary.json
```

---

## 3. Launching the MLflow UI

To inspect experiments, comparison metrics, and artifacts via the interactive MLflow web UI:

```powershell
# From the repository root (using the project virtual environment)
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///storage/mlflow/mlflow.db --port 5000
```

Open your browser to:
`http://localhost:5000`

---

## 4. Parameter & Metric Naming Standards

ModelGuard enforces unambiguous key naming to eliminate collisions between candidate and baseline models:

### Parameters
| Key | Type | Description |
| :--- | :--- | :--- |
| `candidate_n_estimators` | `int` | Number of trees in the Random Forest candidate |
| `candidate_max_depth` | `int` | Maximum depth of trees |
| `candidate_min_samples_split` | `int` | Minimum samples to split an internal node |
| `candidate_random_state` | `int` | Random seed for candidate estimator |
| `baseline_strategy` | `string` | Baseline prediction strategy (default: `"mean"`) |
| `split_train_ratio` | `float` | Chronological split ratio for training partition (`0.70`) |
| `split_val_ratio` | `float` | Chronological split ratio for validation partition (`0.15`) |
| `split_test_ratio` | `float` | Chronological split ratio for test holdout partition (`0.15`) |
| `split_seed` | `int` | Deterministic seed for reproducible boundary adjustments (`42`) |
| `target_column` | `string` | Prediction target column (`traffic_volume`) |
| `timestamp_column` | `string` | Chronological timestamp column (`date_time`) |

### Metrics
| Key | Partition | Description |
| :--- | :--- | :--- |
| `candidate_val_mae` | Validation | Mean Absolute Error of candidate on validation split |
| `candidate_val_rmse` | Validation | Root Mean Squared Error of candidate on validation split |
| `candidate_val_r2` | Validation | $R^2$ coefficient of determination on validation split |
| `baseline_val_mae` | Validation | MAE of dummy baseline on validation split |
| `comparison_delta_mae` | Validation | `candidate_val_mae - baseline_val_mae` (negative indicates improvement) |
| `comparison_mae_improvement_pct` | Validation | Relative percentage improvement over baseline |
| `candidate_test_mae` | Test Holdout | MAE on the final untouched test partition |
| `candidate_test_rmse` | Test Holdout | RMSE on the final untouched test partition |
| `candidate_test_r2` | Test Holdout | $R^2$ on the final untouched test partition |

### Provenance Tags
| Tag Key | Example Value | Purpose |
| :--- | :--- | :--- |
| `modelguard.training_job_id` | `5fe9641a-...` | Strict linkage to ModelGuard `training_jobs.id` |
| `modelguard.training_run_id` | `8da56ac6-...` | Strict linkage to ModelGuard `training_runs.id` |
| `modelguard.dataset_version_id` | `8da56ac6-...` | Provenance link to immutable dataset snapshot |
| `modelguard.is_better_than_baseline` | `"true"` | Boolean flag recording if candidate outperformed baseline |
| `modelguard.execution_host` | `"workstation-1"` | Host machine that executed the training job |
| `modelguard.execution_pid` | `"12480"` | OS process PID for liveness detection |

---

## 5. Offline Model Reload & Verification

To reload and evaluate the trained model in downstream scripts or Jupyter notebooks:

```python
import mlflow.sklearn
import pandas as pd
from app.ml.features import prepare_features

# Set tracking URI to local SQLite database
mlflow.set_tracking_uri("sqlite:///storage/mlflow/mlflow.db")

# Target run ID from TrainingRun.mlflow_run_id
run_id = "<your-mlflow-run-id>"

# 1. Load Candidate Random Forest Pipeline
candidate_pipe = mlflow.sklearn.load_model(f"runs:/{run_id}/candidate_model")

# 2. Load Baseline Dummy Pipeline
baseline_pipe = mlflow.sklearn.load_model(f"runs:/{run_id}/baseline_model")

# 3. Predict on new data
new_df = pd.read_csv("storage/datasets/metro_interstate_traffic_volume_v1.0.1_dad82c80.csv")
X, y = prepare_features(new_df.head(10), target_col="traffic_volume")

predictions = candidate_pipe.predict(X)
print("Candidate predictions:", predictions)
```

---

## 6. Execution Lifecycle, Liveness & Orphan Reconciliation

ModelGuard includes autonomous orphan detection and reconciliation in `TrainingJobService`:

1. **Atomic Execution Guard**: `execute_training_job()` transitions `TrainingJob.status` atomically from `QUEUED` to `RUNNING`. Duplicate or simultaneous requests are rejected with `InvalidOperationError`.
2. **Immediate Run ID Persistence**: As soon as MLflow creates a run, `TrainingRun.mlflow_run_id` is persisted to the database. If this commit fails, the MLflow run is safely terminated.
3. **Conservative Liveness Check**: If a process crashes mid-execution (SIGKILL, OOM), `reconcile_orphaned_runs()` verifies the host and PID:
   - If the PID no longer exists on the execution host, the orphaned run is transitioned to `FAILED`.
   - If the host is remote or ambiguous, the run is flagged as `AMBIGUOUS_STALE_RUN` without destructive automatic mutation.
4. **Finalization Recovery**: If training succeeded and database records `SUCCEEDED`, but `mlflow.end_run()` failed due to a network or store hiccup, `reconcile_orphaned_runs()` idempotently updates MLflow status to `FINISHED` with tag `modelguard.reconciliation = "finalized_post_execution"`.
