# ModelGuard — Google Colab Execution Guide
## Work Package B: Reproducible Model Training (Metro Interstate Traffic Volume)

This guide documents how to execute the official ModelGuard training pipeline in **Google Colab** using the generated notebook at `experiments/notebooks/metro_traffic_training.ipynb`.

---

## 1. Opening & Uploading to Google Colab

1. Open your browser and navigate to [Google Colab](https://colab.research.google.com/).
2. In the modal, select the **Upload** tab.
3. Click **Browse** and select `experiments/notebooks/metro_traffic_training.ipynb` from your local ModelGuard repository.
4. Verify that the runtime is using standard Python 3. (GPU is optional; the pipeline is optimized to run deterministically on CPU).

---

## 2. Setting Up Repository Code in Colab

The notebook directly executes ModelGuard's reusable backend modules (`app.ml.*`, `app.domain.*`, `app.schemas.*`).
To provide the repository code in Colab:
- **Option A (GitHub Clone):** In Cell 4, clone your ModelGuard repository:
  ```bash
  !git clone https://github.com/<your-username>/modelguard.git modelguard_build_pack
  ```
- **Option B (Direct Upload):** Zip the `backend` and `datasets` folders locally, upload them to Colab's file explorer, and unzip:
  ```bash
  !unzip -q backend.zip -d ./backend
  !unzip -q datasets.zip -d ./datasets
  ```

---

## 3. Dataset Acquisition, Manifest & Registered Identity

ModelGuard enforces strict dataset provenance. The notebook reads the registration manifest (`datasets/manifests/metro_traffic_manifest.json`) and verifies:
1. **Dataset Identity:** UCI Dataset ID `492`, name `metro_interstate_traffic_volume`.
2. **Version Tag:** Must be `v1.0.1` (cleaned snapshot with 11 sensor anomalies converted to missing).
3. **Database Version UUID (`dataset_version_id`):** Must be a genuine registered database UUID and **strictly distinct** from the human-readable version tag.
4. **Cryptographic Checksum (`sha256`):** The physical snapshot uploaded to Colab must match the registered SHA-256 hash in the manifest.
5. **Expected Rows:** Must match `48,204` rows.

### Transferring to Colab
- **CSV Snapshot:** Upload `Metro_Interstate_Traffic_Volume_v1.0.1.csv` (or `Metro_Interstate_Traffic_Volume.csv`) directly into `/content/` (or update `DATASET_PATH` in Cell 5).
- **Manifest:** Ensure `metro_traffic_manifest.json` is located at `datasets/manifests/metro_traffic_manifest.json` or `/content/metro_traffic_manifest.json`.

---

## 4. Executing Cells in Order

Execute each cell sequentially (**Runtime -> Run all** or `Shift + Enter` cell by cell):

1. **Cell 1 (Introduction):** Outlines product scope, dataset context, regression target, and anti-leakage training protocol.
2. **Cell 2 (Runtime Inspection):** Verifies Python environment, RAM, CPU cores, and optional GPU.
3. **Cell 3 (Dependencies):** Installs pinned scikit-learn, pandera, pandas, numpy, pydantic, sqlalchemy, and joblib.
4. **Cell 4 (Repository Setup):** Inserts `backend/` into `sys.path` and tests ModelGuard imports.
5. **Cell 5 (Dataset Manifest Loading):** Loads `metro_traffic_manifest.json` and parses registered identity metadata.
6. **Cell 6 (Cryptographic Integrity & Identity Audit):**
   - Asserts `dataset_version_id` exists and does not equal `"v1.0.1"`.
   - Computes streaming SHA-256 and asserts equality with `manifest['sha256']`.
   - Verifies row count equals `48,204`.
   - Halts execution immediately if any integrity check fails.
7. **Cell 7 (Contract Validation Workflow):**
   - Calls `validate_dataset_file()` from `app.domain.validation_service` to execute canonical Work Package A contract validation.
   - Differentiates `BLOCKING` vs `WARNING` findings. Non-blocking warnings do not halt execution, but blocking findings do.
8. **Cell 8 (Data Exploration):** Displays shape (48,204 rows, 9 columns), dtypes, missing values, date range, and duplicate timestamp count.
9. **Cell 9 (Feature Engineering):** Extracts causal temporal features (`hour`, `day_of_week`, `month`, `is_weekend`), isolates target `traffic_volume`, and drops raw `date_time`.
10. **Cell 10 (Chronological Split Configuration & TrainingService):** Configures non-overlapping temporal partitions (70% Train, 15% Val, 15% Test) and initializes `TrainingService`.
11. **Cell 11 (Preprocessing Pipeline Architecture):** Inspects `ColumnTransformer` (median imputation + standard scaling for numericals; constant imputation + one-hot encoding for categoricals).
12. **Cell 12 (Baseline Model Architecture):** Inspects `DummyRegressor(strategy="mean")` and reviews governance selection criteria.
13. **Cell 13 (Candidate Configuration & Training via TrainingService):** Passes the registered `dataset_version_id` (not the version tag) to `training_service.train_and_evaluate_validation()`.
14. **Cell 14 (Validation Evaluation & Model Comparison):** Displays validation metrics (MAE, RMSE, $R^2$) and residual diagnostics from `experiment_result`.
15. **Cell 15 (Model Selection Gate):** Evaluates `comparison.candidate_superior` and gates test holdout access.
16. **Cell 16 (Final Test Holdout Evaluation):** Calls `TrainingService.evaluate_test_holdout(selected_pipeline, test_df)` to evaluate the selected model *once* on the untouched holdout.
17. **Cell 17 (Summary Table):** Formats comparison metrics across partitions (Validation Baseline vs. Candidate vs. Test Holdout).
18. **Cell 18 (Export & Complete Reproducibility Bundle):**
   - Serializes model artifacts (`selected_model.joblib`, `baseline_dummy.joblib`).
   - Exports predictions (`validation_predictions.csv`, `test_predictions.csv`).
   - Compiles and exports the 18-part `reproducibility_bundle.json`.

---

## 5. Architectural Boundary: Standalone Colab vs. Database-Backed Backend

| Capability | ModelGuard Backend API | Google Colab Standalone Execution |
|---|---|---|
| **Data Ingestion** | `load_validated_dataset(db, dataset_version_id)` queries SQLite/PostgreSQL metadata. | Reads file from `DATASET_PATH` after verifying `calculate_sha256()` and manifest UUID. |
| **Validation** | `DatasetService.validate_dataset_version()` persists `ValidationRun` to DB. | `validate_dataset_file()` runs exact same contract logic, returns `ValidationSummary`. |
| **Training** | `TrainingService` invoked inside background worker/job. | `TrainingService` invoked directly in notebook runtime. |
| **Model Selection** | Automated gate based on `comparison.candidate_superior`. | Automated gate in Cell 15 based on `comparison.candidate_superior`. |
| **Test Holdout** | `TrainingService.evaluate_test_holdout()` evaluated strictly once. | `TrainingService.evaluate_test_holdout()` evaluated strictly once in Cell 16. |
| **Artifacts** | Stored in S3/MinIO with database registration. | Stored in `experiments/outputs/colab_run/` with `reproducibility_bundle.json`. |

---

## 6. Troubleshooting & Diagnostics

| Symptom | Cause | Solution |
|---|---|---|
| `CRITICAL INTEGRITY FAILURE: 'dataset_version_id' is missing` | The manifest has null/empty version UUID. | Ensure the dataset version has been registered in ModelGuard and its UUID recorded in the manifest. |
| `CRITICAL INTEGRITY FAILURE: 'dataset_version_id' equals version tag` | The version tag `v1.0.1` was passed instead of the database UUID. | Populate the manifest with the distinct database UUID generated during registration. |
| `CRITICAL: Dataset SHA-256 integrity mismatch` | The uploaded CSV does not match the manifest hash. | Confirm you uploaded the exact registered uncompressed CSV (`Metro_Interstate_Traffic_Volume.csv`). |
| `CRITICAL: Dataset contract validation FAILED` | Columns missing or sensor range check violated (e.g. 0 Kelvin temp). | Check Cell 7 findings breakdown. Clean or re-export the official snapshot. |
| `ModuleNotFoundError: No module named 'app'` | The backend repository path is not in `sys.path`. | Ensure Cell 4 is executed and `backend/` directory is present in the Colab working directory. |
