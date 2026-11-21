# ModelGuard — API Contracts Specification (OpenAPI v1)

Base URL: `http://localhost:8000/api/v1`  
Prometheus Metrics: `http://localhost:8000/metrics`

---

## 1. System & Health

### `GET /api/v1/health`
Return runtime health status and active service dependencies.
- **Response `200 OK`**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-11-21T18:00:00.000000Z",
    "environment": "production",
    "version": "0.1.0"
  }
  ```

---

## 2. Dataset Management & Validation

### `GET /api/v1/datasets`
List registered datasets.
- **Response `200 OK`**: Array of dataset objects.

### `POST /api/v1/datasets`
Register a new logical dataset.
- **Request Body**:
  ```json
  {
    "name": "metro-interstate-traffic",
    "task_type": "regression",
    "source_uri": "https://archive.ics.uci.edu/dataset/492/urban+traffic",
    "licence": "CC BY 4.0",
    "description": "Hourly traffic volume on I-94 westbound"
  }
  ```

### `POST /api/v1/datasets/{dataset_id}/versions`
Upload and register an immutable snapshot.
- **Request**: Multipart Form Data (`file`: `.csv`, `version_tag`: `string`).
- **Response `201 Created`**:
  ```json
  {
    "id": "uuid",
    "dataset_id": "uuid",
    "version_tag": "v1.0.0",
    "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "byte_size": 4053210,
    "row_count": 48204,
    "created_at": "2026-11-21T18:00:00Z"
  }
  ```

### `POST /api/v1/datasets/versions/{version_id}/validate`
Execute Pandera schema contract validation on the stored snapshot.
- **Query Params**: `contract_version` (default: `"v1"`).
- **Response `200 OK`**:
  ```json
  {
    "id": "uuid",
    "dataset_version_id": "uuid",
    "status": "PASSED",
    "error_count": 0,
    "warning_count": 0,
    "details_json": { "checks_passed": 18, "failures": [] },
    "created_at": "2026-11-21T18:00:00Z"
  }
  ```

---

## 3. Training & Experiment Tracking

### `GET /api/v1/training/jobs`
List all training jobs with associated execution runs and MLflow references.

### `POST /api/v1/training/jobs`
Enqueue a deterministic training job.
- **Request Body**:
  ```json
  {
    "dataset_version_id": "uuid",
    "validation_run_id": "uuid",
    "config": {
      "split": {
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42,
        "target_column": "traffic_volume",
        "timestamp_column": "date_time"
      },
      "hyperparameters": {
        "n_estimators": 50,
        "max_depth": 10
      }
    }
  }
  ```

### `POST /api/v1/training/jobs/{job_id}/execute`
Execute training job synchronously through the 15-stage lifecycle.
- **Response `200 OK`**:
  ```json
  {
    "id": "uuid",
    "training_job_id": "uuid",
    "mlflow_run_id": "f8a12bcde3456789",
    "status": "SUCCEEDED",
    "metrics_json": {
      "candidate_val_mae": 284.15,
      "candidate_val_rmse": 450.20,
      "candidate_val_r2": 0.942,
      "baseline_val_mae": 1720.50,
      "improvement_pct": 83.48
    }
  }
  ```

---

## 4. Model Registry & Release Gate

### `GET /api/v1/models/candidates`
List all registered model candidates.

### `POST /api/v1/models/candidates`
Register a candidate from a successful training run.
- **Request Body**: `{"training_run_id": "uuid"}`.

### `POST /api/v1/models/candidates/{candidate_id}/gate-evaluation`
Evaluate candidate metrics against the Release Gate policy.
- **Request Body**:
  ```json
  {
    "policy": {
      "max_val_mae": 350.0,
      "max_val_rmse": 550.0,
      "min_val_r2": 0.90,
      "min_improvement_pct": 50.0
    }
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "passed": true,
    "evaluated_at": "2026-11-21T18:00:00Z",
    "rejection_reasons": []
  }
  ```

---

## 5. Deployment Orchestration & Rollback

### `GET /api/v1/deployments`
List all past and active deployments.

### `POST /api/v1/deployments/stage`
Stage candidate into staging environment and execute automated smoke testing.
- **Request Body**: `{"candidate_id": "uuid"}`.
- **Response `200 OK`**:
  ```json
  {
    "id": "uuid",
    "environment_name": "staging",
    "candidate_id": "uuid",
    "status": "AWAITING_APPROVAL",
    "smoke_test_results_json": {
      "passed": true,
      "latency_ms": 12.4,
      "sample_output_finite": true
    }
  }
  ```

### `POST /api/v1/deployments/{deployment_id}/approve`
Explicit reviewer sign-off.
- **Request Body**:
  ```json
  {
    "actor": "senior_reviewer",
    "rationale": "Model passed release gate and staging smoke tests"
  }
  ```

### `POST /api/v1/deployments/{deployment_id}/promote`
Atomically promote approved deployment to production champion.

### `POST /api/v1/deployments/{deployment_id}/rollback`
Execute audited, zero-downtime rollback to previous stable champion.
- **Request Body**:
  ```json
  {
    "actor": "on_call_sre",
    "reason": "Covariate drift detected in temperature sensor"
  }
  ```

---

## 6. Inference Serving & Ground-Truth Feedback

### `POST /api/v1/inference/predict`
Serve real-time prediction using the active production champion.
- **Request Body**:
  ```json
  {
    "inputs": [
      {
        "holiday": "None",
        "temp": 288.15,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 40,
        "weather_main": "Clouds",
        "weather_description": "scattered clouds",
        "date_time": "2026-11-21 09:00:00"
      }
    ]
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "inference_id": "inf-uuid-12345",
    "model_name": "metro-traffic-volume",
    "model_version": "1",
    "predictions": [3142.5],
    "latency_ms": 3.8
  }
  ```

### `POST /api/v1/inference/feedback`
Submit delayed ground-truth value for an inference event.
- **Request Body**:
  ```json
  {
    "inference_id": "inf-uuid-12345",
    "actual_value": 3150.0,
    "submitted_by": "induction_loop_01",
    "notes": "Verified roadside sensor reading"
  }
  ```

---

## 7. Monitoring, Drift Analysis & Alerts

### `POST /api/v1/monitoring/drift/input`
Trigger input covariate drift analysis.
- **Request Body**:
  ```json
  {
    "reference_dataset_version_id": "uuid",
    "current_dataset_version_id": "uuid"
  }
  ```

### `GET /api/v1/monitoring/quality`
Retrieve rolling accuracy reports (MAE, RMSE, $R^2$) computed against delayed ground truth.

### `GET /api/v1/monitoring/alerts`
List monitoring alerts with filter by `status` (`OPEN`, `ACKNOWLEDGED`, `RESOLVED`).

### `POST /api/v1/monitoring/alerts/{alert_id}/acknowledge`
Acknowledge an open alert with operator identity.

### `POST /api/v1/monitoring/alerts/{alert_id}/resolve`
Resolve an alert with resolution notes.
