# ModelGuard — Model Registry, Staging & Inference Guide
## Work Package C Operational Runbook

---

### 1. Overview & Architecture

Work Package C implements the Model Candidate Registry, Configurable Release Gate, Staging Deployment with automated Smoke Checks, Version-Aware Prediction, and Audited Human Approval & Promotion.

The lifecycle strictly enforces the state machine:
```text
REGISTERED ──► GATE_PASSED ──► STAGED ──► AWAITING_APPROVAL ──► APPROVED ──► PROMOTED
     │              │            │               │                 │
     ▼              ▼            ▼               ▼                 ▼
   FAILED      GATE_FAILED  SMOKE_FAILED     REJECTED           ARCHIVED
```

Lineage is traceably preserved:
`DatasetVersion → TrainingJob → TrainingRun → MLflow Run → ModelCandidate (Registered Version) → Deployment`

---

### 2. Candidate Registration

Candidate registration extracts the trained `candidate_model` artifact from an executed, immutable `TrainingRun` (`SUCCEEDED`) and registers it into the MLflow Model Registry.

- **Baseline Preservation**: The `baseline_model` artifact is preserved in experiment tracking for evaluation reference and is never registered as a production candidate.
- **Exact Versioning**: MLflow returns an immutable version identifier (e.g. `1`, `2`). Floating identifiers such as `latest` are strictly prohibited.
- **Lineage Metadata**: Links the exact `training_run_id`, `dataset_version_id`, MLflow run ID, model URI (`models:/<model_name>/<version>`), model signature, and tabular input contract.

#### API Invocations:
```bash
# Register candidate from completed TrainingRun
curl -X POST http://localhost:8000/api/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "training_run_id": "0a5c4494-4f95-4b13-a3ea-4e23423f7083",
    "model_name": "metro-traffic-volume"
  }'
```

---

### 3. Configurable Release Gate

The Release Gate evaluates candidates against a versioned, auditable quality policy (`ReleaseGatePolicy`, default `v1.0.0`) using **fail-closed** semantics.

#### Evaluation Criteria:
1. **Dataset Validation Check**: The source `DatasetVersion` must possess at least one passing `ValidationRun` (`status == 'PASSED'`).
2. **Training Integrity**: Underlying `TrainingRun` status must be `SUCCEEDED`.
3. **Metric Thresholds**:
   - Validation MAE $\le 400.0$ (Metro Traffic baseline: $\approx 1720$, candidate: $\approx 280$).
   - Validation RMSE $\le 600.0$.
   - Validation $R^2 \ge 0.85$.
4. **Baseline Superiority**: Candidate must achieve at least $20\%$ lower MAE than the baseline.
5. **Champion Regression Tolerance**: If an active production champion exists, candidate MAE must not regress by more than $5\%$ relative to the champion.
6. **Artifact Load Test**: Artifact at `model_uri` must successfully load via `mlflow.sklearn.load_model`.

If any criterion fails, the gate transition fails to `GATE_FAILED` with exact rejection reasons recorded in `gate_evaluation_json` and an `AuditEvent`. Failed candidates cannot be staged.

---

### 4. Staging Deployment & Smoke Checks

Deploying to staging validates the candidate in an isolated pre-production environment.

#### Workflow:
1. Candidate status verified as `GATE_PASSED`.
2. Staging `Deployment` record created with status `STAGED`.
3. Automated Smoke Test executes:
   - Loads model artifact from exact URI (`models:/metro-traffic-volume/<version>`).
   - Verifies input schema compatibility against tabular contract.
   - Executes representative sample prediction.
   - Asserts non-null, finite numeric output (`np.all(np.isfinite(preds))`).
   - Measures smoke test inference latency.
4. If smoke tests succeed:
   - Deployment status transitions to `AWAITING_APPROVAL`.
   - Candidate status transitions to `AWAITING_APPROVAL`.
   - Staging environment pointer (`active_candidate_id`) updates to the candidate.
5. If smoke tests fail:
   - Deployment status transitions to `SMOKE_FAILED`.
   - Candidate status transitions to `FAILED`.
   - Subsequent approval is blocked.

#### API Invocation:
```bash
curl -X POST http://localhost:8000/api/v1/models/{model_id}/stage
```

---

### 5. Explicit Approval & Exact-Version Promotion

Promotion from staging to production requires explicit human sign-off. Automatic promotions based on metrics or drift are explicitly disallowed.

#### Human Review & Approval:
```bash
curl -X POST http://localhost:8000/api/v1/deployments/{deployment_id}/approve \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "senior-mlops-engineer@company.com",
    "rationale": "Verified validation MAE=276.4, R2=0.92, and zero smoke anomalies on staging."
  }'
```

#### Rejection:
```bash
curl -X POST http://localhost:8000/api/v1/deployments/{deployment_id}/reject \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "reviewer@company.com",
    "reason": "Regressed during weekend congestion simulation."
  }'
```
*Note: Rejection sets deployment and candidate to `REJECTED`; the production champion remains completely unchanged.*

#### Production Promotion:
```bash
curl -X POST http://localhost:8000/api/v1/deployments/{deployment_id}/promote \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "release-manager@company.com",
    "rationale": "Scheduled production rollout."
  }'
```
- Sets MLflow Model Registry alias `champion` pointing to the exact model version via `client.set_registered_model_alias()`.
- Updates ModelGuard `production` environment pointer.
- Archives previous champion in deployment history.

---

### 6. Version-Aware Prediction & Telemetry

#### Endpoint:
`POST /api/v1/predict`

#### Input Contract Validation:
Before the estimator is invoked, the payload is validated:
- All required contract features (`holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`) must be present.
- Timestamps must be valid ISO / datetime strings.
- Numeric features must be non-null and convertible to floats/integers.
- Extraneous or malformed columns trigger HTTP `422 Unprocessable Entity` without calling the estimator.

#### Request Example:
```json
{
  "inputs": [
    {
      "holiday": "None",
      "temp": 288.28,
      "rain_1h": 0.0,
      "snow_1h": 0.0,
      "clouds_all": 40,
      "weather_main": "Clear",
      "weather_description": "sky is clear",
      "date_time": "2026-11-21 09:00:00"
    }
  ]
}
```

#### Response Example:
```json
{
  "inference_id": "4b684cb2-677f-44fe-9892-d95b5424cead",
  "model_name": "metro-traffic-volume",
  "model_version": "1",
  "predictions": [3421.5],
  "latency_ms": 14.8,
  "timestamp": "2026-11-21T16:00:00.123456Z"
}
```

#### Privacy-Safe Telemetry:
Persisted in `PredictionEvent` table:
- `inference_id`: Unique UUID correlating prediction and downstream feedback.
- `candidate_id`: ID of the exact `ModelCandidate`.
- `model_name` & `model_version`: Exact serving version.
- `latency_ms`: Measured execution time in milliseconds.
- `status`: `"SUCCESS"` or `"ERROR"`.
- `prediction_json`: Summary of predictions.
- `input_summary_json`: Non-sensitive input summary (record count, feature names received, null counts). **Raw feature values are omitted by default** to protect user privacy.

---

### 7. Failure Scenarios & Error Codes

| HTTP Status | Condition | Handling |
| :--- | :--- | :--- |
| `400 Bad Request` | Malformed JSON request body | Rejected by FastAPI parser |
| `404 Not Found` | No production model deployed, or model candidate not found | Returns controlled error message |
| `409 Conflict` | Illegal state transition (e.g. promoting unapproved deployment, staging failed candidate) | Returns explanation of required state |
| `422 Unprocessable` | Input contract violation (missing feature, invalid timestamp) | Rejection before estimator execution |
| `500 Server Error` | Model artifact load or estimation failure | Returns 500 and logs error telemetry |

---

### 8. Limitations & Scope Boundary

- **Work Package C Boundary**: Focuses strictly on candidate registration, release gates, staging smoke checks, approval, promotion, and version-aware inference.
- **Rollback**: Rollback workflows and rollback endpoints are part of **Work Package D** (Monitoring, Feedback & Rollback).
- **Delayed Feedback & Drift**: Ingestion of delayed ground-truth labels and Evidently drift detection reports are part of **Work Package D**.
- **Dashboards**: Frontend user interfaces belong to **Work Package E**.
