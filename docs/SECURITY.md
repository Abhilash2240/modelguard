# ModelGuard — Security Architecture & Threat Model

## 1. Executive Summary

ModelGuard is a control plane for machine learning reliability. Operating in mission-critical environments requires strict guarantees around data integrity, provenance verification, telemetry privacy, and administrative auditability.

---

## 2. Security Boundaries & Threat Modeling

```
+-------------------------------------------------------------+
|                     External Clients                        |
|  (React Dashboard, API Clients, Induction Loop Sensors)     |
+------------------------------+------------------------------+
                               |
                        [ HTTPS / TLS ]
                               |
+------------------------------v------------------------------+
|                   API Gateway / Nginx                        |
|   - CORS origin enforcement                                 |
|   - Rate limiting & request body size bounds (max 50MB)     |
+------------------------------+------------------------------+
                               |
+------------------------------v------------------------------+
|                  ModelGuard FastAPI Backend                 |
|   - Pandera Schema Contract validation                      |
|   - Release Gate policy enforcement                         |
|   - Two-phase approval state machine                        |
|   - Privacy-safe telemetry masking                          |
+----+--------------------+--------------------+--------------+
     |                    |                    |
     v                    v                    v
+----+-------+   +--------+-------+   +--------+-------+
|  SQLite/   |   | MLflow Server  |   | Prometheus /   |
| PostgreSQL |   | & Artifacts    |   | Grafana        |
+------------+   +----------------+   +----------------+
```

### Threat Vectors & Mitigations

| Threat | Risk Level | ModelGuard Mitigation |
|---|---|---|
| **Dataset Tampering** | Critical | Streaming SHA-256 fingerprinting on snapshot creation; checksum validation before any training run execution. |
| **Model Poisoning / Degraded Weights** | Critical | Automated release gate enforcing hard metric thresholds (MAE, RMSE, $R^2$) and smoke test execution prior to staging approval. |
| **Unauthorized Model Promotion** | High | Two-phase human-in-the-loop sign-off requiring explicit reviewer identity and rationale. Staging cannot bypass approval. |
| **Inference Telemetry Data Leakage** | High | Raw feature payloads are stripped before persisting prediction events. Only statistical summaries and latency metrics are saved. |
| **Silent Metric Drift in Production** | Medium | Covariate drift detectors (KS-test, PSI) and delayed ground-truth accuracy tracking trigger automated persistent alerts. |

---

## 3. Privacy-Safe Telemetry Design

To prevent sensitive or proprietary operational data from accumulating in inference logs:
1. `PredictionRequest` inputs are validated strictly in-memory against the model input contract.
2. The estimator produces prediction arrays.
3. The persisted `PredictionEvent` record retains:
   - `inference_id`: Unique UUIDv4.
   - `model_name` & `model_version`: Exact lineage identifiers.
   - `latency_ms`: Execution duration in milliseconds.
   - `status`: `"SUCCESS"` or error code.
   - `prediction_json`: The generated model outputs.
   - `input_summary_json`: Metadata only (e.g. `{"record_count": 1, "columns_present": [...]}`). **No raw feature values are stored.**

---

## 4. Cryptographic Provenance & Immutability

1. **Dataset Fingerprinting:** Every dataset version computes a SHA-256 hash. If an underlying file on disk is altered or corrupted, `load_validated_dataset()` detects the hash mismatch and halts execution with `IntegrityError`.
2. **Immutable Audit Events:** Every administrative action (`REGISTER_DATASET`, `CREATE_TRAINING_JOB`, `STAGE_CANDIDATE`, `PROMOTE_DEPLOYMENT`, `ROLLBACK_DEPLOYMENT`) emits an immutable record into `audit_events` containing actor, action, timestamp, and before/after states.

---

## 5. Network Segregation & CORS Policy

- **CORS Configuration:** Restricts allowed origins via `settings.ALLOWED_ORIGINS` (defaults to trusted frontend domains).
- **Service Isolation:** Internal MLflow and Prometheus ports are bound to private networks in containerized deployments. Only Nginx and FastAPI endpoints are exposed externally.
