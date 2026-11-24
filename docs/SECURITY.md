# ModelGuard — Security Architecture & Threat Model

## 1. Executive Summary

ModelGuard is an enterprise-grade control plane for machine learning reliability. Operating in mission-critical environments requires strict guarantees around data integrity, provenance verification, telemetry privacy, administrative auditability, and defense against unauthorized model manipulation or telemetry spoofing.

---

## 2. Authentication & Session Management

ModelGuard implements strict authentication and credential security:

1. **Password Hashing (Argon2id):**
   - User credentials are encrypted using Argon2id (`Type.ID`), memory-hard configuration ($t=3$, $m=65536\,\text{KiB}$, $p=4$) through `argon2-cffi`.
   - Passwords are never stored in plaintext or weak hashes (MD5, SHA-1, standard SHA-256).

2. **Cryptographic JWT Tokens:**
   - Stateless authentication via HMAC-SHA256 (`HS256`) signed JSON Web Tokens.
   - Tokens include subject (`sub`), role claim (`role`), issued-at (`iat`), and expiration (`exp`) timestamps.
   - Default access token lifetime: 120 minutes (`ACCESS_TOKEN_EXPIRE_MINUTES`).
   - Signature secret key configured via `SECRET_KEY` environment variable.

3. **Dual Session Transport:**
   - **Bearer Authorization Header:** `Authorization: Bearer <token>` for API clients, automated pipelines, and CI/CD tools.
   - **Secure HttpOnly Cookie:** `access_token` cookie with `HttpOnly`, `SameSite=Lax` flags to mitigate Cross-Site Scripting (XSS) credential theft in browser environments.

4. **Elimination of Caller Actor Impersonation:**
   - Audit trail and lifecycle endpoints derive the operator/approver identity strictly from the authenticated JWT principal (`current_user.username`).
   - Client requests cannot impersonate other users or roles via request body fields.

---

## 3. Role-Based Access Control (RBAC) Matrix

ModelGuard defines four distinct, non-overlapping operational roles:

| Role | Permissions & Access Scope | Typical Persona |
|---|---|---|
| **`VIEWER`** | Read-only access to datasets, training jobs, models, deployments, alerts, and metrics. Permitted to invoke `/api/v1/predict` against the active champion. | Data Analyst, Auditor, Read-only Dashboard User |
| **`OPERATOR`** | All `VIEWER` permissions plus: triggering on-demand drift analyses, acknowledging/resolving alerts, submitting ground-truth feedback, triggering emergency rollbacks, and running diagnostic inference (`/api/v1/inference/diagnostic`). | MLOps Engineer, On-Call Site Reliability Engineer (SRE) |
| **`APPROVER`** | All `OPERATOR` permissions plus: evaluating model release gates, staging model candidates, reviewing smoke test logs, and approving/promoting staged candidates to production champion. | Lead ML Engineer, Model Risk Officer, Release Approver |
| **`ADMIN`** | Full control plane access: all `APPROVER` permissions plus user provisioning, role assignments, system-level reconfiguration, and administrative audits. | Platform Admin, Infrastructure Lead |

---

## 4. Production Model-Version Governance & Fail-Closed Gate

To prevent serving unapproved, untested, or degraded model versions in production:

1. **Strict Active Champion Resolution:**
   - The primary production inference endpoint (`/api/v1/predict`) resolves candidate routing strictly to the active `PROMOTED` champion deployment in the production environment.
   - If an unpromoted or non-champion model version is specified by an external caller, `/predict` rejects the request with `HTTP 400 Bad Request`.
2. **Diagnostic Serving Isolation:**
   - Staged or archived model evaluation is isolated to the dedicated operator endpoint (`/api/v1/inference/diagnostic`), requiring explicit `OPERATOR`, `APPROVER`, or `ADMIN` credentials.
3. **Fail-Closed Release Gates:**
   - Metric thresholds are validated with fail-closed rules: if metrics are absent, contain `NaN` or `Infinity`, or baseline MAE is non-positive, the candidate is automatically rejected with `GATE_FAILED`.

---

## 5. Security Boundaries & Threat Modeling

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
|   - Argon2id Authentication & RBAC Gate                     |
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
| **Model Poisoning / Degraded Weights** | Critical | Automated release gate enforcing hard metric thresholds (MAE, RMSE, $R^2$) and smoke test execution prior to staging approval. Fail-closed on missing/NaN metrics. |
| **Unauthorized Model Promotion** | High | Two-phase human-in-the-loop sign-off requiring authenticated `APPROVER` or `ADMIN` role. Actor identity derived from JWT. |
| **Inference Telemetry Data Leakage** | High | Raw feature payloads are stripped before persisting prediction events. Only statistical summaries and latency metrics are saved. |
| **Silent Metric Drift in Production** | Medium | Real-production prediction event tracking, covariate drift detectors (KS-test, PSI), delayed ground-truth accuracy tracking, and automated alerts. |
| **Unintended Public Port Exposure** | High | All Docker Compose published ports bind explicitly to `127.0.0.1`, preventing exposure on public network interfaces. |

---

## 6. Privacy-Safe Telemetry Design

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

## 7. Cryptographic Provenance & Immutability

1. **Dataset Fingerprinting:** Every dataset version computes a SHA-256 hash. If an underlying file on disk is altered or corrupted, `load_validated_dataset()` detects the hash mismatch and halts execution with `IntegrityError`.
2. **Immutable Audit Events:** Every administrative action (`REGISTER_DATASET`, `CREATE_TRAINING_JOB`, `STAGE_CANDIDATE`, `PROMOTE_DEPLOYMENT`, `ROLLBACK_DEPLOYMENT`, `ACKNOWLEDGE_ALERT`, `RESOLVE_ALERT`) emits an immutable record into `audit_events` containing authenticated principal, action, timestamp, and before/after states.

---

## 8. Network Segregation & Localhost Isolation

- **Localhost Port Binding:** In Docker Compose, all host published ports bind explicitly to loopback (`127.0.0.1:8000:8000`, `127.0.0.1:3000:80`, `127.0.0.1:5000:5000`, `127.0.0.1:9090:9090`, `127.0.0.1:3001:3000`), isolating the control plane from public network interfaces.
- **CORS Policy:** Restricts allowed origins via `settings.ALLOWED_ORIGINS` (defaults to trusted frontend domains).
- **Service Isolation:** Internal MLflow and Prometheus ports communicate across internal Docker bridge network (`modelguard_default`).
