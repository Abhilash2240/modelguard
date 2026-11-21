# ModelGuard — Product Requirements Document (PRD)

**Product:** ModelGuard — Production ML Reliability Control Plane  
**Version:** 1.0  
**Prepared:** 21 November 2026  
**Status:** Build specification / proposed; implementation has not started  
**Parent strategy:** Hybrid AI Engineering Master Blueprint  
**Primary engineering positioning:** Backend Engineering + ML Engineering + MLOps  
**Initial deployment target:** Local-first, single-developer demonstration using Docker Compose  

---

## 0. Executive decision

ModelGuard is a local-first MLOps control plane for managing the operational lifecycle of tabular machine-learning models. It provides a traceable workflow for dataset registration, data-contract validation, reproducible model training, experiment tracking, model registration, inference, monitoring, controlled model promotion and rollback.

The product must demonstrate more than a training notebook or a page that displays model accuracy. Its core engineering contribution is a reliable connection between **data provenance, model lineage, deployment decisions and observable production behaviour**.

### One-line product definition

> ModelGuard manages the lifecycle of ML models by validating and versioning datasets, recording reproducible training runs, governing model releases, monitoring deployed inference and enabling evidence-based rollback.

### Product promise

A developer should be able to answer, with stored evidence:

- Which dataset snapshot and schema produced this model?
- Which code revision, environment, parameters and random seed were used?
- Which experiment and evaluation report produced the registered model version?
- Which exact model version is serving predictions?
- Is incoming data compatible with the training contract?
- Has input data or model performance changed over time?
- Why was a model promoted, rejected or rolled back?
- Can the previous known-good version be restored and verified?

### Build decision

Build a **modular monolith with a separate training/monitoring worker**. Use FastAPI for the control plane and inference API, PostgreSQL for application state and audit history, MLflow for experiment tracking and model registry, MinIO for local S3-compatible artifact storage, Redis + RQ for background jobs, and Docker Compose for reproducible local operation.

Do not introduce Kubernetes, Kafka, feature stores, distributed training, multi-cloud deployment or a collection of microservices in the first release. Add infrastructure only when a demonstrated requirement justifies it.

---

## 1. Product context and problem

### 1.1 Problem statement

A typical ML prototype can train a model and report a metric, but that is not enough to operate it reliably. Dataset changes can invalidate preprocessing assumptions; training runs may be impossible to reproduce; model artifacts can become detached from their data and code; deployment can replace a better model with a weaker one; and data drift can be confused with actual model-performance degradation.

ModelGuard addresses these operational gaps by recording immutable dataset and model lineage, validating data before training and inference, enforcing deployment gates, collecting prediction telemetry and linking delayed ground-truth labels to the predictions they evaluate.

### 1.2 Primary users

**Primary persona — ML / backend engineer**
- Registers datasets and data contracts.
- Runs training and evaluation pipelines.
- Compares candidate models against a baseline or champion.
- Investigates pipeline failures, drift and inference regressions.

**Secondary persona — model release reviewer**
- Reviews data quality, evaluation evidence, model metadata and deployment risk.
- Approves or rejects promotion.
- Triggers rollback when an approved model violates operational requirements.

For the initial local-first release, authentication can be a simple single-user configuration, but authorization boundaries must be designed so role-based access can be added without rewriting domain logic.

### 1.3 Core use cases

1. Register a dataset with source, licence, checksum, schema and storage metadata.
2. Validate a dataset against a versioned contract and preserve the full validation result.
3. Start a reproducible training run using an immutable dataset version and configuration.
4. Compare candidate metrics against a baseline and register a successful candidate in MLflow.
5. Deploy the candidate to staging and execute inference smoke tests.
6. Approve promotion to production and record the exact model version and approval decision.
7. Send inference requests and record latency, model version, input-contract status and prediction metadata.
8. Detect feature/prediction distribution changes and distinguish them from measured model-quality degradation.
9. Ingest delayed labels and calculate rolling model-quality metrics.
10. Roll back to a previous model version, verify the active version and preserve an audit event.

---

## 2. Goals, non-goals and success criteria

### 2.1 Goals

- Deliver one complete end-to-end ML lifecycle, not a disconnected collection of components.
- Preserve dataset, code, environment, experiment and model lineage.
- Make data validation deterministic and explainable.
- Make training reproducible from a stored run configuration and data snapshot.
- Make model release decisions depend on explicit, configurable gates.
- Keep inference observable, version-aware and traceable to a registered model.
- Demonstrate data drift, prediction drift and delayed-label model-quality monitoring as separate concepts.
- Demonstrate a verified rollback to a known-good model.
- Provide a usable React dashboard and a clean Docker Compose setup.
- Publish reproducible experiments, test evidence and honest limitations.

### 2.2 Non-goals for the first release

- No general AutoML platform or arbitrary notebook execution environment.
- No distributed or GPU-heavy training requirement.
- No Kubernetes, Kafka, Spark or multi-cloud deployment.
- No feature store or online/offline feature consistency platform.
- No automatic model promotion based only on a drift alert.
- No autonomous remediation or unreviewed production promotion.
- No multi-tenant SaaS claims or public execution of untrusted training code.
- No claim that a statistical drift detector proves model failure or causality.
- No claim of production readiness for regulated financial, healthcare or safety-critical decisions.

### 2.3 Success criteria

The release is successful when a clean-machine setup can:

1. Ingest the primary UCI traffic dataset and create a dataset version with source citation, licence and SHA-256 checksum.
2. Validate the dataset against a versioned data contract and persist row-level/column-level findings.
3. Run a chronological training pipeline with a baseline and at least one candidate estimator.
4. Record parameters, metrics, tags, code revision, data snapshot, environment/lockfile fingerprint and artifacts in MLflow and the ModelGuard database.
5. Register a candidate model and verify its signature / input schema.
6. Deploy that exact version to staging, run smoke tests and require explicit approval before production promotion.
7. Serve a prediction and record request ID, model version, latency and prediction metadata.
8. Run a deterministic drift/data-quality scenario and show its report and alert state.
9. Attach a delayed ground-truth label to a prior prediction and calculate a measured model-quality metric.
10. Roll back to the prior approved model and verify the active model identifier.
11. Pass unit, integration, API and end-to-end tests in CI.

---

## 3. Product workflow

### 3.1 Main lifecycle

```text
Dataset source / upload
        |
        v
Dataset registration + immutable snapshot + SHA-256
        |
        v
Versioned data contract (Pandera)
        |
        +---- validation failed --> quarantine / report / stop
        |
        v
Validated dataset version
        |
        v
Training job (separate worker)
        |
        v
Baseline + candidate evaluation (chronological split)
        |
        +---- gate failed --> candidate rejected / report
        |
        v
MLflow run + registered model version + lineage
        |
        v
Staging deployment + smoke / compatibility tests
        |
        v
Human approval + release gate
        |
        v
Production active-model pointer / MLflow alias
        |
        v
Inference + request telemetry + prediction log
        |
        +--> input / prediction drift monitoring
        |
        +--> delayed labels --> measured model quality
        |
        v
Alert / investigate / rollback to known-good version
```

### 3.2 Critical distinction: three monitoring layers

ModelGuard must not conflate these signals:

1. **Data quality / contract violations:** missing required columns, invalid types, null spikes, duplicates, impossible ranges, invalid categories or timestamp problems. These can be detected immediately at ingestion or inference.
2. **Data / prediction drift:** changes in feature or prediction distributions relative to a reference period. Drift is an early-warning signal, not proof of worse model performance.
3. **Model quality degradation:** actual deterioration measured against ground-truth labels that arrive later. Examples include MAE/RMSE for regression and PR-AUC, precision, recall or calibration for classification.

All three need separate statuses, thresholds, timestamps and evidence.

---

## 4. Scope and functional requirements

Priority values: **P0 = required for the first complete vertical slice**, **P1 = required for a strong portfolio release**, **P2 = later enhancement**.

### 4.1 Dataset registry and provenance

| ID | Priority | Requirement | Acceptance evidence |
|---|---|---|---|
| DS-01 | P0 | Register a dataset from a supported local file or curated dataset adapter | Dataset record contains name, task, source, licence, schema, creation time and storage reference |
| DS-02 | P0 | Create immutable dataset versions | Each version records SHA-256, byte size, row count, columns, dtypes and source metadata |
| DS-03 | P0 | Preserve dataset lineage | Training run links to exact dataset version and validation run |
| DS-04 | P1 | Preview dataset profile | Dashboard displays sample, row/column counts, dtypes, missingness and summary statistics |
| DS-05 | P1 | Prevent accidental replacement | New content creates a new version; it never silently overwrites a referenced snapshot |
| DS-06 | P1 | Record attribution and usage constraints | Dataset page and exported report include source, citation and licence information |

### 4.2 Data validation and contracts

| ID | Priority | Requirement | Acceptance evidence |
|---|---|---|---|
| DV-01 | P0 | Define a versioned tabular schema | Required columns, types, nullability and domain checks are stored |
| DV-02 | P0 | Validate before training | Failed validation prevents training from starting |
| DV-03 | P0 | Persist validation evidence | Per-check status, affected column, observed value/count and explanation are stored |
| DV-04 | P1 | Support strictness and configurable severity | Blocking failures differ from warnings |
| DV-05 | P1 | Validate inference payloads against model signature | Invalid requests return a safe 4xx response and are counted without invoking the model |
| DV-06 | P1 | Support explicit schema evolution | Contract changes require a new contract version and validation review |

Use Pandera for dataframe-level contracts in the initial implementation. Keep validation rules in a ModelGuard-owned schema/configuration layer; Pandera is an implementation adapter, not the source of business policy.

### 4.3 Training and experiment tracking

| ID | Priority | Requirement | Acceptance evidence |
|---|---|---|---|
| TR-01 | P0 | Submit a training job asynchronously | API returns a job/run identifier without waiting for training to finish |
| TR-02 | P0 | Use an immutable dataset version | Training job fails if the referenced source is missing or its checksum does not match |
| TR-03 | P0 | Use deterministic data splitting and recorded random seeds | Repeated runs with the same environment/configuration are reproducible within documented tolerances |
| TR-04 | P0 | Train a simple baseline and candidate model | Report shows baseline and candidate metrics on the same temporal holdout |
| TR-05 | P0 | Track all experiment metadata in MLflow | Parameters, metrics, tags, model/artifacts and dataset information are visible |
| TR-06 | P1 | Capture code and environment provenance | Git SHA, dependency lockfile hash, Python version and pipeline version are attached |
| TR-07 | P1 | Store evaluation artifacts | Predictions, residual/error summaries, feature metadata and metric report are available |
| TR-08 | P1 | Support bounded retry/cancellation | Failed jobs have explicit failure category and cannot be marked successful by retry side effects |

Use a scikit-learn `Pipeline` so preprocessing is fit only on training data and is serialized with the estimator. No preprocessing fit may use validation or test rows.

### 4.4 Model registry and release governance

| ID | Priority | Requirement | Acceptance evidence |
|---|---|---|---|
| MR-01 | P0 | Register models in MLflow | ModelGuard links its candidate record to the exact MLflow model/version/URI |
| MR-02 | P0 | Enforce a configurable evaluation gate | Candidate must pass configured task metrics and compatibility checks before staging |
| MR-03 | P0 | Record model signature and input contract | Staging/inference use the same validated contract that was approved |
| MR-04 | P1 | Compare candidate with current champion | Candidate report shows absolute metrics and delta against baseline/champion |
| MR-05 | P1 | Record promotion approval | Approval links actor, timestamp, model version, target environment and decision rationale |
| MR-06 | P1 | Support immutable model version identifiers | A deployment references a specific version, not a floating `latest` reference |
| MR-07 | P1 | Support rollback | Previous known-good model can be reactivated and verified |

Use MLflow aliases for mutable references such as `champion` or `staging-candidate`; do not build around legacy MLflow model stages. Keep ModelGuard's own approval and deployment state in its database. MLflow is the model/experiment registry, not the sole audit log.

### 4.5 Inference and prediction logging

| ID | Priority | Requirement | Acceptance evidence |
|---|---|---|---|
| INF-01 | P0 | Expose a typed prediction endpoint | Valid input returns prediction, inference ID, model name and exact version |
| INF-02 | P0 | Record prediction telemetry | Store request timestamp, model version, latency, status and non-sensitive prediction metadata |
| INF-03 | P1 | Provide prediction batch endpoint | Bounded batches are validated and return per-record outcomes |
| INF-04 | P1 | Reject incompatible model/data signatures | No silent column reordering, implicit feature invention or unsafe coercion |
| INF-05 | P1 | Support delayed labels | A feedback endpoint links actual outcomes to inference IDs |
| INF-06 | P1 | Preserve privacy | Do not retain raw personal data by default; configure sampling/retention and redact logs |

### 4.6 Monitoring, drift and alerts

| ID | Priority | Requirement | Acceptance evidence |
|---|---|---|---|
| MO-01 | P0 | Monitor service health | Request count, error count, latency histogram and active model version are observable |
| MO-02 | P0 | Detect input contract violations | Invalid requests are visible by reason and column without leaking raw payloads |
| MO-03 | P1 | Compare production/reference feature distributions | Drift report records reference/current windows, method, threshold and result |
| MO-04 | P1 | Monitor prediction distribution | Prediction drift is separately reported from input drift |
| MO-05 | P1 | Calculate model quality after labels arrive | Rolling metric report includes sample count and label coverage |
| MO-06 | P1 | Generate alerts with explicit severity | Alert links to the report, model version, time window and triggering evidence |
| MO-07 | P1 | Support deterministic fault injection | A documented scenario produces a known validation/drift/quality alert |
| MO-08 | P2 | Add alert acknowledgement and resolution workflow | State changes are audited |

Use Evidently or a small, well-tested statistical adapter for drift reporting. Keep the drift interface replaceable. Thresholds must be configurable and calibrated against the baseline/reference data; do not hard-code a universal PSI or p-value threshold and claim it is universally valid.

### 4.7 Deployment controller and rollback

| ID | Priority | Requirement | Acceptance evidence |
|---|---|---|---|
| DEP-01 | P0 | Create staging deployment for a registered version | Staging references an exact immutable model version |
| DEP-02 | P0 | Run readiness and smoke checks | Health/readiness, schema compatibility and sample prediction must pass |
| DEP-03 | P0 | Require explicit human approval for production | No production pointer/alias changes before approval |
| DEP-04 | P1 | Promote atomically | Persist previous version, new version, approval and deployment result |
| DEP-05 | P1 | Roll back to previous approved version | Inference confirms the previous exact version is active |
| DEP-06 | P2 | Add shadow/canary evaluation | New model can be compared without initially changing user-facing predictions |

No model may be promoted solely because the drift detector fires. Promotion and rollback require policy gates and an auditable decision.

### 4.8 Dashboard and reporting

P0/P1 dashboard surfaces:
- Overview: system health, current champion, latest runs, unresolved alerts.
- Datasets: versions, metadata, provenance, schemas and validation history.
- Training: queued/running/completed jobs, parameters, metrics and artifact links.
- Model registry: versions, lineage, evaluation comparison and signatures.
- Deployments: staging/production model version, approval history and rollback action.
- Monitoring: service metrics, input/prediction drift, delayed-label model quality and alerts.
- Audit: who did what, when, to which model/dataset and with what result.

The dashboard is a thin operational UI. Do not recreate the entire MLflow UI; link to MLflow for deep run/artifact inspection.

---

## 5. Dataset strategy and ML experiments

### 5.1 Primary MVP dataset — Metro Interstate Traffic Volume

- Publisher: UCI Machine Learning Repository.
- Dataset ID: 492.
- Size: 48,204 observations, 8 input features plus target.
- Period: hourly observations from 2012–2018.
- Task: regression.
- Features: holiday, temperature, rain, snow, cloud cover, weather category/description and timestamp.
- Target: hourly traffic volume.
- Licence: CC BY 4.0; include attribution.
- Why selected: moderate size, explicit timestamp, chronological structure and weather/context features make it suitable for reproducible temporal splits and controlled monitoring demonstrations.
- Official page: https://archive.ics.uci.edu/dataset/492/urban+traffic
- Citation: Hogue, J. (2019). Metro Interstate Traffic Volume. UCI Machine Learning Repository. DOI: 10.24432/C5X60B.

### 5.2 Secondary classification dataset — Bank Marketing

- Publisher: UCI Machine Learning Repository.
- Dataset ID: 222.
- Task: binary classification — subscription of a term deposit.
- Full older variant: 45,211 observations and 16 input features as described by UCI.
- `bank-additional-full.csv`: 41,188 examples with 20 inputs, ordered by date from May 2008 to November 2010.
- Licence: CC BY 4.0; include attribution.
- Important leakage rule: the `duration` field is only known after the phone call. Exclude it for a pre-call decision use case, or clearly define a separate post-call prediction scenario.
- Official page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Citation: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. DOI: 10.24432/C5K306.

### 5.3 Additional regression/sensor dataset — Appliances Energy Prediction

- Publisher: UCI Machine Learning Repository.
- Dataset ID: 374.
- Size: 19,735 observations; 28 model input features plus target.
- Sampling: 10-minute measurements over approximately 4.5 months.
- Task: regression; target is appliance energy consumption in Wh.
- Licence: CC BY 4.0; include attribution.
- Useful detail: includes temperature/humidity sensor variables and two random variables (`rv1`, `rv2`) that can be used to test feature filtering and non-predictive columns.
- Official page: https://archive.ics.uci.edu/dataset/374/appliancesenergyprediction
- Citation: Candanedo, L. (2017). Appliances Energy Prediction. UCI Machine Learning Repository. DOI: 10.24432/C5VC8G.

### 5.4 Optional regression dataset — Bike Sharing

- Publisher: UCI Machine Learning Repository.
- Dataset ID: 275.
- Size: 17,389 observations and 13 features.
- Period: hourly/daily bike rental counts in 2011–2012.
- Task: regression.
- Licence: CC BY 4.0; include attribution.
- Official page: https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset
- Citation: Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. DOI: 10.24432/C5W894.

### 5.5 Excluded from default public demo — UCI Air Quality

This dataset has 9,358 hourly sensor observations and documented sensor/concept drift, but the UCI page states that it is for research purposes only and commercial use is excluded. Do not make it the default public/commercial demo data source. It may be used in a research-only experiment if the use restriction is respected and documented.

Official page: https://archive.ics.uci.edu/dataset/387/air+quality
DOI: 10.24432/C5060Z.

### 5.6 Synthetic deterministic failure fixtures

Public benchmark datasets do not cover every operational fault in a controlled and repeatable way. Add deterministic synthetic fixtures generated from fixed seeds:

- Schema failure: missing/renamed column, unexpected column, changed dtype, unknown category.
- Data-quality failure: null spike, duplicate rows, impossible range, malformed timestamp, invalid target.
- Drift: mean/variance shift, categorical frequency shift, abrupt level shift, gradual shift and seasonal change.
- Prediction drift: input window changes produce a shifted prediction distribution.
- Concept/performance degradation: controlled target relationship change followed by delayed labels; measure actual error separately from input drift.
- Pipeline failure: missing source, checksum mismatch, corrupted artifact, failed training command, time limit exceeded.
- Serving failure: model signature mismatch, unavailable artifact, inference timeout and invalid payload.
- Release failure: candidate fails metric gate, staging smoke test fails, approval is denied, rollback target is unavailable.

Every fixture needs a manifest with scenario ID, seed, source snapshot, transformation, expected detection, expected severity and expected system response. Synthetic scenarios are for reliability testing, not evidence of real-world model performance.

### 5.7 Dataset handling rules

- Do not commit raw datasets or trained model binaries to Git unless they are tiny fixtures expressly intended for tests.
- Store raw files and model artifacts in MinIO; store metadata and hashes in PostgreSQL.
- Persist original source, DOI, licence, attribution and retrieval timestamp.
- Record the exact file hash and schema for every dataset version.
- Never silently refresh a dataset in place.
- Use chronological splits for datasets with time ordering; never shuffle away temporal structure in the primary experiment.
- Fit encoders, imputers and scalers on the training partition only.
- Keep a final untouched test period for release evaluation.
- Avoid claiming generalization beyond the documented source population and time range.

---

## 6. ML methodology and evaluation protocol

### 6.1 Primary regression experiment

Dataset: Metro Interstate Traffic Volume.

- Establish a naive baseline (for example, training-period mean or a clearly specified seasonal baseline where feasible).
- Establish a compact ML candidate such as HistGradientBoostingRegressor or RandomForestRegressor.
- Build preprocessing and estimator as a single scikit-learn Pipeline.
- Derive timestamp features from the input timestamp; do not use future target values.
- Use chronological train/validation/test partitions, with exact date boundaries recorded in the run manifest.
- Report MAE, RMSE and R²; include residual analysis and prediction-vs-actual plots.
- Compare all candidates on the same untouched temporal test window.
- Tune only on training/validation data; do not repeatedly inspect or optimize against the final test period.

### 6.2 Secondary classification experiment

Dataset: UCI Bank Marketing `bank-additional-full.csv` when using its ordered variant.

- Use a chronological split based on documented row order; verify the source ordering before relying on it.
- Exclude `duration` for a pre-call prediction workflow to avoid post-outcome leakage.
- Include a DummyClassifier baseline and a compact candidate such as HistGradientBoostingClassifier or a suitable categorical preprocessing + tree pipeline.
- Report PR-AUC, ROC-AUC, precision, recall, F1 and confusion matrix.
- Select the decision threshold using validation data; document the threshold and operating trade-off.
- Report class prevalence and always include metric denominators/sample counts.

### 6.3 Model release gates

Release gates are configuration, not universal magic constants. Example gate categories:

- Required validation checks passed.
- Model signature compatible with the serving contract.
- Candidate metric meets the configured minimum.
- Candidate does not exceed configured regression tolerance against the current champion.
- Artifact checksum and load test pass.
- Staging health and sample inference tests pass.
- Required reviewer approval exists.

The exact metric thresholds must be set after baseline experiments and stored in a versioned policy file. A drift report alone is not a release or rollback command.

### 6.4 Monitoring metrics

**Service metrics**
- Request count by endpoint/status.
- Error count/rate.
- Inference latency histogram (p50/p95/p99 derived from histogram buckets or stored request samples).
- Active model version and model-load failures.
- Queue depth, worker failures and job duration.

**Data metrics**
- Missingness, invalid rows, duplicate rate and schema violations.
- Numeric distribution changes and categorical frequency changes.
- Reference/current window size and coverage.

**Model metrics**
- Prediction distribution.
- Regression: rolling MAE/RMSE and optional R² once labels are available.
- Classification: rolling PR-AUC/ROC-AUC, precision, recall, F1 and calibration as label volume permits.
- Label delay/coverage and sample count.

Every report must include reference window, current window, computation time, source version, method, threshold/policy version and caveats.

---

## 7. Recommended architecture

### 7.1 Logical architecture

```text
                    React + TypeScript Dashboard
                                |
                         FastAPI Control Plane
                    /           |             \
             PostgreSQL       Redis/RQ       MLflow API
          app state/audit       queue       runs + registry
                    \           |             /
                     \       Worker         /
                      \   /    |    \       /
                       Data  Training  Monitor
                       validation pipeline  jobs
                         |       |       |
                         |       |       +----> Evidently / metrics
                         |       |
                         +-------+--------------> MinIO artifacts
                                |
                          Model artifact
                                |
                      FastAPI inference routes
                                |
                    Prediction logs + feedback labels
                                |
                    Prometheus metrics / Grafana
```

### 7.2 Technology decisions

| Layer | Selected technology | Reason / constraint |
|---|---|---|
| Backend API | Python + FastAPI + Pydantic v2 | Typed API and service boundaries |
| ORM | SQLAlchemy 2.x | Durable relational model and transaction control |
| Migrations | Alembic | Reproducible schema evolution |
| Application DB | PostgreSQL | Dataset/model/deployment metadata, jobs and audit events |
| Queue | Redis + RQ | Separate long-running work from HTTP request handlers |
| Experiment tracking | MLflow | Run metadata, parameters, metrics, artifacts and lineage |
| Model registry | MLflow Model Registry | Immutable model versions and alias references |
| Artifact storage | MinIO (S3-compatible) | Local object storage for datasets and artifacts |
| Data contract validation | Pandera | Typed dataframe schema and data checks |
| Drift / evaluation | Evidently (adapter) + project metrics | Reusable drift reports and testable monitoring logic |
| ML pipeline | pandas, NumPy, scikit-learn | CPU-first tabular baselines and candidate models |
| Frontend | React + TypeScript + Vite | Dashboard and workflow visibility |
| Visual styling | Existing project design system or simple Tailwind | Keep UI consistent and maintainable |
| Monitoring | Prometheus + Grafana | Service and worker metrics with visual dashboards |
| Logging | Structured JSON logs | Correlate request, run, model and deployment IDs |
| Tracing | OpenTelemetry after core metrics/logging | Add end-to-end traces when service boundaries justify it |
| Packaging | Docker + Docker Compose | Clean, reproducible local startup |
| CI | GitHub Actions | Lint, type-check, test, build and Compose smoke tests |

### 7.3 Important architectural boundaries

- The API validates requests and schedules work; it does not train models inside request handlers.
- The worker owns ingestion, validation, training, monitoring and other long-running jobs.
- PostgreSQL is the source of truth for ModelGuard workflow state and audit records.
- MLflow owns experiment tracking and model artifact/registry operations; ModelGuard stores references and release decisions.
- MinIO stores large raw files and artifacts; PostgreSQL stores object keys, hashes and metadata.
- Inference uses an immutable model version. Never load a moving `latest` model reference for each request.
- The production model pointer is switched only through a transactionally controlled release service with a recorded previous version and a verified result.
- The first release can serve inference through the FastAPI application, but keep model loading and prediction logic behind a serving interface so it can be separated later if needed.

### 7.4 Local Docker Compose services

Required for the full local demo:
- `api`
- `worker`
- `postgres`
- `redis`
- `mlflow`
- `minio`
- `frontend`

Add `prometheus` and `grafana` for the monitoring release gate. Avoid an additional orchestrator server until the actual workflows require it. RQ is the initial job queue; do not add Prefect and RQ together without a documented need.

### 7.5 Model artifact format and loading

- Train models as scikit-learn Pipelines with preprocessing included.
- Log model signature and input example through MLflow.
- Validate artifact load in the worker and in staging before promotion.
- Treat pickle/joblib-based artifacts as trusted code: only load artifacts produced by the controlled pipeline or a trusted registry. Do not load arbitrary user-uploaded model binaries.
- Use artifact checksums and record the artifact URI.

---

## 8. Data model proposal

Use UUID primary keys. Keep domain state in ModelGuard's PostgreSQL database and reference MLflow/MinIO resources explicitly.

| Entity | Important fields | Purpose |
|---|---|---|
| `datasets` | id, name, task_type, source_uri, licence, description, created_at | Logical dataset identity |
| `dataset_versions` | id, dataset_id, version, object_uri, sha256, bytes, rows, columns_json, schema_hash, retrieved_at | Immutable dataset snapshot |
| `data_contracts` | id, dataset_id, version, contract_json, status, created_at | Versioned validation contract |
| `validation_runs` | id, dataset_version_id, contract_id, status, summary_json, started_at, ended_at | Validation execution evidence |
| `training_jobs` | id, dataset_version_id, config_json, status, queue_job_id, created_at | Durable training job state |
| `training_runs` | id, training_job_id, mlflow_run_id, git_sha, lock_hash, seed, status, metrics_json | Reproducible training metadata |
| `model_candidates` | id, training_run_id, mlflow_model_name, mlflow_version, model_uri, signature_json, status | Candidate and registry reference |
| `evaluation_reports` | id, candidate_id, dataset_version_id, split_manifest_json, metrics_json, artifact_uri | Evaluation record |
| `deployment_environments` | id, name, active_model_candidate_id, updated_at | Environment pointer |
| `deployments` | id, environment_id, candidate_id, previous_candidate_id, status, approved_by, approved_at, started_at, ended_at | Promotion/rollback history |
| `prediction_events` | id, inference_id, candidate_id, request_time, latency_ms, status, prediction_json, input_summary_json | Inference telemetry without raw sensitive payload by default |
| `feedback_labels` | id, inference_id, actual_value, received_at, source | Delayed ground truth |
| `monitoring_reports` | id, candidate_id, reference_window_json, current_window_json, report_type, metrics_json, status | Drift and quality reports |
| `alerts` | id, report_id, severity, state, title, evidence_json, created_at, resolved_at | Alert lifecycle |
| `audit_events` | id, actor, action, resource_type, resource_id, before_json, after_json, reason, created_at | Decision and operation history |

### Data integrity constraints

- Unique `(dataset_id, version)`.
- Unique `(dataset_version_id, contract_id, validation_attempt)` where a retry identity is needed.
- Unique MLflow run/model-version references where appropriate.
- `inference_id` unique and idempotent for feedback submission.
- Only one active production deployment pointer per model/environment.
- Deployment approval must reference the exact candidate/model version and target environment.
- Rollback must reference a previously successful deployment version.
- Foreign-key cascade behaviour must not delete audit, prediction or lineage records accidentally.

---

## 9. API contract proposal

Prefix all routes with `/api/v1`. Use Pydantic request/response schemas, consistent error codes and pagination. Long-running work returns `202 Accepted` with a job/run identifier.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/datasets` | Register dataset metadata/upload reference |
| GET | `/datasets` | List datasets |
| GET | `/datasets/{dataset_id}` | Dataset details and versions |
| POST | `/datasets/{dataset_id}/versions` | Register immutable dataset version |
| GET | `/datasets/{dataset_id}/versions/{version_id}` | Inspect snapshot metadata |
| POST | `/dataset-versions/{version_id}/validate` | Schedule validation |
| GET | `/validation-runs/{run_id}` | Validation result and findings |
| POST | `/training-jobs` | Submit training configuration |
| GET | `/training-jobs/{job_id}` | Job status |
| GET | `/training-runs/{run_id}` | Run provenance and results |
| GET | `/models` | List registered model candidates |
| GET | `/models/{model_id}` | Model lineage and evaluation |
| POST | `/models/{model_id}/stage` | Deploy exact version to staging |
| POST | `/deployments/{deployment_id}/approve` | Approve exact release |
| POST | `/deployments/{deployment_id}/reject` | Reject release with reason |
| POST | `/deployments/{deployment_id}/promote` | Promote approved candidate |
| POST | `/deployments/{deployment_id}/rollback` | Roll back to recorded previous version |
| GET | `/deployments` | Deployment history and active versions |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Bounded batch prediction |
| POST | `/feedback` | Attach a ground-truth label to inference ID |
| POST | `/monitoring/jobs` | Schedule a monitoring report |
| GET | `/monitoring/reports` | List monitoring reports |
| GET | `/alerts` | List current/historical alerts |
| GET | `/audit-events` | Search audit history |
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness including required dependencies |
| GET | `/metrics` | Prometheus metrics endpoint |

### API behaviour requirements

- Return `400` for malformed payloads, `404` for missing resources, `409` for illegal lifecycle transitions, `422` for schema/semantic validation failures and `503` for unavailable dependencies where appropriate.
- Use idempotency keys for job submission, promotion and feedback ingestion where retry duplication is possible.
- Never expose raw object-store credentials or private artifact paths to the browser.
- Do not return raw data rows in logs or error messages.
- Enforce maximum upload size and maximum prediction batch size.

---

## 10. Frontend requirements

### 10.1 Core pages

1. **Overview:** active model, latest training run, data quality status, drift/quality status, service health and latest alerts.
2. **Datasets:** dataset list, versions, source/licence, checksum, schema and validation history.
3. **Dataset detail:** preview, column profile, contract, validation findings and snapshot metadata.
4. **Training:** submit job, choose dataset version/config, view queue/run status and compare metrics.
5. **Experiments:** baseline/candidate comparison and MLflow links.
6. **Model registry:** model lineage, version, signature, evaluation result and status.
7. **Deployments:** staging/production, active version, approval record and rollback.
8. **Monitoring:** service latency/errors, data drift, prediction drift, delayed-label quality and report evidence.
9. **Alerts & audit:** searchable event history and resolution state.

### 10.2 UX rules

- Make status labels explicit: queued, running, succeeded, failed, blocked, awaiting approval, deployed, rolled back.
- Distinguish “no data”, “not yet evaluated” and “healthy”.
- Every metric shows a sample count and time window.
- Every alert links to supporting report evidence.
- Show exact model/dataset version identifiers wherever an action changes state.
- Confirm irreversible or release-affecting actions.
- Make failed checks visible; never show a green success state when a check was skipped.

---

## 11. Security, privacy and reliability

### 11.1 Threats and controls

| Risk | Control |
|---|---|
| Untrusted dataset or malformed file | File size limits, safe parsers, type checks, content hash, no arbitrary execution during ingestion |
| Malicious uploaded model artifact | Do not load arbitrary user-supplied pickle/joblib; load only artifacts from trusted controlled training runs |
| Secret leakage | `.env`/secret injection, never log secrets, keep credentials out of MLflow params and model context |
| Unsafe training pipeline | Run controlled training code in a worker/container with resource and time limits; no host-home mount or Docker socket |
| Artifact tampering | SHA-256 artifact checksums and registry URI verification |
| Unauthorized promotion | Exact-version approval, least privilege, auditable release gate |
| Rollback race | Transactional deployment record, optimistic concurrency/version check and post-switch verification |
| Data privacy leakage | Avoid storing raw prediction inputs by default; redact, sample and apply retention limits |
| Operational failure | Job retries with idempotency, explicit states, health/readiness checks and restart recovery |

### 11.2 Local sandbox posture

The MVP is a controlled local engineering demonstration, not a public arbitrary-code execution service. Use a non-root worker container, CPU/memory/time limits, limited writable directories and no ambient cloud/Git credentials. If arbitrary customer-supplied training code is introduced later, require a stronger isolation design and security review before deployment.

### 11.3 Reliability requirements

- API requests remain responsive while training/monitoring runs.
- Worker restart does not erase completed run records.
- Jobs have clear timeout, retry and terminal failure states.
- A model deployment cannot be marked successful until the serving version has been verified.
- Database/object-store/MLflow outages produce visible degraded states.
- Rollback is idempotent and preserves audit history.

---

## 12. Testing and quality strategy

### Unit tests
- Dataset hash and version logic.
- Contract validation and error classification.
- Training split boundaries and no-leakage rules.
- Metric calculation and threshold policies.
- Model signature validation.
- Model lifecycle and deployment state transitions.
- Drift report classification.
- Feedback idempotency and label matching.
- Rollback pointer selection.

### Integration tests
- PostgreSQL persistence/migrations.
- Redis/RQ enqueue, worker execution and retry behaviour.
- MLflow run creation, model registration and artifact resolution.
- MinIO upload/download and checksum checks.
- Pandera validation integration.
- Inference with a registry-backed model version.
- Prometheus metrics endpoint and alert persistence.

### End-to-end tests
- Register -> validate -> train -> evaluate -> register model -> stage -> approve -> promote -> predict -> add label -> monitor -> rollback.
- Contract failure blocks a training job.
- Poor candidate metric fails promotion.
- Staging smoke test failure prevents production change.
- Approval rejection leaves production model unchanged.
- Worker restart resumes or safely marks interrupted jobs.

### Security tests
- Oversized file upload rejected.
- Path traversal and unsafe file names rejected.
- Secrets do not appear in logs or model metadata.
- Invalid inference schema never reaches estimator.
- Unauthorized promotion/rollback rejected.
- Malformed/corrupt model artifacts are not loaded.
- Prediction logging omits sensitive raw inputs by default.

### Code quality gates
- Ruff format/lint.
- Pyright or mypy (choose one and configure consistently).
- Pytest unit/integration suite.
- Frontend lint, type-check and build.
- Docker Compose startup/health smoke test.
- Dependency lockfile committed; CI installs from lock.

---

## 13. Repository structure proposal

Create only modules required by the active implementation slice. This is a proposed target layout, not a requirement to pre-create empty files.

```text
modelguard/
├── backend/
│   ├── app/
│   │   ├── api/v1/routes/
│   │   ├── core/                 # settings, logging, errors, security
│   │   ├── db/                   # session, models, migrations
│   │   ├── domain/               # lifecycle rules and entities
│   │   ├── schemas/              # Pydantic contracts
│   │   ├── repositories/         # persistence interfaces/adapters
│   │   ├── services/             # dataset, training, registry, deployment
│   │   ├── ml/                   # preprocessing, train, evaluate, inference
│   │   ├── monitoring/           # drift, quality, alerts, metrics
│   │   ├── integrations/         # mlflow, minio, redis, evidently
│   │   └── workers/              # RQ jobs and worker entrypoint
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── api/
│   │   ├── e2e/
│   │   └── security/
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── features/
│   │   ├── pages/
│   │   ├── services/
│   │   └── types/
│   └── package.json
├── datasets/
│   ├── manifests/
│   └── fixtures/                 # small synthetic test fixtures only
├── experiments/
│   ├── configs/
│   └── reports/
├── infrastructure/
│   ├── docker/
│   └── prometheus/
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   ├── ER_DIAGRAM.md
│   ├── API_CONTRACTS.md
│   ├── SECURITY.md
│   ├── DATASETS.md
│   ├── EVALUATION.md
│   ├── ARCHITECTURE_DECISIONS.md
│   └── INTERVIEW_PREPARATION.md
├── scripts/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── uv.lock
```

---

## 14. Delivery sequence (work packages, not a change to the parent portfolio roadmap)

Implement the product as sequential vertical slices. Do not code all modules at once and do not create additional portfolio-level projects or parallel subprojects.

### Work package A — reproducible data foundation
- Repository scaffold, settings, logging, migrations.
- Dataset registration/versioning and checksum.
- Metro traffic data adapter and attribution manifest.
- Pandera contract and deterministic validation report.
- CLI/script to import and validate the dataset.

**Exit gate:** dataset registration and validation are reproducible; invalid data is blocked with evidence.

### Work package B — reproducible training and experiment tracking
- Async job submission and worker.
- Chronological split manifest.
- Baseline + candidate scikit-learn Pipeline.
- MLflow tracking, parameters, metrics, model signature and artifacts.
- Run provenance and evaluation report.

**Exit gate:** a clean rerun with identical config/data/seed produces traceable results within documented numeric tolerances.

### Work package C — model registry, staging and inference
- Candidate record linked to MLflow model version.
- Configurable release gate.
- Staging deployment and smoke checks.
- Version-aware prediction endpoint and telemetry.
- Explicit production approval and exact-version promotion.

**Exit gate:** candidate cannot reach production without passing checks and explicit approval.

### Work package D — monitoring, feedback and rollback
- Prometheus metrics and Grafana dashboard.
- Input/data and prediction drift reports.
- Delayed feedback labels and rolling quality metrics.
- Deterministic drift/performance fault scenarios.
- Audited rollback to previous known-good model.

**Exit gate:** drift, quality degradation and service failures are distinguishable; rollback is verified and auditable.

### Work package E — dashboard, hardening and portfolio evidence
- React dashboard views for all core lifecycle stages.
- E2E tests and CI.
- Clean Docker Compose setup.
- HLD, LLD, schema, API and security docs.
- README, screenshots/demo and INTERVIEW_PREPARATION.md.
- Benchmark/evaluation report with raw sample counts and honest limitations.

**Exit gate:** a reviewer can run the project from a clean machine and reproduce the complete workflow without undocumented manual repair.

---

## 15. Release definition of done

- Clean setup instructions work on a fresh environment.
- Dataset source, licence and checksum are preserved.
- Data-contract failures block invalid training/inference.
- Training run provenance is complete and queryable.
- Baseline and candidate are evaluated on the same chronological test window.
- MLflow records run/model lineage and artifacts.
- Exact model version is staged and smoke-tested.
- Explicit approval is required for production promotion.
- Prediction requests identify the serving model version.
- Data drift, prediction drift and ground-truth quality are separate reports.
- A deterministic incident can be demonstrated and explained.
- Rollback restores a known-good model and is audited.
- Tests and CI pass; skipped checks are disclosed.
- README, diagrams, evaluation and interview docs are accurate.
- No unsupported claims such as “production-grade enterprise MLOps” or “real-time drift prevention”.

---

## 16. Resume and interview positioning — only after implementation

### One-line description

**ModelGuard — a production ML reliability control plane that versions datasets, validates data contracts, tracks reproducible training, governs model releases and monitors deployed inference for drift and performance degradation.**

### 30-second explanation

“I built ModelGuard to address the gap between training a model and operating it reliably. It registers immutable dataset versions, validates data contracts, tracks reproducible training runs and model lineage, and requires explicit evaluation and approval gates before deployment. Once a model is serving, it records inference telemetry, detects input and prediction drift, evaluates delayed ground-truth labels and supports audited rollback to a known-good model. I evaluate the workflow using public UCI datasets and deterministic failure-injection scenarios.”

### Resume bullet templates — fill only with measured results

- “Built a Python/FastAPI MLOps control plane integrating PostgreSQL, MLflow and object storage to version datasets, track reproducible training runs and govern model releases.”
- “Implemented schema validation, drift monitoring and delayed-label model-quality evaluation; detected **[N of M]** injected failure scenarios with **[measured precision/recall]**.”
- “Designed model promotion and rollback gates with exact-version lineage and audit records; verified recovery across **[N]** deployment scenarios.”

Never use bracketed metrics until experiments have been executed and the raw evidence is recorded.

---

## 17. Research and implementation references

### Public dataset sources
- UCI Metro Interstate Traffic Volume: https://archive.ics.uci.edu/dataset/492/urban+traffic
- UCI Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
- UCI Appliances Energy Prediction: https://archive.ics.uci.edu/dataset/374/appliancesenergyprediction
- UCI Bike Sharing: https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset
- UCI Air Quality (research-use-only restriction): https://archive.ics.uci.edu/dataset/387/air+quality

### Research benchmark context
- MLOps-Bench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1402/
- MLOps-Bench preprint: https://arxiv.org/abs/2610.23506

### Technical documentation
- MLflow Tracking: https://mlflow.org/docs/latest/tracking/
- MLflow Model Registry: https://mlflow.org/docs/latest/registry/
- MLflow Registry Workflows: https://mlflow.org/docs/latest/ml/model-registry/workflow/
- Pandera DataFrame Schemas: https://pandera.readthedocs.io/en/stable/dataframe_schemas.html
- Evidently: https://github.com/evidentlyai/evidently
- Prefect (reference only; not selected for initial queue): https://www.prefect.io/how-it-works
- Prometheus Instrumentation: https://prometheus.io/docs/practices/instrumentation/

---

## 18. Final engineering principles

1. Evidence over claims: every status and metric must be backed by stored results.
2. Reproducibility over convenience: record exact data, code, environment and model references.
3. Deterministic enforcement over model-generated decisions: AI/ML outputs never bypass release policy.
4. Drift is a signal, not a diagnosis: measure actual quality when labels become available.
5. Safe deployment over automation theatre: promotion and rollback require explicit, auditable state transitions.
6. Modular monolith over premature microservices.
7. CPU-first: the project must run on a typical developer machine; no large GPU training requirement.
8. Add tools only to solve an actual requirement; keep versions pinned and verify documentation at implementation time.
9. Preserve the parent blueprint's ModelGuard scope; request approval before scope or architecture expansion.
10. Report completed, tested, measured and planned work separately.
