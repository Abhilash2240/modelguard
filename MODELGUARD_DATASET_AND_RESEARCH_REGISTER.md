# ModelGuard — Dataset and Research Register

**Version:** 1.0  
**Prepared:** 21 November 2026  
**Purpose:** Evidence register for PRD dataset selection, attribution, operational experiments and MLOps design.

---

## 1. Recommended dataset lineup

| Priority | Dataset | UCI ID | Size / task | Licence | Planned use |
|---|---|---:|---|---|---|
| Primary | Metro Interstate Traffic Volume | 492 | 48,204 rows, 8 features, hourly regression, 2012–2018 | CC BY 4.0 | Main reproducible MLOps lifecycle and temporal monitoring demo |
| Secondary | Bank Marketing (`bank-additional-full`) | 222 | 41,188 rows, 20 inputs, date-ordered binary classification | CC BY 4.0 | Classification pipeline, thresholding, PR-AUC and model registry flexibility |
| Additional | Appliances Energy Prediction | 374 | 19,735 rows, 28 features, 10-minute temporal regression | CC BY 4.0 | Sensor-like features, temporal validation, feature relevance and quality tests |
| Optional | Bike Sharing | 275 | 17,389 rows, 13 features, hourly/daily regression | CC BY 4.0 | Secondary regression demo and season/weather-related feature distribution changes |
| Research-only | Air Quality | 387 | 9,358 hourly sensor observations, regression | Research only; commercial use excluded by UCI page | Optional academic sensor-drift exercise only; not default public demo |

### Why Metro Interstate Traffic Volume is the primary dataset

It has a clear timestamp, hourly observations across multiple years, 48,204 rows, eight predictor features, a numeric traffic-volume target, weather context and no missing values according to UCI. UCI states CC BY 4.0. It is small enough for CPU training but has genuine temporal structure for chronological evaluation and monitoring windows.

### Why Bank Marketing is secondary

It adds a classification workflow with a different metric family and data types. UCI describes multiple variants; `bank-additional-full.csv` contains 41,188 examples and 20 inputs ordered by date from May 2008 to November 2010. The 45,211-row `bank-full` variant is the older version with 17 inputs. The `duration` variable must be excluded for pre-call prediction because it is only known after the call; otherwise the experiment risks target leakage.

### Why Appliances Energy Prediction is additional

It provides 19,735 10-minute observations over around 4.5 months, a timestamp and multiple temperature/humidity sensor variables. UCI notes two random variables (`rv1`, `rv2`) were included for testing regression models and filtering non-predictive attributes. It can support feature-quality and temporal drift exercises.

### Why not make Air Quality the default dataset

UCI documents sensor/concept drift and missing values, but explicitly says commercial use is excluded and use is exclusively for research. ModelGuard's default public demo should use CC BY 4.0 datasets instead. Air Quality can be used only in a research-labelled test with the restriction respected.

---

## 2. Dataset-specific operational design

### 2.1 Metro Interstate Traffic Volume

**Source:** https://archive.ics.uci.edu/dataset/492/urban+traffic  
**Citation:** Hogue, J. (2019). Metro Interstate Traffic Volume. UCI Machine Learning Repository. DOI: 10.24432/C5X60B.  
**Licence:** CC BY 4.0; preserve attribution and licence details in README and dataset manifest.

Proposed task:
- Regression target: `traffic_volume`.
- Predictors: holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description and date_time-derived features.
- Time-based split: chronological train/validation/test; exact boundaries must be selected after inspecting the actual dataset and stored in the run manifest.
- Candidate models: baseline regressor and a compact tree-based estimator inside a scikit-learn Pipeline.
- Metrics: MAE, RMSE and R², with sample counts and residual plots.
- Monitoring: current input windows vs training/reference window; prediction distribution; delayed actual traffic values linked to inference IDs; rolling MAE/RMSE.

Cautions:
- Derive hour/day/month features only from the timestamp available at prediction time.
- Do not introduce target lag features unless a correct online history/state implementation is included.
- Do not shuffle before the final temporal evaluation.
- A feature distribution change is not proof that traffic prediction errors increased.

### 2.2 Bank Marketing

**Source:** https://archive.ics.uci.edu/dataset/222/bank+marketing  
**Citation:** Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. DOI: 10.24432/C5K306.  
**Licence:** CC BY 4.0; preserve attribution.

Proposed task:
- Classification target: `y` (term-deposit subscription).
- Use the ordered `bank-additional-full.csv` variant for a time-ordered holdout after confirming file ordering.
- Baseline: DummyClassifier.
- Candidate: categorical/numeric preprocessing + compact tree classifier.
- Metrics: PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix and prevalence.
- Tune classification threshold only on validation data.

Leakage and usage cautions:
- Exclude `duration` from a pre-call decision scenario: call duration is not available before initiating the call and is strongly tied to the completed interaction.
- Keep this as a public educational benchmark, not an operational banking decision system.
- Record the exact source variant and original file checksum because UCI exposes multiple versions.

### 2.3 Appliances Energy Prediction

**Source:** https://archive.ics.uci.edu/dataset/374/appliancesenergyprediction  
**Citation:** Candanedo, L. (2017). Appliances Energy Prediction. UCI Machine Learning Repository. DOI: 10.24432/C5VC8G.  
**Licence:** CC BY 4.0; preserve attribution.

Proposed task:
- Regression target: `Appliances` (Wh).
- Timestamp: `date`.
- Features include `lights`, room temperature/humidity values, outside weather variables, and `rv1` / `rv2` random variables.
- Use a chronological split.
- Use this to test whether schema/data validation and feature lineage preserve or identify non-predictive columns.

Cautions:
- Only around 4.5 months are covered; do not claim broad seasonal generalization.
- Sensor values are correlated and the experiment should report limitations.

### 2.4 Bike Sharing

**Source:** https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset  
**Citation:** Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. DOI: 10.24432/C5W894.  
**Licence:** CC BY 4.0; preserve attribution.

Proposed task:
- Regression target: hourly/daily bike rental count.
- Inputs include timestamp/calendar/weather features.
- Use time-aware splits, not random shuffling.
- Note the dataset includes `casual`, `registered` and `cnt` fields; if the target is `cnt`, exclude `casual` and `registered` if those values would not be available at inference time, to avoid target leakage through their arithmetic relationship.

---

## 3. Synthetic fault-injection catalogue

Public datasets are for benchmark-like ML evaluation. Deterministic synthetic fixtures are for testing system behaviour. Keep these separate in reports.

| ID | Scenario | Injection | Expected system response |
|---|---|---|---|
| FI-01 | Missing feature | Remove a required input column | Contract fails; inference/training blocked |
| FI-02 | Renamed column | Rename a schema field | Compatibility error with expected/observed schema |
| FI-03 | Dtype change | Numeric column becomes string | Validation blocks or explicitly coerces only if policy permits |
| FI-04 | Unknown category | Add an unseen category | Defined unknown-category behaviour; warning or rejection by contract |
| FI-05 | Missingness spike | Inject nulls into required fields | Quality report identifies column and rate |
| FI-06 | Duplicate rows | Duplicate a batch | Duplicate check reports count/rate; policy decides block/warn |
| FI-07 | Out-of-range values | Inject impossible temperatures/counts | Domain validation fails |
| FI-08 | Timestamp corruption | Invalid, future or out-of-order timestamps | Temporal validation reports invalid records |
| FI-09 | Numeric feature drift | Shift mean/variance or distribution | Drift report identifies affected features/window |
| FI-10 | Categorical drift | Change category frequencies | Drift report identifies changed categories |
| FI-11 | Prediction drift | Change input population so outputs shift | Prediction-distribution report; no claim of model degradation without labels |
| FI-12 | Delayed labels | Delay or omit feedback | Label coverage and quality calculation are marked incomplete |
| FI-13 | Concept/performance shift | Change target relationship in a controlled stream | Delayed-label quality metrics detect actual error increase |
| FI-14 | Corrupt artifact | Truncate or modify artifact bytes | Checksum/load test fails; candidate cannot be staged |
| FI-15 | Pipeline timeout | Force a job to exceed time budget | Job moves to explicit timeout/failure state |
| FI-16 | Staging smoke failure | Candidate cannot load or predict | Promotion blocked; active production model unchanged |
| FI-17 | Approval rejected | Reviewer rejects candidate | No production alias/pointer change |
| FI-18 | Rollback | Switch to last known-good model | Exact prior model version is active and verified |

Every scenario manifest should store:
- scenario ID and description;
- random seed;
- source dataset/version hash;
- transformation code/version;
- reference and current time windows;
- expected detector/result;
- expected system response;
- actual run/report IDs.

---

## 4. Benchmark research reference: MLOps-Bench

**Official paper:** https://aclanthology.org/2026.findings-emnlp.1402/  
**Preprint:** https://arxiv.org/abs/2610.23506

The official ACL Anthology record describes MLOps-Bench as a repository-grounded benchmark for production ML engineering. It contains 212 validated tasks from 31 public ML repositories, with 636 fail-to-pass tests and 856 pass-to-pass tests. It covers data pipelines, feature engineering, model training, model serving, monitoring, CI/CD, governance and end-to-end integration.

How ModelGuard should use this research:
- Treat lifecycle operations as executable engineering workflows, not merely a model score.
- Test both intended behaviour (fail-to-pass) and preservation of existing behaviour (pass-to-pass).
- Build local fault scenarios with clear oracles before considering external benchmark tasks.
- Do not claim equivalence to or superiority over MLOps-Bench based on a small local test suite.

MLOps-Bench is a benchmark for ML engineering repository tasks; it is not one of ModelGuard's training datasets.

---

## 5. Tooling research and design decisions

### MLflow

Official docs:
- Tracking: https://mlflow.org/docs/latest/tracking/
- Registry: https://mlflow.org/docs/latest/registry/
- Registry workflows: https://mlflow.org/docs/latest/ml/model-registry/workflow/

Use for experiment tracking, parameters, metrics, artifacts, dataset association, model signatures, model versions, aliases and lineage. MLflow's current documentation describes model aliases for mutable named references such as a `champion` model. Do not build a new implementation around legacy fixed model stages.

The MLflow tracking documentation identifies a backend store for run metadata and an artifact store for large artifacts. A database-backed store is required for the self-hosted model registry. ModelGuard should use PostgreSQL for durable MLflow metadata and MinIO for local S3-compatible artifacts, with deployment/approval state separately stored by ModelGuard.

### Pandera

Official docs: https://pandera.readthedocs.io/en/stable/dataframe_schemas.html

Use for dataframe schema definitions, required columns, types, nullability and custom checks. Preserve ModelGuard's own contract versions and validation records so another validator can be adopted later.

### Evidently

Project: https://github.com/evidentlyai/evidently

Use as a replaceable implementation for data/model evaluation reports and drift checks. Evidently describes support for data/ML metrics, reports and test suites. Do not equate a drift report with model performance degradation; use delayed labels for performance measurement.

### Prometheus

Official instrumentation guidance: https://prometheus.io/docs/practices/instrumentation/

Instrument online-serving, batch-processing and worker subsystems. Start with request count, errors, latency, last successful training/monitoring run, queue/worker status and resource-related metrics. Avoid unbounded/high-cardinality labels such as raw user IDs or request IDs.

### Prefect

Reference: https://www.prefect.io/how-it-works

Prefect offers observable flow/task execution and retry support. It is not selected for the first implementation because Redis + RQ is the simpler initial job queue. Do not deploy both Prefect and RQ in the MVP. Re-evaluate orchestration only if multi-step scheduling/retry complexity exceeds the lightweight worker design.

---

## 6. Attribution template for repository

Include a dataset notice in `docs/DATASETS.md` and the README, for example:

> Metro Interstate Traffic Volume — Hogue, J. (2019), UCI Machine Learning Repository, DOI: 10.24432/C5X60B, licensed under CC BY 4.0. Used for reproducible model-training and MLOps workflow demonstrations. The original dataset and its licence remain with their respective rights holders.

Add corresponding attribution blocks for every other dataset actually shipped or used in experiments. Do not state that the complete application is licensed under the dataset's licence; the dataset and project source code have separate licensing.
