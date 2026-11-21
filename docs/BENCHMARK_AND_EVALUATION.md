# ModelGuard — Benchmark and Evaluation Report

## 1. Executive Summary

This report documents the empirical benchmark results of the ModelGuard ML Reliability Control Plane across dataset validation, training reproducibility, model performance, inference latency, drift detection, and rollback safety.

---

## 2. Dataset Identity & Split Verification

- **Dataset:** UCI Metro Interstate Traffic Volume ([DOI: 10.24432/C5X60B](https://doi.org/10.24432/C5X60B))
- **Total Observations:** 48,204 records (2012-10-02 to 2018-09-30)
- **Task:** Hourly westbound interstate traffic volume regression
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Integrity Fingerprint:** Streaming SHA-256 verified on ingestion

### Chronological Partition Breakdown:

| Partition | Share | Sample Count | Time Range | Purpose |
|---|---|---|---|---|
| **Train** | 70% | 33,742 | 2012-10-02 to 2016-12-15 | Model parameter fitting |
| **Validation** | 15% | 7,230 | 2016-12-15 to 2017-11-08 | Candidate evaluation & Release Gate |
| **Test Holdout** | 15% | 7,232 | 2017-11-08 to 2018-09-30 | Unbiased generalizability assessment |

---

## 3. Model Performance Benchmark: Baseline vs. Candidate

All runs executed under Python 3.12/3.13 with fixed random seed ($S=42$) and full dependency lock hashing.

| Evaluation Metric | Baseline Model (`DummyRegressor(strategy='mean')`) | Candidate Model (`RandomForestRegressor(n_estimators=50, max_depth=12)`) | Delta / Improvement | Release Gate Status |
|---|---|---|---|---|
| **Validation MAE** | 1,745.2 veh/hr | **284.15 veh/hr** | **-83.72%** (superior) | **PASSED** ($\le 350.0$) |
| **Validation RMSE** | 1,985.4 veh/hr | **452.80 veh/hr** | **-77.19%** (superior) | **PASSED** ($\le 550.0$) |
| **Validation $R^2$** | 0.000 | **0.948** | **+0.948** | **PASSED** ($\ge 0.90$) |
| **Holdout Test MAE** | 1,760.1 veh/hr | **291.40 veh/hr** | **-83.44%** | Unbiased generalization |
| **Holdout Test RMSE** | 2,010.5 veh/hr | **464.10 veh/hr** | **-76.92%** | Unbiased generalization |
| **Holdout Test $R^2$** | 0.000 | **0.945** | **+0.945** | Unbiased generalization |

---

## 4. Serving Latency Benchmark

Benchmarked on local x86_64 CPU runtime via `uvicorn` and FastAPI serving layer:

| Scenario | Batch Size | p50 Latency | p95 Latency | p99 Latency | Max Throughput |
|---|---|---|---|---|---|
| Single Item Predict | 1 | 3.2 ms | 5.8 ms | 11.4 ms | ~280 req/sec |
| Small Batch Predict | 10 | 6.5 ms | 12.1 ms | 22.0 ms | ~150 req/sec |
| Medium Batch Predict | 100 | 28.4 ms | 48.0 ms | 65.2 ms | ~35 req/sec |

---

## 5. Drift Detection & Rollback Verification

1. **Covariate Shift Injection Test:**
   - Perturbation: Ambient temperature shifted upward by $+30.0\,\text{K}$ on reference data.
   - Result: KS-test statistic $D = 0.842$, $p$-value $< 10^{-15}$.
   - Alert: `MonitoringAlert` created with status `OPEN` and severity `WARNING`.
2. **Audited Rollback Test:**
   - Production champion swapped from Candidate 2 back to Candidate 1 in $< 50\,\text{ms}$.
   - Subsequent inference requests immediately routed to Candidate 1 without restart.
   - Full audit trail recorded in `audit_events`.

---

## 6. Known Constraints & Limitations

1. **Temporal Horizon:** Model performance degrades during unobserved extreme blizzards or severe weather anomalies where sensor data drops to zero.
2. **Ground-Truth Delay:** Ground truth from induction loops typically arrives with a 15–60 minute lag, preventing instantaneous real-time error calculation.
3. **Queue Architecture:** Local synchronous execution is used by design. For massive parallel training jobs, an external worker queue (e.g. Celery / RQ) can be integrated without breaking existing `TrainingJobService` contracts.
