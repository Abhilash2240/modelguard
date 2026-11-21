# ModelGuard — Interactive Demonstration Script

This demonstration script provides a step-by-step walkthrough to present ModelGuard live in an engineering interview or stakeholder demo.

---

## Preparation (2 Minutes)

1. Launch backend in Terminal 1:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. Launch frontend in Terminal 2:
   ```bash
   cd frontend
   npm run dev
   ```
3. Open your browser to `http://localhost:5173`.

---

## 10-Step Operational Walkthrough

### Step 1: System Overview (`/` Overview Page)
- **Goal:** Show real-time operational health.
- **Narrative:**
  > *"Here is the ModelGuard Control Plane Overview. At a glance, the engineering team sees the system health status, the current active production champion model, the candidate in staging awaiting approval, the count of open alerts, and key operational metrics."*

### Step 2: Dataset Lineage & Integrity (`/datasets`)
- **Goal:** Demonstrate immutable snapshots and contract validation.
- **Narrative:**
  > *"Before any training occurs, data must be fingerprinted. Notice our registered UCI Metro Traffic dataset snapshot v1.0.0 with its cryptographic SHA-256 digest and Pandera validation status: PASSED with 0 blocking errors."*

### Step 3: Reproducible Training & MLflow Tracking (`/training`)
- **Goal:** Show deterministic 1:1 training job orchestration.
- **Narrative:**
  > *"Every training job is executed through a 15-stage transactional lifecycle. In this table, we see our candidate model achieving 284 MAE—an 83.7% improvement over the 1,745 MAE baseline—with seed locked to 42 and direct link to the MLflow tracking run."*

### Step 4: Model Registry & Release Gate (`/models`)
- **Goal:** Show quantitative release gating before deployment.
- **Narrative:**
  > *"Here in the Model Registry, models aren't deployed blindly. The Release Gate evaluated candidate metrics against hard thresholds: MAE ≤ 350, RMSE ≤ 550, and R² ≥ 0.90. The candidate passed all gates and is authorized for staging."*

### Step 5: Staging Smoke Checks (`/deployments`)
- **Goal:** Demonstrate automated smoke testing before human approval.
- **Narrative:**
  > *"When staged, ModelGuard immediately runs an automated smoke test loading the exact serialized pipeline and verifying non-finite outputs. Latency was measured at 12ms. The deployment is now held in AWAITING_APPROVAL."*

### Step 6: Human Sign-Off & Promotion (`/deployments`)
- **Goal:** Complete two-phase human-in-the-loop promotion.
- **Action:** Click **"Approve"**, enter actor `lead_ml_engineer` and rationale `"Metrics verified on holdout partition"`, then click **"Promote to Production"**.
- **Narrative:**
  > *"Notice how governance is enforced: a human must provide explicit sign-off. Promotion atomically shifts the MLflow alias 'champion' to our new candidate."*

### Step 7: Low-Latency Inference Serving (`/inference`)
- **Goal:** Real-time prediction and schema validation.
- **Action:** Click **"Execute Prediction"** with the sample weather/traffic payload.
- **Narrative:**
  > *"The active champion serves predictions in ~3.5ms. If we modify the input to violate schema (e.g. invalid weather category), the request is rejected before reaching the model. Notice the privacy guarantee: raw features are stripped from telemetry."*

### Step 8: Delayed Ground-Truth Feedback (`/inference`)
- **Goal:** Idempotent ground-truth submission.
- **Action:** In the Delayed Feedback section, submit an actual traffic volume of `3150` for the inference ID.
- **Narrative:**
  > *"When induction loop roadside sensor actuals arrive, we submit ground truth. The submission is strictly idempotent on duplicate values and flags conflicts if contradictory actuals are provided."*

### Step 9: Statistical Drift Detection (`/monitoring`)
- **Goal:** Kolmogorov-Smirnov test and PSI drift reports.
- **Action:** View the input covariate drift report.
- **Narrative:**
  > *"ModelGuard compares real-time input distributions against the baseline snapshot. When temperature shifted +30K, the two-sample KS test detected a p-value < 10⁻¹⁵ and automatically triggered a persistent Warning alert."*

### Step 10: Zero-Downtime Audited Rollback (`/deployments` & `/alerts`)
- **Goal:** Immediate remediation with full audit trail.
- **Action:** On the Deployments page, click **"Rollback"** on the production deployment.
- **Narrative:**
  > *"In an emergency, the on-call engineer triggers a rollback. ModelGuard atomically points the MLflow 'champion' alias back to the previous stable version in under 50ms without container restart or downtime, logging an immutable audit event."*
