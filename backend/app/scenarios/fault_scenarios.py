from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.errors import InferenceError
from app.db import models
from app.services.drift_service import DriftAnalysisService
from app.services.feedback_service import FeedbackService
from app.services.inference_service import InferenceService


@dataclass
class ScenarioExecutionResult:
    scenario_name: str
    seed: int
    transformation_applied: str
    expected_detection: str
    expected_system_response: str
    actual_detected: bool
    evidence: dict[str, Any]
    limitations: str


class FeatureDistributionShiftScenario:
    """Scenario 1: Deterministic temperature distribution shift (extreme cold snap / heatwave)."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(self, db: Session, base_df: pd.DataFrame) -> ScenarioExecutionResult:
        np.random.seed(self.seed)
        df_perturbed = base_df.copy()
        # Add deterministic shift +25.0K to temperature
        df_perturbed["temp"] = df_perturbed["temp"] + 25.0 + np.random.normal(0, 1.0, len(df_perturbed))

        service = DriftAnalysisService(db)
        report = service.analyze_input_drift(
            current_df=df_perturbed,
            reference_df=base_df,
            model_name="metro-traffic-volume",
            model_version="1",
        )

        detected = report.is_drift_detected and "temp" in report.summary_json.get("drifting_features", [])
        return ScenarioExecutionResult(
            scenario_name="Feature Distribution Shift (Temperature Shock)",
            seed=self.seed,
            transformation_applied="Added deterministic +25.0K shock with Gaussian noise to 'temp' feature.",
            expected_detection="Kolmogorov-Smirnov test p-value < 0.05 and PSI > 0.25 on 'temp'.",
            expected_system_response="DATA_DRIFT MonitoringReport and WARNING/CRITICAL MonitoringAlert.",
            actual_detected=detected,
            evidence=report.summary_json,
            limitations="Synthetic demonstration. Does not reflect real-world model accuracy degradation without ground truth.",
        )


class MissingValueSpikeScenario:
    """Scenario 2: Sensor blackout causing missing values in clouds_all."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(self, db: Session, base_df: pd.DataFrame) -> ScenarioExecutionResult:
        np.random.seed(self.seed)
        df_perturbed = base_df.copy()
        # Randomly mask 40% of clouds_all to NaN
        mask = np.random.rand(len(df_perturbed)) < 0.40
        df_perturbed.loc[mask, "clouds_all"] = np.nan

        service = DriftAnalysisService(db)
        report = service.analyze_input_drift(
            current_df=df_perturbed,
            reference_df=base_df,
            model_name="metro-traffic-volume",
            model_version="1",
        )

        detected = report.is_drift_detected and "clouds_all" in report.summary_json.get("drifting_features", [])
        return ScenarioExecutionResult(
            scenario_name="Missing Value Spike (Sensor Drop)",
            seed=self.seed,
            transformation_applied="Masked 40% of 'clouds_all' values to NaN.",
            expected_detection="Missing-rate shift delta > 0.10 triggering drift detection.",
            expected_system_response="DATA_DRIFT MonitoringReport and MonitoringAlert.",
            actual_detected=detected,
            evidence=report.summary_json,
            limitations="Synthetic demonstration. Evaluates data-pipeline health, not intrinsic model degradation.",
        )


class InvalidCategoricalValuesScenario:
    """Scenario 3: Upstream data provider introduces unseen category 'AlienVortex'."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(self, db: Session, base_df: pd.DataFrame) -> ScenarioExecutionResult:
        df_perturbed = base_df.copy()
        # Inject unseen category
        df_perturbed.loc[0, "weather_main"] = "AlienVortex"

        service = DriftAnalysisService(db)
        report = service.analyze_input_drift(
            current_df=df_perturbed,
            reference_df=base_df,
            model_name="metro-traffic-volume",
            model_version="1",
        )

        detected = report.is_drift_detected and "weather_main" in report.summary_json.get("drifting_features", [])
        return ScenarioExecutionResult(
            scenario_name="Invalid Categorical Value Injection",
            seed=self.seed,
            transformation_applied="Injected unobserved categorical level 'AlienVortex' into 'weather_main'.",
            expected_detection="Total Variation Distance (TVD) shift on categorical frequency.",
            expected_system_response="DATA_DRIFT MonitoringReport and MonitoringAlert.",
            actual_detected=detected,
            evidence=report.summary_json,
            limitations="Synthetic injection. Demonstrates schema drift detection.",
        )


class PredictionDistributionShiftScenario:
    """Scenario 4: Macro demand shift alters model output distribution."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(self, db: Session) -> ScenarioExecutionResult:
        np.random.seed(self.seed)
        ref_preds = np.random.normal(loc=3000, scale=800, size=100)
        curr_preds = np.random.normal(loc=6500, scale=900, size=100)  # Heavy shift

        service = DriftAnalysisService(db)
        report = service.analyze_prediction_drift(
            current_predictions=curr_preds,
            reference_predictions=ref_preds,
            model_name="metro-traffic-volume",
            model_version="1",
        )

        detected = report.is_drift_detected
        return ScenarioExecutionResult(
            scenario_name="Prediction Distribution Shift",
            seed=self.seed,
            transformation_applied="Shifted prediction mean from ~3000 to ~6500.",
            expected_detection="Two-sample KS-test p-value < 0.05 and PSI > 0.25 on predicted values.",
            expected_system_response="PREDICTION_DRIFT MonitoringReport and PREDICTION_DRIFT Alert.",
            actual_detected=detected,
            evidence=report.summary_json,
            limitations="Evaluates prediction output shift independently of input features and ground truth.",
        )


class PerformanceDegradationScenario:
    """Scenario 5: Concept drift where delayed labels reveal catastrophic accuracy loss (MAE > 500)."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(self, db: Session) -> ScenarioExecutionResult:
        np.random.seed(self.seed)
        # Create 12 synthetic prediction events and matching diverging feedback
        feedback_service = FeedbackService(db)

        for i in range(12):
            inf_id = f"scenario5-inf-{self.seed}-{i}"
            pred_val = 3200.0 + float(np.random.normal(0, 50))
            # Delayed actuals diverge by > 800 units
            actual_val = pred_val + 750.0 + float(np.random.normal(0, 50))

            event = models.PredictionEvent(
                inference_id=inf_id,
                model_name="metro-traffic-volume",
                model_version="1",
                latency_ms=12.5,
                status="SUCCESS",
                prediction_json={"predictions": [pred_val], "count": 1},
                input_summary_json={"row_count": 1},
            )
            db.add(event)
            db.flush()

            feedback_service.submit_feedback(
                inference_id=inf_id,
                actual_value=actual_val,
                submitted_by="scenario_harness",
            )

        report = feedback_service.calculate_rolling_quality(
            model_name="metro-traffic-volume",
            model_version="1",
            min_samples=10,
            max_mae_threshold=450.0,
        )

        detected = report.status == "SUFFICIENT_LABELS" and (report.mae or 0.0) > 450.0
        return ScenarioExecutionResult(
            scenario_name="Concept Degradation With Delayed Labels",
            seed=self.seed,
            transformation_applied="Generated 12 prediction events where ground truth deviates by ~750 units.",
            expected_detection="Rolling MAE > 450.0 threshold evaluation.",
            expected_system_response="QUALITY_DEGRADATION Alert with CRITICAL severity.",
            actual_detected=detected,
            evidence={"mae": report.mae, "rmse": report.rmse, "r2": report.r2},
            limitations="Synthetic concept degradation. Demonstrates feedback reconciliation.",
        )


class ServingFailureScenario:
    """Scenario 6: Missing artifact or corrupt URI causing controlled serving failure and metric increment."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(self, db: Session) -> ScenarioExecutionResult:
        inf_service = InferenceService(db, tracking_uri="file:///invalid/path/to/nowhere")

        # Create dummy candidate with invalid URI
        cand = models.ModelCandidate(
            id=models.generate_uuid(),
            training_run_id=models.generate_uuid(),
            dataset_version_id=models.generate_uuid(),
            mlflow_model_name="metro-traffic-volume",
            mlflow_model_version="999",
            model_uri="models:/metro-traffic-volume/999",
            artifact_uri="file:///invalid/path/to/nowhere",
            status="PROMOTED",
        )
        db.add(cand)
        db.flush()

        failed_controlled = False
        try:
            inf_service._get_model(cand)
        except InferenceError:
            failed_controlled = True

        return ScenarioExecutionResult(
            scenario_name="Serving Failure on Corrupted Artifact URI",
            seed=self.seed,
            transformation_applied="Attempted model load against unreachable URI 'models:/metro-traffic-volume/999'.",
            expected_detection="InferenceError raised and model_load_failures_total metric incremented.",
            expected_system_response="Controlled HTTP 500 / InferenceError without crashing service.",
            actual_detected=failed_controlled,
            evidence={"model_uri": cand.model_uri, "load_failed": failed_controlled},
            limitations="Fault simulation for infrastructure resilience.",
        )
