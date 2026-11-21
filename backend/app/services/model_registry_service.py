import math
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.models
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import EntityNotFoundError, InvalidOperationError, ModelRegistryError
from app.core.logging import logger
from app.db import models
from app.schemas.models import GateEvaluationResult, ReleaseGatePolicy


def utc_now() -> datetime:
    return datetime.now(UTC)


class ModelRegistryService:
    """Service managing MLflow Model Registry integration, candidate tracking, and release gates."""

    def __init__(self, db: Session, tracking_uri: str | None = None) -> None:
        self.db = db
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI

    def _get_client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.tracking_uri)

    def register_candidate(
        self,
        training_run_id: str,
        model_name: str = "metro-traffic-volume",
    ) -> models.ModelCandidate:
        """Register the trained candidate model from an immutable TrainingRun in the MLflow Model Registry."""
        run = self.db.query(models.TrainingRun).filter(models.TrainingRun.id == training_run_id).first()
        if not run:
            raise EntityNotFoundError(f"TrainingRun '{training_run_id}' not found.")

        if run.status != "SUCCEEDED":
            raise InvalidOperationError(
                f"Cannot register model from TrainingRun '{training_run_id}' with status '{run.status}'. Expected 'SUCCEEDED'."
            )

        if not run.mlflow_run_id:
            raise ModelRegistryError(f"TrainingRun '{training_run_id}' does not have an associated MLflow run ID.")

        client = self._get_client()

        # 1. Ensure Registered Model exists in MLflow
        try:
            client.get_registered_model(model_name)
        except Exception:
            try:
                client.create_registered_model(
                    model_name,
                    description=f"ModelGuard registered candidate model for task {run.dataset_version.dataset.name if run.dataset_version and run.dataset_version.dataset else 'metro-traffic'}",
                )
            except Exception as e:
                logger.debug(f"Registered model creation note: {e}")

        # 2. Register candidate_model artifact (preserve baseline in tracking only)
        artifact_source = f"runs:/{run.mlflow_run_id}/candidate_model"
        try:
            model_version_obj = client.create_model_version(
                name=model_name,
                source=artifact_source,
                run_id=run.mlflow_run_id,
                description=f"Candidate model registered from ModelGuard TrainingRun {run.id}",
                tags={
                    "modelguard.training_run_id": run.id,
                    "modelguard.dataset_version_id": run.dataset_version_id,
                },
            )
            exact_version = str(model_version_obj.version)
        except Exception as e:
            raise ModelRegistryError(f"Failed to create model version in MLflow Model Registry: {e}") from e

        # 3. Extract signature and input schema
        signature_dict: dict[str, Any] | None = None
        try:
            model_info = mlflow.models.get_model_info(artifact_source)
            if model_info.signature:
                signature_dict = model_info.signature.to_dict()
        except Exception as e:
            logger.warning(f"Could not extract model signature from MLflow: {e}")

        # Authoritative tabular input contract specification
        input_contract = {
            "required_features": [
                {"name": "holiday", "dtype": "str", "nullable": False},
                {"name": "temp", "dtype": "float", "nullable": True},
                {"name": "rain_1h", "dtype": "float", "nullable": True},
                {"name": "snow_1h", "dtype": "float", "nullable": False},
                {"name": "clouds_all", "dtype": "int", "nullable": False},
                {"name": "weather_main", "dtype": "str", "nullable": False},
                {"name": "weather_description", "dtype": "str", "nullable": False},
                {"name": "date_time", "dtype": "str", "nullable": False},
            ],
            "target": "traffic_volume",
        }

        model_uri = f"models:/{model_name}/{exact_version}"

        candidate = models.ModelCandidate(
            training_run_id=run.id,
            dataset_version_id=run.dataset_version_id,
            mlflow_model_name=model_name,
            mlflow_model_version=exact_version,
            model_uri=model_uri,
            artifact_uri=run.artifact_uri or artifact_source,
            signature_json=signature_dict,
            input_contract_json=input_contract,
            metrics_json=run.metrics_json,
            status="REGISTERED",
        )
        self.db.add(candidate)
        self.db.flush()

        audit = models.AuditEvent(
            actor="system",
            action="REGISTER_MODEL_CANDIDATE",
            resource_type="model_candidate",
            resource_id=candidate.id,
            before_state_json=None,
            after_state_json={
                "candidate_id": candidate.id,
                "mlflow_model_name": model_name,
                "mlflow_model_version": exact_version,
                "model_uri": model_uri,
                "status": "REGISTERED",
            },
            reason=f"Registered candidate model version {exact_version} from TrainingRun {run.id}",
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(candidate)

        logger.info(f"Registered ModelCandidate '{candidate.id}' (MLflow model '{model_name}' v{exact_version}).")
        return candidate

    def evaluate_release_gate(
        self,
        candidate_id: str,
        policy: ReleaseGatePolicy | None = None,
    ) -> GateEvaluationResult:
        """Evaluate candidate model against versioned release policy using fail-closed semantics."""
        policy = policy or ReleaseGatePolicy()
        candidate = self.db.query(models.ModelCandidate).filter(models.ModelCandidate.id == candidate_id).first()
        if not candidate:
            raise EntityNotFoundError(f"ModelCandidate '{candidate_id}' not found.")

        criteria_results: dict[str, bool] = {}
        rejection_reasons: list[str] = []
        evaluated_metrics: dict[str, Any] = {}

        # 1. Dataset validation check
        val_runs = (
            self.db.query(models.ValidationRun)
            .filter(models.ValidationRun.dataset_version_id == candidate.dataset_version_id)
            .all()
        )
        passed_validation = any(vr.status == "PASSED" for vr in val_runs)
        criteria_results["dataset_validation_passed"] = passed_validation
        if not passed_validation and policy.require_dataset_validation:
            rejection_reasons.append("DatasetVersion has no passing ValidationRun.")

        # 2. Training execution check
        train_run = candidate.training_run
        train_succeeded = train_run is not None and train_run.status == "SUCCEEDED"
        criteria_results["training_succeeded"] = train_succeeded
        if not train_succeeded and policy.require_training_succeeded:
            rejection_reasons.append("Underlying TrainingRun status is not SUCCEEDED.")

        # 3. Metrics availability check
        metrics = candidate.metrics_json or {}
        has_required_metrics = all(
            k in metrics for k in ("candidate_val_mae", "candidate_val_rmse", "candidate_val_r2")
        )
        criteria_results["required_metrics_present"] = has_required_metrics
        if not has_required_metrics:
            rejection_reasons.append(
                "Evaluation metrics missing candidate_val_mae, candidate_val_rmse, or candidate_val_r2."
            )
        else:
            val_mae = float(metrics["candidate_val_mae"])
            val_rmse = float(metrics["candidate_val_rmse"])
            val_r2 = float(metrics["candidate_val_r2"])

            evaluated_metrics["candidate_val_mae"] = val_mae
            evaluated_metrics["candidate_val_rmse"] = val_rmse
            evaluated_metrics["candidate_val_r2"] = val_r2

            # 4. Candidate metric thresholds
            mae_ok = val_mae <= policy.max_val_mae
            rmse_ok = val_rmse <= policy.max_val_rmse
            if math.isnan(val_r2):
                r2_ok = policy.min_val_r2 <= -99.0
            else:
                r2_ok = val_r2 >= policy.min_val_r2

            criteria_results["max_val_mae_met"] = mae_ok
            criteria_results["max_val_rmse_met"] = rmse_ok
            criteria_results["min_val_r2_met"] = r2_ok

            if not mae_ok:
                rejection_reasons.append(
                    f"Candidate val MAE ({val_mae:.2f}) exceeds policy limit ({policy.max_val_mae:.2f})."
                )
            if not rmse_ok:
                rejection_reasons.append(
                    f"Candidate val RMSE ({val_rmse:.2f}) exceeds policy limit ({policy.max_val_rmse:.2f})."
                )
            if not r2_ok:
                if math.isnan(val_r2):
                    rejection_reasons.append("Candidate val R² is NaN (undefined due to sample size).")
                else:
                    rejection_reasons.append(
                        f"Candidate val R² ({val_r2:.4f}) below policy minimum ({policy.min_val_r2:.4f})."
                    )

            # 5. Baseline comparison check
            baseline_mae = metrics.get("baseline_val_mae")
            if baseline_mae is not None and float(baseline_mae) > 0:
                base_mae = float(baseline_mae)
                evaluated_metrics["baseline_val_mae"] = base_mae
                improvement_pct = ((base_mae - val_mae) / base_mae) * 100.0
                evaluated_metrics["baseline_improvement_pct"] = improvement_pct
                baseline_ok = improvement_pct >= policy.min_mae_improvement_pct
                criteria_results["beats_baseline"] = baseline_ok
                if not baseline_ok:
                    rejection_reasons.append(
                        f"Candidate MAE improvement over baseline ({improvement_pct:.1f}%) is below required {policy.min_mae_improvement_pct:.1f}%."
                    )
            else:
                criteria_results["beats_baseline"] = False
                rejection_reasons.append("Baseline validation MAE missing or non-positive; cannot compute improvement.")

        # 6. Current champion comparison check (if champion exists)
        prod_env = (
            self.db.query(models.DeploymentEnvironment)
            .filter(models.DeploymentEnvironment.name == "production")
            .first()
        )
        if prod_env and prod_env.active_candidate_id and prod_env.active_candidate_id != candidate.id:
            champion = prod_env.active_candidate
            if champion and champion.metrics_json and "candidate_val_mae" in champion.metrics_json:
                champ_mae = float(champion.metrics_json["candidate_val_mae"])
                evaluated_metrics["champion_val_mae"] = champ_mae
                max_allowed_mae = champ_mae * (1.0 + (policy.max_regression_pct_vs_champion / 100.0))
                champ_ok = val_mae <= max_allowed_mae
                criteria_results["champion_regression_within_tolerance"] = champ_ok
                if not champ_ok:
                    rejection_reasons.append(
                        f"Candidate val MAE ({val_mae:.2f}) regresses by more than {policy.max_regression_pct_vs_champion}% against champion ({champ_mae:.2f})."
                    )

        # 7. Model artifact load test
        load_ok = False
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            loaded = mlflow.sklearn.load_model(candidate.model_uri)
            load_ok = loaded is not None and hasattr(loaded, "predict")
            criteria_results["model_artifact_loadable"] = load_ok
        except Exception as e:
            criteria_results["model_artifact_loadable"] = False
            rejection_reasons.append(f"Model artifact could not be loaded from '{candidate.model_uri}': {e}")

        # Gate decision
        passed = len(rejection_reasons) == 0 and all(criteria_results.values())

        candidate.status = "GATE_PASSED" if passed else "GATE_FAILED"
        result = GateEvaluationResult(
            passed=passed,
            policy_version=policy.version,
            evaluated_metrics=evaluated_metrics,
            criteria_results=criteria_results,
            rejection_reasons=rejection_reasons,
        )
        candidate.gate_evaluation_json = result.model_dump()
        self.db.commit()

        audit = models.AuditEvent(
            actor="system",
            action="EVALUATE_RELEASE_GATE",
            resource_type="model_candidate",
            resource_id=candidate.id,
            before_state_json=None,
            after_state_json={
                "status": candidate.status,
                "passed": passed,
                "rejection_reasons": rejection_reasons,
                "policy_version": policy.version,
            },
            reason=f"Release gate evaluation {'PASSED' if passed else 'FAILED'} for candidate {candidate.id}",
        )
        self.db.add(audit)
        self.db.commit()

        logger.info(
            f"Release gate evaluation for ModelCandidate '{candidate.id}' finished: {'PASSED' if passed else 'FAILED'}."
        )
        return result

    def get_candidate(self, candidate_id: str) -> models.ModelCandidate:
        candidate = self.db.query(models.ModelCandidate).filter(models.ModelCandidate.id == candidate_id).first()
        if not candidate:
            raise EntityNotFoundError(f"ModelCandidate '{candidate_id}' not found.")
        return candidate

    def list_candidates(
        self,
        status: str | None = None,
        dataset_version_id: str | None = None,
    ) -> list[models.ModelCandidate]:
        query = self.db.query(models.ModelCandidate)
        if status:
            query = query.filter(models.ModelCandidate.status == status)
        if dataset_version_id:
            query = query.filter(models.ModelCandidate.dataset_version_id == dataset_version_id)
        return query.order_by(models.ModelCandidate.created_at.desc()).all()
