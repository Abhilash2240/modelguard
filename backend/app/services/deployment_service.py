import time
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import (
    DeploymentError,
    EntityNotFoundError,
    InvalidOperationError,
    RollbackError,
)
from app.core.logging import logger
from app.db import models
from app.ml.features import prepare_features
from app.services.model_registry_service import ModelRegistryService


def utc_now() -> datetime:
    return datetime.now(UTC)


SAMPLE_SMOKE_INPUT = [
    {
        "holiday": "None",
        "temp": 288.15,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 40,
        "weather_main": "Clouds",
        "weather_description": "scattered clouds",
        "date_time": "2026-11-21 09:00:00",
    }
]


class DeploymentService:
    """Service orchestrating staging deployments, smoke testing, approvals, and exact-version promotion."""

    def __init__(self, db: Session, tracking_uri: str | None = None) -> None:
        self.db = db
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI

    def _get_client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.tracking_uri)

    def _get_or_create_environment(self, name: str) -> models.DeploymentEnvironment:
        env = self.db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == name).first()
        if not env:
            env = models.DeploymentEnvironment(name=name)
            self.db.add(env)
            self.db.flush()
        return env

    def stage_candidate(self, candidate_id: str) -> models.Deployment:
        """Stage an exact registered model version and execute representative smoke tests."""
        candidate = self.db.query(models.ModelCandidate).filter(models.ModelCandidate.id == candidate_id).first()
        if not candidate:
            raise EntityNotFoundError(f"ModelCandidate '{candidate_id}' not found.")

        # Ensure release gate has been evaluated and passed
        if candidate.status == "REGISTERED":
            reg_service = ModelRegistryService(self.db, self.tracking_uri)
            gate_res = reg_service.evaluate_release_gate(candidate_id)
            self.db.refresh(candidate)
            if not gate_res.passed:
                raise InvalidOperationError(
                    f"Candidate '{candidate_id}' failed release gate: {'; '.join(gate_res.rejection_reasons)}"
                )

        if candidate.status != "GATE_PASSED":
            raise InvalidOperationError(
                f"Candidate '{candidate_id}' cannot be staged with status '{candidate.status}'. Expected 'GATE_PASSED'."
            )

        staging_env = self._get_or_create_environment("staging")

        # Create Staging Deployment
        deployment = models.Deployment(
            environment_id=staging_env.id,
            candidate_id=candidate.id,
            previous_candidate_id=staging_env.active_candidate_id,
            status="STAGED",
        )
        self.db.add(deployment)
        self.db.flush()

        # Run smoke test on exact immutable model URI
        smoke_passed = False
        smoke_results: dict[str, Any] = {}
        try:
            t0 = time.perf_counter()
            mlflow.set_tracking_uri(self.tracking_uri)
            model = mlflow.sklearn.load_model(candidate.model_uri)
            model_loaded = model is not None and hasattr(model, "predict")

            df_sample = pd.DataFrame(SAMPLE_SMOKE_INPUT)
            X, _ = prepare_features(df_sample, target_col="traffic_volume")
            sample_preds = model.predict(X)
            smoke_latency = (time.perf_counter() - t0) * 1000.0

            output_finite = bool(np.all(np.isfinite(sample_preds)))
            smoke_passed = model_loaded and output_finite

            smoke_results = {
                "passed": smoke_passed,
                "model_loaded": model_loaded,
                "sample_input_valid": True,
                "sample_output_finite": output_finite,
                "latency_ms": smoke_latency,
                "sample_prediction": float(sample_preds[0]),
            }
        except Exception as e:
            smoke_passed = False
            smoke_results = {
                "passed": False,
                "model_loaded": False,
                "sample_input_valid": False,
                "sample_output_finite": False,
                "error": str(e),
            }

        deployment.smoke_test_results_json = smoke_results

        if smoke_passed:
            deployment.status = "AWAITING_APPROVAL"
            candidate.status = "AWAITING_APPROVAL"
            staging_env.active_candidate_id = candidate.id
            staging_env.updated_at = utc_now()
        else:
            deployment.status = "SMOKE_FAILED"
            candidate.status = "FAILED"
            self.db.commit()
            raise DeploymentError(
                f"Staging smoke checks failed: {smoke_results.get('error', 'Non-finite predictions')}"
            )

        audit = models.AuditEvent(
            actor="system",
            action="STAGE_MODEL_CANDIDATE",
            resource_type="deployment",
            resource_id=deployment.id,
            before_state_json=None,
            after_state_json={
                "deployment_id": deployment.id,
                "candidate_id": candidate.id,
                "environment": "staging",
                "status": deployment.status,
                "smoke_results": smoke_results,
            },
            reason=f"Staged candidate version {candidate.mlflow_model_version} with smoke test validation",
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(deployment)

        logger.info(
            f"Staged ModelCandidate '{candidate.id}' in deployment '{deployment.id}' (status: {deployment.status})."
        )
        return deployment

    def approve_deployment(
        self,
        deployment_id: str,
        actor: str,
        rationale: str,
    ) -> models.Deployment:
        """Explicit human approval for a staged model version."""
        deployment = self.db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
        if not deployment:
            raise EntityNotFoundError(f"Deployment '{deployment_id}' not found.")

        if deployment.status != "AWAITING_APPROVAL":
            raise InvalidOperationError(
                f"Deployment '{deployment_id}' with status '{deployment.status}' cannot be approved. Expected 'AWAITING_APPROVAL'."
            )

        smoke_res = deployment.smoke_test_results_json or {}
        if not smoke_res.get("passed"):
            raise InvalidOperationError(
                f"Deployment '{deployment_id}' cannot be approved because staging smoke check did not pass."
            )

        now = utc_now()
        deployment.status = "APPROVED"
        deployment.approved_by = actor
        deployment.approved_at = now
        deployment.approval_rationale = rationale

        if deployment.candidate:
            deployment.candidate.status = "APPROVED"
            deployment.candidate.updated_at = now

        audit = models.AuditEvent(
            actor=actor,
            action="APPROVE_DEPLOYMENT",
            resource_type="deployment",
            resource_id=deployment.id,
            before_state_json={"status": "AWAITING_APPROVAL"},
            after_state_json={"status": "APPROVED", "approved_by": actor, "rationale": rationale},
            reason=rationale,
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(deployment)

        logger.info(f"Deployment '{deployment.id}' APPROVED by '{actor}'.")
        return deployment

    def reject_deployment(
        self,
        deployment_id: str,
        actor: str,
        reason: str,
    ) -> models.Deployment:
        """Explicit human rejection of a staged model version, leaving production unchanged."""
        deployment = self.db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
        if not deployment:
            raise EntityNotFoundError(f"Deployment '{deployment_id}' not found.")

        if deployment.status not in ("STAGED", "AWAITING_APPROVAL"):
            raise InvalidOperationError(
                f"Deployment '{deployment_id}' with status '{deployment.status}' cannot be rejected."
            )

        now = utc_now()
        deployment.status = "REJECTED"
        deployment.approval_rationale = reason
        deployment.completed_at = now

        if deployment.candidate:
            deployment.candidate.status = "REJECTED"
            deployment.candidate.updated_at = now

        audit = models.AuditEvent(
            actor=actor,
            action="REJECT_DEPLOYMENT",
            resource_type="deployment",
            resource_id=deployment.id,
            before_state_json={"status": deployment.status},
            after_state_json={"status": "REJECTED", "rejected_by": actor, "reason": reason},
            reason=reason,
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(deployment)

        logger.info(f"Deployment '{deployment.id}' REJECTED by '{actor}'. Production model remains unchanged.")
        return deployment

    def promote_deployment(
        self,
        deployment_id: str,
        actor: str,
    ) -> models.Deployment:
        """Promote an approved staged deployment to production with exact MLflow alias update."""
        staging_deployment = self.db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
        if not staging_deployment:
            raise EntityNotFoundError(f"Deployment '{deployment_id}' not found.")

        if staging_deployment.status != "APPROVED":
            raise InvalidOperationError(
                f"Deployment '{deployment_id}' must be APPROVED before promotion. Current status is '{staging_deployment.status}'."
            )

        candidate = staging_deployment.candidate
        if not candidate or candidate.status != "APPROVED":
            raise InvalidOperationError("Associated ModelCandidate must be APPROVED before promotion.")

        smoke_res = staging_deployment.smoke_test_results_json or {}
        if not smoke_res.get("passed"):
            raise InvalidOperationError("Cannot promote deployment whose staging smoke check did not pass.")

        prod_env = self._get_or_create_environment("production")
        previous_candidate_id = prod_env.active_candidate_id

        # Update MLflow Model Registry alias 'champion'
        client = self._get_client()
        try:
            client.set_registered_model_alias(
                name=candidate.mlflow_model_name,
                alias="champion",
                version=candidate.mlflow_model_version,
            )
        except Exception as e:
            raise DeploymentError(
                f"Failed to update MLflow 'champion' alias to version '{candidate.mlflow_model_version}': {e}"
            ) from e

        now = utc_now()

        # Create Production Deployment Record
        prod_deployment = models.Deployment(
            id=models.generate_uuid(),
            environment_id=prod_env.id,
            candidate_id=candidate.id,
            previous_candidate_id=previous_candidate_id,
            status="PROMOTED",
            smoke_test_results_json=staging_deployment.smoke_test_results_json,
            approved_by=staging_deployment.approved_by or actor,
            approved_at=staging_deployment.approved_at or now,
            approval_rationale=staging_deployment.approval_rationale or f"Promoted by {actor}",
            created_at=now,
            completed_at=now,
        )
        self.db.add(prod_deployment)
        self.db.flush()

        # Update environment pointer and statuses
        prod_env.active_candidate_id = candidate.id
        prod_env.updated_at = now

        candidate.status = "PROMOTED"
        candidate.updated_at = now

        staging_deployment.status = "PROMOTED"
        staging_deployment.completed_at = now

        # Archive previous production candidate if different
        if previous_candidate_id and previous_candidate_id != candidate.id:
            prev_cand = (
                self.db.query(models.ModelCandidate).filter(models.ModelCandidate.id == previous_candidate_id).first()
            )
            if prev_cand:
                prev_cand.status = "ARCHIVED"
                prev_cand.updated_at = now

        audit = models.AuditEvent(
            actor=actor,
            action="PROMOTE_MODEL_CANDIDATE",
            resource_type="deployment",
            resource_id=prod_deployment.id,
            before_state_json={"previous_production_candidate_id": previous_candidate_id},
            after_state_json={
                "production_deployment_id": prod_deployment.id,
                "candidate_id": candidate.id,
                "mlflow_model_name": candidate.mlflow_model_name,
                "mlflow_model_version": candidate.mlflow_model_version,
                "environment": "production",
                "status": "PROMOTED",
            },
            reason=f"Promoted model version {candidate.mlflow_model_version} to production champion",
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(prod_deployment)

        logger.info(
            f"Successfully PROMOTED ModelCandidate '{candidate.id}' (v{candidate.mlflow_model_version}) to production champion."
        )
        return prod_deployment

    def rollback_deployment(
        self,
        deployment_id: str,
        actor: str,
        reason: str,
        target_candidate_id: str | None = None,
    ) -> models.Deployment:
        """Audited rollback restoring a previously successful production model version with safe compensation."""
        deployment = self.db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
        if not deployment:
            raise EntityNotFoundError(f"Deployment '{deployment_id}' not found.")

        prod_env = self._get_or_create_environment("production")
        current_candidate_id = prod_env.active_candidate_id

        target_candidate: models.ModelCandidate | None = None
        if target_candidate_id:
            target_candidate = (
                self.db.query(models.ModelCandidate).filter(models.ModelCandidate.id == target_candidate_id).first()
            )
            if not target_candidate:
                raise EntityNotFoundError(f"Target ModelCandidate '{target_candidate_id}' not found.")
        elif deployment.previous_candidate_id and deployment.previous_candidate_id != current_candidate_id:
            target_candidate = (
                self.db.query(models.ModelCandidate)
                .filter(models.ModelCandidate.id == deployment.previous_candidate_id)
                .first()
            )
        elif deployment.candidate_id != current_candidate_id:
            target_candidate = deployment.candidate
        else:
            # Find the most recent historical promoted candidate in production different from current
            prior_deployment = (
                self.db.query(models.Deployment)
                .filter(
                    models.Deployment.environment_id == prod_env.id,
                    models.Deployment.candidate_id != current_candidate_id,
                )
                .order_by(models.Deployment.created_at.desc())
                .first()
            )
            if prior_deployment and prior_deployment.candidate:
                target_candidate = prior_deployment.candidate

        if not target_candidate:
            raise InvalidOperationError("Unable to resolve a valid known-good model version for rollback.")

        if current_candidate_id == target_candidate.id:
            raise InvalidOperationError(
                f"Target candidate '{target_candidate.id}' (v{target_candidate.mlflow_model_version}) is already active in production."
            )

        # Pre-switch verification: test artifact loading and prediction sanity
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            test_model = mlflow.sklearn.load_model(target_candidate.model_uri)
            df_sample = pd.DataFrame(SAMPLE_SMOKE_INPUT)
            X, _ = prepare_features(df_sample, target_col="traffic_volume")
            sample_preds = test_model.predict(X)
            if not bool(np.all(np.isfinite(sample_preds))):
                raise RollbackError(
                    f"Target candidate v{target_candidate.mlflow_model_version} produced non-finite predictions during pre-rollback check."
                )
        except Exception as e:
            raise RollbackError(
                f"Pre-rollback verification failed for candidate v{target_candidate.mlflow_model_version}: {e}"
            ) from e

        current_candidate = (
            self.db.query(models.ModelCandidate).filter(models.ModelCandidate.id == current_candidate_id).first()
            if current_candidate_id
            else None
        )
        prev_version = current_candidate.mlflow_model_version if current_candidate else None

        client = self._get_client()
        # Atomic MLflow alias switch
        try:
            client.set_registered_model_alias(
                name=target_candidate.mlflow_model_name,
                alias="champion",
                version=target_candidate.mlflow_model_version,
            )
        except Exception as e:
            raise RollbackError(
                f"Failed to update MLflow 'champion' alias to restored version v{target_candidate.mlflow_model_version}: {e}"
            ) from e

        # Database transaction with safe compensation if persistence fails
        now = utc_now()
        try:
            new_deployment = models.Deployment(
                id=models.generate_uuid(),
                environment_id=prod_env.id,
                candidate_id=target_candidate.id,
                previous_candidate_id=current_candidate_id,
                status="PROMOTED",
                smoke_test_results_json={"rollback": True, "verified_at": now.isoformat()},
                approved_by=actor,
                approved_at=now,
                approval_rationale=f"ROLLBACK from v{prev_version} to v{target_candidate.mlflow_model_version}: {reason}",
                created_at=now,
                completed_at=now,
            )
            self.db.add(new_deployment)
            self.db.flush()

            prod_env.active_candidate_id = target_candidate.id
            prod_env.updated_at = now

            target_candidate.status = "PROMOTED"
            target_candidate.updated_at = now

            if current_candidate:
                current_candidate.status = "ROLLED_BACK"
                current_candidate.updated_at = now

            audit = models.AuditEvent(
                actor=actor,
                action="ROLLBACK_MODEL_CANDIDATE",
                resource_type="deployment",
                resource_id=new_deployment.id,
                before_state_json={
                    "active_candidate_id": current_candidate_id,
                    "active_version": prev_version,
                },
                after_state_json={
                    "active_candidate_id": target_candidate.id,
                    "active_version": target_candidate.mlflow_model_version,
                    "rollback_deployment_id": new_deployment.id,
                },
                reason=reason,
            )
            self.db.add(audit)
            self.db.commit()
            self.db.refresh(new_deployment)
        except Exception as db_exc:
            logger.error(f"Database persistence failed during rollback. Executing MLflow compensation: {db_exc}")
            self.db.rollback()
            if prev_version:
                try:
                    client.set_registered_model_alias(
                        name=target_candidate.mlflow_model_name,
                        alias="champion",
                        version=prev_version,
                    )
                except Exception as comp_exc:
                    logger.critical(f"Compensating MLflow alias revert also failed: {comp_exc}")
            raise RollbackError(
                f"Rollback aborted and compensated due to database persistence failure: {db_exc}"
            ) from db_exc

        logger.info(
            f"Successfully ROLLED BACK production model from v{prev_version} to v{target_candidate.mlflow_model_version}."
        )
        return new_deployment

    def list_deployments(self, environment_name: str | None = None) -> list[models.Deployment]:
        query = self.db.query(models.Deployment)
        if environment_name:
            query = query.join(models.DeploymentEnvironment).filter(
                models.DeploymentEnvironment.name == environment_name
            )
        return query.order_by(models.Deployment.created_at.desc()).all()
