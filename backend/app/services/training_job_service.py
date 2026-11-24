import hashlib
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any

from mlflow.tracking import MlflowClient
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import (
    DatabasePersistenceError,
    EntityNotFoundError,
    InvalidOperationError,
)
from app.core.logging import logger
from app.db import models
from app.ml.data_loader import load_validated_dataset
from app.ml.tracking import BaseTracker, MLflowTracker, NullTracker
from app.ml.training import TrainingService
from app.schemas.training import ReconcileReport, SplitConfig, TrainingJobCreate


def utc_now() -> datetime:
    return datetime.now(UTC)


def _to_utc_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def pid_exists(pid: int) -> bool:
    """Check if process PID exists on local machine using standard library only."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def get_current_git_sha() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_dependency_lock_hash() -> str | None:
    for lock_file in ("uv.lock", "poetry.lock", "requirements.txt", "pyproject.toml"):
        p = settings.BASE_DIR / lock_file
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()
            except Exception:
                pass
    return None


class TrainingJobService:
    """Authoritative lifecycle service for TrainingJob and TrainingRun.

    Architectural Invariants:
    - 1:1 strict mapping: TrainingJob <-> TrainingRun <-> MLflow run.
    - Synchronous, in-process execution.
    - Atomic duplicate-execution guard.
    - Decoupled ML training engine (TrainingService).
    - Conservative process liveness and orphan recovery.
    - Resilient error handling that preserves original training exceptions.
    """

    def __init__(self, db: Session, tracker: BaseTracker | None = None) -> None:
        self.db = db
        if tracker is not None:
            self.tracker = tracker
        elif settings.MLFLOW_ENABLED:
            self.tracker = MLflowTracker(
                tracking_uri=settings.MLFLOW_TRACKING_URI,
                experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
                artifact_root=settings.MLFLOW_ARTIFACT_ROOT,
            )
        else:
            self.tracker = NullTracker()

    def create_training_job(self, data: TrainingJobCreate) -> models.TrainingJob:
        """Create a new TrainingJob in QUEUED state."""
        dataset_version = (
            self.db.query(models.DatasetVersion).filter(models.DatasetVersion.id == data.dataset_version_id).first()
        )
        if not dataset_version:
            raise EntityNotFoundError(f"DatasetVersion '{data.dataset_version_id}' not found.")

        if data.validation_run_id:
            val_run = (
                self.db.query(models.ValidationRun).filter(models.ValidationRun.id == data.validation_run_id).first()
            )
            if not val_run:
                raise EntityNotFoundError(f"ValidationRun '{data.validation_run_id}' not found.")

        job = models.TrainingJob(
            dataset_version_id=data.dataset_version_id,
            validation_run_id=data.validation_run_id,
            config_json=data.config.model_dump(),
            status="QUEUED",
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        audit = models.AuditEvent(
            actor="system",
            action="CREATE_TRAINING_JOB",
            resource_type="training_job",
            resource_id=job.id,
            after_state_json={"status": "QUEUED", "dataset_version_id": job.dataset_version_id},
            reason="Created new model training job",
        )
        self.db.add(audit)
        self.db.commit()

        logger.info(f"Created TrainingJob '{job.id}' for DatasetVersion '{job.dataset_version_id}'.")
        return job

    def execute_training_job(self, job_id: str) -> models.TrainingRun:
        """Execute a TrainingJob through the approved 15-stage lifecycle.

        Guarantees:
        - Atomic status transition from QUEUED to RUNNING.
        - Concurrent / duplicate executions rejected with InvalidOperationError.
        - Immediate persistence of MLflow run ID.
        - Preserves root exception when cleanup operations fail.
        """
        # =========================================================================
        # Stage 1: Atomic Execution Guard & Duplicate Protection (Tx 1)
        # =========================================================================
        stmt = (
            update(models.TrainingJob)
            .where(models.TrainingJob.id == job_id, models.TrainingJob.status == "QUEUED")
            .values(status="RUNNING", updated_at=utc_now())
        )
        res = self.db.execute(stmt)
        self.db.commit()

        if res.rowcount == 0:
            current_job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
            if not current_job:
                raise EntityNotFoundError(f"TrainingJob '{job_id}' not found.")
            raise InvalidOperationError(
                f"Cannot execute TrainingJob '{job_id}': status is '{current_job.status}', expected 'QUEUED'."
            )

        job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).one()

        # =========================================================================
        # Stage 2 & 3: TrainingRun Instantiation & Initial DB Persistence (Tx 2)
        # =========================================================================
        host = socket.gethostname()
        pid = os.getpid()
        cfg = job.config_json or {}
        split_cfg_dict = cfg.get("split", {})
        split_config = SplitConfig.model_validate(split_cfg_dict) if split_cfg_dict else SplitConfig()

        run = models.TrainingRun(
            id=models.generate_uuid(),
            training_job_id=job.id,
            dataset_version_id=job.dataset_version_id,
            git_sha=get_current_git_sha(),
            dependency_lock_hash=get_dependency_lock_hash(),
            python_version=platform.python_version(),
            random_seed=split_config.seed,
            status="RUNNING",
            started_at=utc_now(),
        )
        self.db.add(run)

        audit_start = models.AuditEvent(
            actor="system",
            action="TRAINING_RUN_STARTED",
            resource_type="training_run",
            resource_id=run.id,
            after_state_json={"status": "RUNNING", "host": host, "pid": pid},
            reason="Training run execution started",
        )
        self.db.add(audit_start)
        self.db.commit()
        self.db.refresh(run)

        mlflow_run_id: str | None = None

        try:
            # =====================================================================
            # Stage 4 & 5: MLflow Initialization & Run Creation
            # =====================================================================
            identity_tags = {
                "modelguard.execution_host": host,
                "modelguard.execution_pid": str(pid),
                "modelguard.random_seed": str(split_config.seed),
            }
            mlflow_run_id = self.tracker.start_training_run(
                job_id=job.id,
                run_id=run.id,
                dataset_version_id=job.dataset_version_id,
                tags=identity_tags,
            )

            # =====================================================================
            # Stage 6: Immediate Persistence of MLflow Run ID (Tx 3)
            # =====================================================================
            if mlflow_run_id is not None:
                try:
                    run.mlflow_run_id = mlflow_run_id
                    self.db.commit()
                except SQLAlchemyError as db_err:
                    self.db.rollback()
                    try:
                        self.tracker.end_run(status="FAILED", tags={"failure_stage": "persist_mlflow_run_id"})
                    except Exception as clean_err:
                        logger.warning(f"Error terminating MLflow run during DB failure: {clean_err}")
                    raise DatabasePersistenceError(
                        f"Failed to persist MLflow run ID '{mlflow_run_id}' for run '{run.id}': {db_err}"
                    ) from db_err

            # =====================================================================
            # Stage 7: Dataset Snapshot & Checksum Verification
            # =====================================================================
            df, version, _ = load_validated_dataset(self.db, job.dataset_version_id)

            # =====================================================================
            # Stage 8 & 9: TrainingService Execution & Validation Evaluation
            # =====================================================================
            training_service = TrainingService(split_config=split_config)
            candidate_hp = cfg.get("hyperparameters", {})

            result, test_df = training_service.train_and_evaluate_validation(
                df=df,
                dataset_version_id=version.id,
                dataset_sha256=version.sha256_hash,
                candidate_hyperparameters=candidate_hp,
            )

            # =====================================================================
            # Stage 10 & 11: Candidate Selection & Test Holdout Evaluation
            # =====================================================================
            test_eval, test_predictions_df = TrainingService.evaluate_test_holdout(
                pipeline=result.candidate_pipeline,
                test_df=test_df,
                target_col=split_config.target_column,
                model_name="candidate_random_forest",
            )

            # =====================================================================
            # Stage 12 & 13: Model, Metric & Artifact Logging
            # =====================================================================
            # Parameters
            params: dict[str, Any] = {
                "split_train_ratio": split_config.train_ratio,
                "split_val_ratio": split_config.val_ratio,
                "split_test_ratio": split_config.test_ratio,
                "split_seed": split_config.seed,
                "target_column": split_config.target_column,
                "timestamp_column": split_config.timestamp_column,
                "baseline_strategy": "mean",
            }
            for k, v in candidate_hp.items():
                params[f"candidate_{k}"] = v
            self.tracker.log_parameters(params)

            # Metrics
            metrics: dict[str, float] = {
                "candidate_val_mae": result.candidate_val_eval.metrics.mae,
                "candidate_val_rmse": result.candidate_val_eval.metrics.rmse,
                "candidate_val_r2": result.candidate_val_eval.metrics.r2,
                "baseline_val_mae": result.baseline_val_eval.metrics.mae,
                "baseline_val_rmse": result.baseline_val_eval.metrics.rmse,
                "baseline_val_r2": result.baseline_val_eval.metrics.r2,
                "comparison_delta_mae": result.comparison.delta_mae,
                "comparison_delta_rmse": result.comparison.delta_rmse,
                "comparison_delta_r2": result.comparison.delta_r2,
                "candidate_test_mae": test_eval.metrics.mae,
                "candidate_test_rmse": test_eval.metrics.rmse,
                "candidate_test_r2": test_eval.metrics.r2,
            }
            if result.baseline_val_eval.metrics.mae > 0:
                metrics["comparison_mae_improvement_pct"] = (
                    (result.baseline_val_eval.metrics.mae - result.candidate_val_eval.metrics.mae)
                    / result.baseline_val_eval.metrics.mae
                    * 100.0
                )
            self.tracker.log_metrics(metrics)

            # Artifacts
            artifacts: dict[str, Any] = {
                "manifests/split_manifest.json": result.split_manifest.model_dump(mode="json"),
                "manifests/feature_metadata.json": result.feature_metadata.model_dump(mode="json"),
                "predictions/validation_predictions.csv": result.val_predictions,
                "predictions/test_predictions.csv": test_predictions_df,
                "evaluations/comparison_summary.json": result.comparison.model_dump(mode="json"),
            }
            self.tracker.log_artifacts(artifacts)

            # Models
            from app.ml.features import prepare_features

            X_sample, _ = prepare_features(df.head(5), target_col=split_config.target_column)
            self.tracker.log_models(
                candidate_pipeline=result.candidate_pipeline,
                baseline_pipeline=result.baseline_pipeline,
                input_example=X_sample,
            )

            # =====================================================================
            # Stage 14: MLflow Run Finalization
            # =====================================================================
            final_tags = {
                "modelguard.is_better_than_baseline": str(result.comparison.candidate_superior).lower(),
            }
            self.tracker.end_run(status="FINISHED", tags=final_tags)

            # =====================================================================
            # Stage 15: Database Finalization (Tx 4)
            # =====================================================================
            run.status = "SUCCEEDED"
            run.completed_at = utc_now()
            run.metrics_json = metrics
            run.artifact_uri = self.tracker.get_artifact_uri()
            if mlflow_run_id:
                run.split_manifest_uri = f"runs:/{mlflow_run_id}/manifests/split_manifest.json"

            job.status = "SUCCEEDED"
            job.updated_at = utc_now()

            audit_success = models.AuditEvent(
                actor="system",
                action="TRAINING_SUCCEEDED",
                resource_type="training_job",
                resource_id=job.id,
                after_state_json={"status": "SUCCEEDED", "metrics": metrics},
                reason="Training and evaluation completed successfully",
            )
            self.db.add(audit_success)
            self.db.commit()
            self.db.refresh(run)

            logger.info(f"TrainingJob '{job.id}' and TrainingRun '{run.id}' SUCCEEDED.")
            return run

        except Exception as orig_exc:
            logger.error(f"Execution failure in TrainingJob '{job_id}': {orig_exc}", exc_info=True)

            # Resilient MLflow cleanup: cleanup failure MUST NOT mask orig_exc
            try:
                self.tracker.end_run(status="FAILED", tags={"error": str(orig_exc)[:500]})
            except Exception as clean_err:
                logger.warning(f"Secondary error during MLflow cleanup: {clean_err}")

            # Database failure transition
            try:
                self.db.rollback()
                run.status = "FAILED"
                run.completed_at = utc_now()
                job.status = "FAILED"
                job.updated_at = utc_now()

                audit_fail = models.AuditEvent(
                    actor="system",
                    action="TRAINING_FAILED",
                    resource_type="training_job",
                    resource_id=job.id,
                    after_state_json={"status": "FAILED", "error": str(orig_exc)},
                    reason=f"Training failed: {orig_exc}",
                )
                self.db.add(audit_fail)
                self.db.commit()
            except Exception as db_fail_err:
                logger.error(f"Failed to persist failure state to DB: {db_fail_err}")

            raise orig_exc from None

    def retry_training_job(self, failed_job_id: str) -> models.TrainingJob:
        """Retry a failed TrainingJob by creating a new TrainingJob linked via audit trail (Option A)."""
        failed_job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == failed_job_id).first()
        if not failed_job:
            raise EntityNotFoundError(f"TrainingJob '{failed_job_id}' not found.")

        if failed_job.status != "FAILED":
            raise InvalidOperationError(
                f"Cannot retry TrainingJob '{failed_job_id}': status is '{failed_job.status}', expected 'FAILED'."
            )

        new_job = models.TrainingJob(
            dataset_version_id=failed_job.dataset_version_id,
            validation_run_id=failed_job.validation_run_id,
            config_json=failed_job.config_json,
            status="QUEUED",
        )
        self.db.add(new_job)
        self.db.commit()
        self.db.refresh(new_job)

        audit = models.AuditEvent(
            actor="system",
            action="RETRY_TRAINING_JOB",
            resource_type="training_job",
            resource_id=new_job.id,
            before_state_json={
                "original_job_id": failed_job.id,
                "original_run_id": failed_job.training_run.id if failed_job.training_run else None,
            },
            after_state_json={"retry_job_id": new_job.id, "status": "QUEUED"},
            reason=f"Retry initiated for failed job {failed_job.id}",
        )
        self.db.add(audit)
        self.db.commit()

        logger.info(f"Retried failed TrainingJob '{failed_job_id}' -> created new TrainingJob '{new_job.id}'.")
        return new_job

    def reconcile_orphaned_runs(self, max_age_seconds: int = 3600) -> ReconcileReport:
        """Reconcile unlinked, stale, or interrupted training runs between DB and MLflow."""
        report = ReconcileReport()
        cutoff_time = utc_now() - timedelta(seconds=max_age_seconds)
        current_host = socket.gethostname()

        client: MlflowClient | None = None
        exp_name = settings.MLFLOW_EXPERIMENT_NAME
        if isinstance(self.tracker, MLflowTracker):
            try:
                client = MlflowClient(self.tracker.tracking_uri)
                exp_name = self.tracker.experiment_name
            except Exception as e:
                logger.error(f"Cannot initialize MlflowClient during reconciliation: {e}")
                return report
        elif settings.MLFLOW_ENABLED and not isinstance(self.tracker, NullTracker):
            try:
                client = MlflowClient(settings.MLFLOW_TRACKING_URI)
            except Exception as e:
                logger.error(f"Cannot initialize MlflowClient during reconciliation: {e}")
                return report

        # 1. Inspect all TrainingRuns currently marked RUNNING in DB
        running_runs = self.db.query(models.TrainingRun).filter(models.TrainingRun.status == "RUNNING").all()

        for run in running_runs:
            job = run.training_job

            # Case A: Unlinked MLflow Run (mlflow_run_id is NULL)
            if run.mlflow_run_id is None and client is not None:
                try:
                    exp = client.get_experiment_by_name(exp_name)
                    if exp:
                        filter_str = f"tags.modelguard.training_run_id = '{run.id}'"
                        matched = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=filter_str)

                        if len(matched) == 1:
                            found_run = matched[0]
                            run.mlflow_run_id = found_run.info.run_id
                            logger.info(f"Recovered unlinked MLflow run '{found_run.info.run_id}' for run '{run.id}'.")
                            report.reconciled_runs.append(run.id)
                        elif len(matched) > 1:
                            logger.warning(f"Multiple MLflow runs matched tag for run '{run.id}'. Quarantining.")
                            for m in matched:
                                client.set_tag(m.info.run_id, "modelguard.quarantine", "duplicate_tag_collision")
                                client.set_terminated(m.info.run_id, status="FAILED")
                            run.status = "FAILED"
                            job.status = "FAILED"
                            report.orphaned_runs_failed.append(run.id)
                            self.db.commit()
                            continue
                        else:
                            # 0 matches: run was never created
                            if _to_utc_aware(run.created_at) < cutoff_time:
                                run.status = "FAILED"
                                job.status = "FAILED"
                                report.orphaned_runs_failed.append(run.id)
                                self.db.commit()
                                continue
                except Exception as err:
                    logger.warning(f"Error querying MLflow by tags for run '{run.id}': {err}")

            # Case B: Execution Liveness Check
            if run.status == "RUNNING":
                audit_event = (
                    self.db.query(models.AuditEvent)
                    .filter(
                        models.AuditEvent.resource_id == run.id,
                        models.AuditEvent.action == "TRAINING_RUN_STARTED",
                    )
                    .first()
                )
                start_data = audit_event.after_state_json if audit_event else {}
                exec_host = start_data.get("host")
                exec_pid = start_data.get("pid")

                is_dead_process = False
                if exec_host == current_host and exec_pid:
                    if not pid_exists(int(exec_pid)):
                        is_dead_process = True

                if is_dead_process:
                    logger.info(f"Process PID {exec_pid} on {exec_host} is dead. Terminating orphaned run '{run.id}'.")
                    run.status = "FAILED"
                    run.completed_at = utc_now()
                    job.status = "FAILED"
                    if run.mlflow_run_id and client:
                        try:
                            client.set_terminated(run.mlflow_run_id, status="FAILED")
                        except Exception:
                            pass
                    report.orphaned_runs_failed.append(run.id)
                    self.db.commit()
                    continue

                # Conservative Policy: Remote or unknown host
                if _to_utc_aware(run.created_at) < cutoff_time:
                    logger.warning(
                        f"Run '{run.id}' is older than {max_age_seconds}s but liveness is ambiguous (host: {exec_host}). Reporting without mutation."
                    )
                    report.ambiguous_runs.append(run.id)

        # 2. Inspect DB SUCCEEDED runs where MLflow run was left unfinalized (RUNNING)
        if client is not None:
            succeeded_runs = (
                self.db.query(models.TrainingRun)
                .filter(
                    models.TrainingRun.status == "SUCCEEDED",
                    models.TrainingRun.mlflow_run_id.isnot(None),
                )
                .all()
            )
            for s_run in succeeded_runs:
                try:
                    m_run = client.get_run(s_run.mlflow_run_id)
                    if m_run.info.status == "RUNNING":
                        logger.info(f"Reconciling unfinalized MLflow run '{s_run.mlflow_run_id}' to FINISHED.")
                        client.set_tag(s_run.mlflow_run_id, "modelguard.reconciliation", "finalized_post_execution")
                        client.set_terminated(s_run.mlflow_run_id, status="FINISHED")
                        report.reconciled_runs.append(s_run.id)
                except Exception as err:
                    logger.warning(f"Error checking MLflow status for run '{s_run.mlflow_run_id}': {err}")

        self.db.commit()
        return report

    def list_training_jobs(
        self,
        status: str | None = None,
        dataset_version_id: str | None = None,
    ) -> list[models.TrainingJob]:
        """List training jobs ordered chronologically descending with optional filtering."""
        query = self.db.query(models.TrainingJob).order_by(models.TrainingJob.created_at.desc())
        if status:
            query = query.filter(models.TrainingJob.status == status)
        if dataset_version_id:
            query = query.filter(models.TrainingJob.dataset_version_id == dataset_version_id)
        return query.all()

    def get_training_job(self, job_id: str) -> models.TrainingJob:
        """Retrieve a specific training job by ID."""
        job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            raise EntityNotFoundError(f"TrainingJob '{job_id}' not found.")
        return job

    def list_training_runs(
        self,
        status: str | None = None,
        dataset_version_id: str | None = None,
    ) -> list[models.TrainingRun]:
        """List training runs ordered chronologically descending with optional filtering."""
        query = self.db.query(models.TrainingRun).order_by(models.TrainingRun.created_at.desc())
        if status:
            query = query.filter(models.TrainingRun.status == status)
        if dataset_version_id:
            query = query.filter(models.TrainingRun.dataset_version_id == dataset_version_id)
        return query.all()

    def get_training_run(self, run_id: str) -> models.TrainingRun:
        """Retrieve a specific training run by ID."""
        run = self.db.query(models.TrainingRun).filter(models.TrainingRun.id == run_id).first()
        if not run:
            raise EntityNotFoundError(f"TrainingRun '{run_id}' not found.")
        return run
