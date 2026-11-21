import json
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.errors import ArtifactLoggingError, TrackingInitializationError, TrackingLoggingError
from app.core.logging import logger


class TrackingFinalizationWarning(UserWarning):
    """Warning issued when MLflow run finalization fails but core execution succeeded."""

    pass


class BaseTracker(ABC):
    """Abstract interface defining the ModelGuard experiment tracking contract."""

    @abstractmethod
    def start_training_run(
        self,
        job_id: str,
        run_id: str,
        dataset_version_id: str,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Initialize or start a tracking run. Returns the run ID or None if disabled."""
        pass

    @abstractmethod
    def log_parameters(self, params: dict[str, Any]) -> None:
        """Log experiment, split, and model hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric evaluation and comparison metrics."""
        pass

    @abstractmethod
    def log_models(
        self,
        candidate_pipeline: Pipeline,
        baseline_pipeline: Pipeline,
        input_example: pd.DataFrame | None = None,
    ) -> None:
        """Log fitted baseline and candidate scikit-learn pipelines as separate named artifacts."""
        pass

    @abstractmethod
    def log_artifacts(self, artifacts: dict[str, Any | Path | pd.DataFrame]) -> None:
        """Log files, prediction tables, and serialized manifest artifacts."""
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED", tags: dict[str, str] | None = None) -> None:
        """Finalize the active tracking run with the specified terminal status."""
        pass

    @abstractmethod
    def get_run_id(self) -> str | None:
        """Return the active or last-managed tracking run ID."""
        pass

    @abstractmethod
    def get_artifact_uri(self) -> str | None:
        """Return the base artifact URI of the active run."""
        pass

    @abstractmethod
    def resolve_experiment(self, experiment_name: str) -> str | None:
        """Resolve or create the tracking experiment ID."""
        pass


class NullTracker(BaseTracker):
    """Explicit no-op tracker used when experiment tracking is disabled (MLFLOW_ENABLED=False).

    Semantics:
    - Returns None for run ID.
    - Never generates dummy or fake IDs.
    - Performs zero filesystem or network I/O.
    """

    def __init__(self) -> None:
        self._run_id: str | None = None

    def start_training_run(
        self,
        job_id: str,
        run_id: str,
        dataset_version_id: str,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        logger.info("Tracking disabled (NullTracker active); returning None for run ID.")
        return None

    def log_parameters(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_models(
        self,
        candidate_pipeline: Pipeline,
        baseline_pipeline: Pipeline,
        input_example: pd.DataFrame | None = None,
    ) -> None:
        pass

    def log_artifacts(self, artifacts: dict[str, Any | Path | pd.DataFrame]) -> None:
        pass

    def end_run(self, status: str = "FINISHED", tags: dict[str, str] | None = None) -> None:
        pass

    def get_run_id(self) -> str | None:
        return None

    def get_artifact_uri(self) -> str | None:
        return None

    def resolve_experiment(self, experiment_name: str) -> str | None:
        return None


class MLflowTracker(BaseTracker):
    """Authoritative MLflow experiment tracking implementation for ModelGuard.

    Guarantees:
    - 1:1 mapping with ModelGuard TrainingRun.
    - Mandatory identity tags: modelguard.training_job_id and modelguard.training_run_id.
    - Unambiguous metric/parameter key namespaces.
    - Decoupled baseline and candidate model artifacts within the same run.
    - Cross-platform path normalization (Windows & Linux).
    - Lazy filesystem directory initialization (zero import-time side effects).
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        artifact_root: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        self.artifact_root = artifact_root or settings.MLFLOW_ARTIFACT_ROOT
        self._client: MlflowClient | None = None
        self._active_run_id: str | None = None
        self._active_experiment_id: str | None = None
        self._artifact_uri: str | None = None

    def _ensure_initialized(self) -> None:
        """Perform lazy initialization of MLflow tracking environment and storage directories."""
        if self._client is not None:
            return

        try:
            # Create local storage directories lazily if local SQLite/file tracking
            parsed = urlparse(self.tracking_uri)
            if parsed.scheme in ("sqlite", "file", ""):
                settings.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
                settings.MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = MlflowClient(self.tracking_uri)
        except Exception as err:
            raise TrackingInitializationError(
                f"Failed to initialize MLflow tracking client with URI '{self.tracking_uri}': {err}"
            ) from err

    def resolve_experiment(self, experiment_name: str | None = None) -> str:
        """Resolve an existing MLflow experiment ID or create a new one with the authoritative artifact root."""
        self._ensure_initialized()
        name = experiment_name or self.experiment_name

        try:
            assert self._client is not None
            exp = self._client.get_experiment_by_name(name)
            if exp is not None:
                self._active_experiment_id = exp.experiment_id
                return exp.experiment_id

            exp_id = self._client.create_experiment(
                name=name,
                artifact_location=self.artifact_root,
            )
            self._active_experiment_id = exp_id
            return exp_id
        except Exception as err:
            raise TrackingInitializationError(f"Failed to resolve or create MLflow experiment '{name}': {err}") from err

    def start_training_run(
        self,
        job_id: str,
        run_id: str,
        dataset_version_id: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start an MLflow active run with mandatory ModelGuard provenance tags."""
        self._ensure_initialized()
        exp_id = self.resolve_experiment()

        # Build mandatory identity tags
        run_tags = {
            "modelguard.training_job_id": str(job_id),
            "modelguard.training_run_id": str(run_id),
            "modelguard.dataset_version_id": str(dataset_version_id),
        }
        if tags:
            run_tags.update(tags)

        try:
            active_run = mlflow.start_run(
                experiment_id=exp_id,
                run_name=f"job_{job_id[:8]}_run_{run_id[:8]}",
                tags=run_tags,
            )
            self._active_run_id = active_run.info.run_id
            self._artifact_uri = active_run.info.artifact_uri
            logger.info(
                f"Started MLflow run '{self._active_run_id}' for TrainingJob '{job_id}' (TrainingRun '{run_id}')."
            )
            return self._active_run_id
        except Exception as err:
            raise TrackingInitializationError(f"Failed to start MLflow run for job '{job_id}': {err}") from err

    def log_parameters(self, params: dict[str, Any]) -> None:
        """Log parameters with stringification of non-primitive types and length bounds."""
        if not self._active_run_id:
            raise TrackingLoggingError("Cannot log parameters: no active MLflow run.")

        formatted_params: dict[str, str | int | float | bool] = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, (int, float, bool)):
                formatted_params[k] = v
            else:
                formatted_params[k] = str(v)[:500]

        try:
            mlflow.log_params(formatted_params)
        except Exception as err:
            raise TrackingLoggingError(f"Failed to log MLflow parameters: {err}") from err

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics."""
        if not self._active_run_id:
            raise TrackingLoggingError("Cannot log metrics: no active MLflow run.")

        # Filter out NaN or non-finite values
        clean_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                val = float(v)
                if not (np.isnan(val) or np.isinf(val)):
                    clean_metrics[k] = val
            except (ValueError, TypeError):
                continue

        try:
            mlflow.log_metrics(clean_metrics, step=step)
        except Exception as err:
            raise TrackingLoggingError(f"Failed to log MLflow metrics: {err}") from err

    def log_models(
        self,
        candidate_pipeline: Pipeline,
        baseline_pipeline: Pipeline,
        input_example: pd.DataFrame | None = None,
    ) -> None:
        """Log candidate and baseline models as separate artifacts within the single active MLflow run."""
        if not self._active_run_id:
            raise ArtifactLoggingError("Cannot log models: no active MLflow run.")

        try:
            signature = None
            sample_df = None
            if input_example is not None and not input_example.empty:
                sample_df = input_example.head(5)
                try:
                    preds = candidate_pipeline.predict(sample_df)
                    signature = infer_signature(sample_df, preds)
                except Exception as sig_err:
                    logger.warning(f"Could not infer model signature: {sig_err}")

            # 1. Log Candidate Model
            mlflow.sklearn.log_model(
                sk_model=candidate_pipeline,
                artifact_path="candidate_model",
                signature=signature,
                input_example=sample_df,
            )

            # 2. Log Baseline Model
            mlflow.sklearn.log_model(
                sk_model=baseline_pipeline,
                artifact_path="baseline_model",
                signature=signature,
                input_example=sample_df,
            )
            logger.info("Successfully logged candidate_model and baseline_model to MLflow.")
        except Exception as err:
            raise ArtifactLoggingError(f"Failed to log scikit-learn models to MLflow: {err}") from err

    def log_artifacts(self, artifacts: dict[str, Any | Path | pd.DataFrame]) -> None:
        """Log artifacts into structured subdirectories (manifests, predictions, evaluations)."""
        if not self._active_run_id:
            raise ArtifactLoggingError("Cannot log artifacts: no active MLflow run.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                for name, content in artifacts.items():
                    if isinstance(content, pd.DataFrame):
                        dest = tmp_path / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        content.to_csv(dest, index=False, lineterminator="\n")
                        mlflow.log_artifact(
                            dest.as_posix(), artifact_path=dest.parent.name if dest.parent != tmp_path else None
                        )
                    elif isinstance(content, (dict, list)):
                        dest = tmp_path / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with open(dest, "w", encoding="utf-8") as f:
                            json.dump(content, f, indent=2, default=str)
                        mlflow.log_artifact(
                            dest.as_posix(), artifact_path=dest.parent.name if dest.parent != tmp_path else None
                        )
                    elif isinstance(content, Path) and content.is_file():
                        mlflow.log_artifact(content.as_posix(), artifact_path=name)
                    elif isinstance(content, str) and Path(content).is_file():
                        mlflow.log_artifact(Path(content).as_posix(), artifact_path=name)
            except Exception as err:
                raise ArtifactLoggingError(f"Failed to log artifacts to MLflow: {err}") from err

    def end_run(self, status: str = "FINISHED", tags: dict[str, str] | None = None) -> None:
        """Finalize the active MLflow run. If finalization fails, emits TrackingFinalizationWarning."""
        if not self._active_run_id:
            return

        try:
            if tags:
                mlflow.set_tags(tags)
            mlflow.end_run(status=status)
            logger.info(f"Finalized MLflow run '{self._active_run_id}' with status '{status}'.")
        except MlflowException as m_err:
            warnings.warn(
                f"Failed to finalize MLflow run '{self._active_run_id}' ({m_err})",
                category=TrackingFinalizationWarning,
                stacklevel=2,
            )
        except Exception as err:
            warnings.warn(
                f"Unexpected error finalizing MLflow run '{self._active_run_id}': {err}",
                category=TrackingFinalizationWarning,
                stacklevel=2,
            )
        finally:
            self._active_run_id = None

    def get_run_id(self) -> str | None:
        return self._active_run_id

    def get_artifact_uri(self) -> str | None:
        return self._artifact_uri
