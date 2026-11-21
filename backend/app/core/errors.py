from typing import Any

from fastapi import HTTPException, status


class ModelGuardError(Exception):
    """Base domain exception for ModelGuard."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class EntityNotFoundError(ModelGuardError):
    """Raised when an entity is not found."""

    pass


class IntegrityError(ModelGuardError):
    """Raised when checksum or data integrity verification fails."""

    pass


class ContractValidationError(ModelGuardError):
    """Raised when dataset fails required data contract checks."""

    pass


class InvalidOperationError(ModelGuardError):
    """Raised when an illegal lifecycle state transition is attempted."""

    pass


class StorageError(ModelGuardError):
    """Raised when reading or writing dataset artifacts fails."""

    pass


class TrackingError(ModelGuardError):
    """Base exception for experiment tracking failures."""

    pass


class TrackingInitializationError(TrackingError):
    """Raised when MLflow experiment or run initialization fails."""

    pass


class TrackingLoggingError(TrackingError):
    """Raised when logging parameters, metrics, or tags fails."""

    pass


class ArtifactLoggingError(TrackingError):
    """Raised when persisting model or evaluation artifacts fails."""

    pass


class DatabasePersistenceError(ModelGuardError):
    """Raised when persisting critical run or job state to the database fails."""

    pass


class ReconciliationError(ModelGuardError):
    """Raised when reconciliation of orphaned runs encounters an unrecoverable failure."""

    pass


class ModelRegistryError(ModelGuardError):
    """Base exception for model registry and candidate operations."""

    pass


class ReleaseGateError(ModelGuardError):
    """Raised when model candidate fails the release gate."""

    pass


class DeploymentError(ModelGuardError):
    """Raised when staging or production deployment fails."""

    pass


class InferenceError(ModelGuardError):
    """Raised when model inference execution fails."""

    pass


class InputContractViolationError(InferenceError):
    """Raised when incoming inference payload violates model input contract or signature."""

    pass


class FeedbackError(ModelGuardError):
    """Base exception for delayed feedback operations."""

    pass


class FeedbackConflictError(FeedbackError):
    """Raised when conflicting ground-truth feedback is submitted for an inference."""

    pass


class InsufficientLabelsError(FeedbackError):
    """Raised when quality evaluation cannot be calculated due to insufficient ground-truth labels."""

    pass


class RollbackError(DeploymentError):
    """Raised when deployment rollback fails or is aborted."""

    pass


class DriftAnalysisError(ModelGuardError):
    """Raised when input or prediction drift calculation fails."""

    pass


class MonitoringAlertError(ModelGuardError):
    """Raised when alert operations fail."""

    pass


def not_found_exception_handler(entity_name: str, entity_id: Any) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{entity_name} with identifier '{entity_id}' not found.",
    )


def conflict_exception_handler(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=message,
    )


def validation_exception_handler(message: str, details: dict[str, Any] | None = None) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error": message, "details": details or {}},
    )
