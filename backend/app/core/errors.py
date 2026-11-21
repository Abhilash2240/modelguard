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
