from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.core.errors import (
    ContractValidationError,
    EntityNotFoundError,
    IntegrityError,
    StorageError,
)
from app.db import models
from app.domain.hashing import calculate_sha256


def load_validated_dataset(
    db: Session,
    dataset_version_id: str,
) -> tuple[pd.DataFrame, models.DatasetVersion, models.ValidationRun]:
    """Load an exact, immutable dataset snapshot from storage after verifying integrity and validation evidence.

    Constraints enforced:
    1. Dataset version must exist in the database.
    2. Physical file must exist at recorded storage URI.
    3. File SHA-256 digest must match recorded database hash exactly (anti-tampering).
    4. Dataset version must have an associated validation run with status PASSED or WARNING (zero blocking errors).
    """
    version = db.query(models.DatasetVersion).filter(models.DatasetVersion.id == dataset_version_id).first()
    if not version:
        raise EntityNotFoundError(f"DatasetVersion with id '{dataset_version_id}' not found.")

    file_path = Path(version.storage_uri)
    if not file_path.is_file():
        raise StorageError(f"Dataset snapshot file not found at path: {file_path}")

    # Cryptographic integrity check
    current_hash = calculate_sha256(file_path)
    if current_hash != version.sha256_hash:
        raise IntegrityError(
            f"Dataset snapshot integrity check failed! Expected SHA-256: {version.sha256_hash}, found: {current_hash}."
        )

    # Work Package A validation check
    latest_val_run = (
        db.query(models.ValidationRun)
        .filter(models.ValidationRun.dataset_version_id == dataset_version_id)
        .order_by(models.ValidationRun.created_at.desc())
        .first()
    )
    if not latest_val_run:
        raise ContractValidationError(
            f"Dataset version '{dataset_version_id}' has no validation runs. Validation is required before training."
        )

    if latest_val_run.status not in ("PASSED", "WARNING") or latest_val_run.error_count > 0:
        raise ContractValidationError(
            f"Dataset version '{dataset_version_id}' failed contract validation (status: {latest_val_run.status}, "
            f"blocking errors: {latest_val_run.error_count}). Training is prohibited on unvalidated data."
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise StorageError(f"Failed to read CSV dataset at {file_path}: {e}")

    return df, version, latest_val_run
