import shutil
from pathlib import Path

import pandas as pd
from app.core.config import settings
from app.core.errors import EntityNotFoundError, IntegrityError, InvalidOperationError, StorageError
from app.core.logging import logger
from app.db import models
from app.domain.hashing import calculate_schema_hash, calculate_sha256
from app.domain.validation_service import validate_dataset_file
from app.schemas.dataset import DatasetCreate
from sqlalchemy.orm import Session


class DatasetService:
    def __init__(self, db: Session):
        self.db = db

    def register_dataset(self, data: DatasetCreate) -> models.Dataset:
        existing = self.db.query(models.Dataset).filter(models.Dataset.name == data.name).first()
        if existing:
            raise InvalidOperationError(f"Dataset with name '{data.name}' already exists.")

        dataset = models.Dataset(
            name=data.name,
            task_type=data.task_type,
            source_uri=data.source_uri,
            licence=data.licence,
            description=data.description,
        )
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)

        # Audit event
        audit = models.AuditEvent(
            actor="system",
            action="REGISTER_DATASET",
            resource_type="dataset",
            resource_id=dataset.id,
            after_state_json={"name": dataset.name, "task_type": dataset.task_type},
            reason="Initial dataset metadata registration",
        )
        self.db.add(audit)
        self.db.commit()

        logger.info(f"Registered dataset '{dataset.name}' (id: {dataset.id})")
        return dataset

    def get_dataset(self, dataset_id: str) -> models.Dataset:
        dataset = self.db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
        if not dataset:
            raise EntityNotFoundError(f"Dataset with id '{dataset_id}' not found.")
        return dataset

    def list_datasets(self) -> list[models.Dataset]:
        return self.db.query(models.Dataset).all()

    def create_dataset_version(
        self,
        dataset_id: str,
        version_tag: str,
        source_file_path: Path,
    ) -> models.DatasetVersion:
        dataset = self.get_dataset(dataset_id)

        # Check for duplicate version tag
        existing_version = (
            self.db.query(models.DatasetVersion)
            .filter(
                models.DatasetVersion.dataset_id == dataset_id,
                models.DatasetVersion.version_tag == version_tag,
            )
            .first()
        )
        if existing_version:
            raise InvalidOperationError(
                f"Version '{version_tag}' already exists for dataset '{dataset.name}'. Immutable versions cannot be replaced."
            )

        if not source_file_path.is_file():
            raise StorageError(f"Source file not found at: {source_file_path}")

        # Compute hash and size
        sha256 = calculate_sha256(source_file_path)
        byte_size = source_file_path.stat().st_size

        # Inspect dataframe columns & rows
        try:
            df = pd.read_csv(source_file_path)
        except Exception as e:
            raise StorageError(f"Could not parse CSV file: {e}")

        row_count = len(df)
        columns_info = [{"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
        schema_hash = calculate_schema_hash(columns_info)

        # Persist immutable copy in storage directory
        target_filename = f"{dataset.name}_{version_tag}_{sha256[:8]}.csv"
        target_path = settings.DATASETS_DIR / target_filename
        shutil.copy2(source_file_path, target_path)

        version_record = models.DatasetVersion(
            dataset_id=dataset.id,
            version_tag=version_tag,
            storage_uri=str(target_path),
            sha256_hash=sha256,
            byte_size=byte_size,
            row_count=row_count,
            columns_json=columns_info,
            schema_hash=schema_hash,
        )
        self.db.add(version_record)
        self.db.commit()
        self.db.refresh(version_record)

        # Audit event
        audit = models.AuditEvent(
            actor="system",
            action="CREATE_DATASET_VERSION",
            resource_type="dataset_version",
            resource_id=version_record.id,
            after_state_json={
                "version_tag": version_tag,
                "sha256": sha256,
                "rows": row_count,
            },
            reason="Created immutable dataset snapshot",
        )
        self.db.add(audit)
        self.db.commit()

        logger.info(
            f"Created immutable snapshot {version_tag} for {dataset.name} (SHA-256: {sha256[:8]}..., rows: {row_count})"
        )
        return version_record

    def get_dataset_version(self, version_id: str) -> models.DatasetVersion:
        version = (
            self.db.query(models.DatasetVersion)
            .filter(models.DatasetVersion.id == version_id)
            .first()
        )
        if not version:
            raise EntityNotFoundError(f"Dataset version with id '{version_id}' not found.")
        return version

    def validate_version(
        self,
        version_id: str,
        contract_version: str = "v1",
    ) -> models.ValidationRun:
        version = self.get_dataset_version(version_id)
        dataset = self.get_dataset(version.dataset_id)

        # Verify physical file integrity before validating
        file_path = Path(version.storage_uri)
        current_hash = calculate_sha256(file_path)
        if current_hash != version.sha256_hash:
            raise IntegrityError(
                f"Checksum mismatch for version {version.version_tag}! Recorded: {version.sha256_hash}, Found: {current_hash}"
            )

        # Execute contract validation
        summary = validate_dataset_file(
            file_path=file_path,
            dataset_name=dataset.name,
            dataset_version_id=version.id,
            contract_version=contract_version,
        )

        # Store validation run
        validation_run = models.ValidationRun(
            dataset_version_id=version.id,
            status=summary.status,
            error_count=summary.error_count,
            warning_count=summary.warning_count,
            details_json=summary.model_dump(mode="json"),
        )
        self.db.add(validation_run)
        self.db.commit()
        self.db.refresh(validation_run)

        # Audit event
        audit = models.AuditEvent(
            actor="system",
            action="VALIDATE_DATASET_VERSION",
            resource_type="validation_run",
            resource_id=validation_run.id,
            after_state_json={
                "status": summary.status,
                "error_count": summary.error_count,
                "warning_count": summary.warning_count,
            },
            reason=f"Executed Pandera contract validation ({contract_version})",
        )
        self.db.add(audit)
        self.db.commit()

        logger.info(
            f"Validation complete for version {version.version_tag}: status={summary.status}, errors={summary.error_count}, warnings={summary.warning_count}"
        )
        return validation_run
