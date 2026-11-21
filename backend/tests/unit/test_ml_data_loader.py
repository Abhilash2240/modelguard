import tempfile
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.errors import ContractValidationError, EntityNotFoundError, IntegrityError
from app.db import models
from app.domain.hashing import calculate_sha256
from app.ml.data_loader import load_validated_dataset


def test_load_validated_dataset_success(db: Session, valid_csv_path: Path):
    # 1. Setup clean validated dataset version
    sha256 = calculate_sha256(valid_csv_path)
    dataset = models.Dataset(
        name="metro_traffic_test",
        task_type="regression",
        source_uri="https://archive.ics.uci.edu",
        licence="CC BY 4.0",
    )
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.0",
        storage_uri=str(valid_csv_path),
        sha256_hash=sha256,
        byte_size=valid_csv_path.stat().st_size,
        row_count=10,
        columns_json=[],
        schema_hash="hash",
    )
    db.add(version)
    db.commit()

    val_run = models.ValidationRun(
        dataset_version_id=version.id,
        status="PASSED",
        error_count=0,
        warning_count=0,
        details_json={},
    )
    db.add(val_run)
    db.commit()

    # Load should succeed
    df, v_out, vr_out = load_validated_dataset(db, version.id)
    assert len(df) == 10
    assert v_out.id == version.id
    assert vr_out.status == "PASSED"


def test_load_dataset_missing_version_raises(db: Session):
    with pytest.raises(EntityNotFoundError):
        load_validated_dataset(db, "non-existent-uuid")


def test_load_dataset_tampered_hash_raises(db: Session):
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".csv") as f:
        f.write(b"a,b\n1,2\n")
        f_path = Path(f.name)

    try:
        dataset = models.Dataset(
            name="tamper_test",
            task_type="regression",
            source_uri="https://test.org",
            licence="MIT",
        )
        db.add(dataset)
        db.commit()

        # Record a bogus SHA-256
        version = models.DatasetVersion(
            dataset_id=dataset.id,
            version_tag="v1.0.0",
            storage_uri=str(f_path),
            sha256_hash="0000000000000000000000000000000000000000000000000000000000000000",
            byte_size=8,
            row_count=1,
            columns_json=[],
            schema_hash="h",
        )
        db.add(version)
        db.commit()

        with pytest.raises(IntegrityError, match="integrity check failed"):
            load_validated_dataset(db, version.id)
    finally:
        if f_path.exists():
            f_path.unlink()


def test_load_dataset_unvalidated_raises(db: Session, valid_csv_path: Path):
    sha256 = calculate_sha256(valid_csv_path)
    dataset = models.Dataset(
        name="unvalidated_test",
        task_type="regression",
        source_uri="https://test.org",
        licence="MIT",
    )
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.0",
        storage_uri=str(valid_csv_path),
        sha256_hash=sha256,
        byte_size=valid_csv_path.stat().st_size,
        row_count=10,
        columns_json=[],
        schema_hash="h",
    )
    db.add(version)
    db.commit()

    # Case 1: No validation run at all
    with pytest.raises(ContractValidationError, match="has no validation runs"):
        load_validated_dataset(db, version.id)

    # Case 2: Validation run status is FAILED
    failed_run = models.ValidationRun(
        dataset_version_id=version.id,
        status="FAILED",
        error_count=2,
        warning_count=0,
        details_json={},
    )
    db.add(failed_run)
    db.commit()

    with pytest.raises(ContractValidationError, match="failed contract validation"):
        load_validated_dataset(db, version.id)
