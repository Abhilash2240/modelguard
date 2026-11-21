"""Script to register the official UCI Metro Interstate Traffic Volume dataset in ModelGuard."""

import sys
from pathlib import Path

# Add backend directory to path
repo_root = Path(__file__).resolve().parent.parent
backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.schemas.dataset import DatasetCreate  # noqa: E402
from app.services.dataset_service import DatasetService  # noqa: E402


def register_metro_traffic() -> dict:
    source_csv = repo_root / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(f"Source dataset not found at {source_csv}")

    print("Initializing database tables...")
    init_db()

    db = SessionLocal()
    try:
        service = DatasetService(db)

        # 1. Register logical dataset if not already registered
        from app.db import models

        dataset = db.query(models.Dataset).filter(models.Dataset.name == "metro_interstate_traffic_volume").first()
        if not dataset:
            print("Registering logical dataset 'metro_interstate_traffic_volume'...")
            dataset_payload = DatasetCreate(
                name="metro_interstate_traffic_volume",
                task_type="regression",
                source_uri="https://archive.ics.uci.edu/dataset/492/urban+traffic",
                licence="CC BY 4.0",
                description="UCI Metro Interstate Traffic Volume hourly traffic volume benchmark (Dataset ID: 492)",
            )
            dataset = service.register_dataset(dataset_payload)
            print(f"Dataset registered: ID={dataset.id}, Name={dataset.name}")
        else:
            print(f"Dataset already registered: ID={dataset.id}, Name={dataset.name}")

        # 2. Check or create immutable version v1.0.0
        version = (
            db.query(models.DatasetVersion)
            .filter(
                models.DatasetVersion.dataset_id == dataset.id,
                models.DatasetVersion.version_tag == "v1.0.0",
            )
            .first()
        )
        if not version:
            print(f"Creating immutable dataset version 'v1.0.0' from {source_csv}...")
            version = service.create_dataset_version(
                dataset_id=dataset.id,
                version_tag="v1.0.0",
                source_file_path=source_csv,
            )
            print(f"Version registered: ID={version.id}")
        else:
            print(f"Version already exists: ID={version.id}")

        print(f"Version Tag       : {version.version_tag}")
        print(f"SHA-256 Checksum  : {version.sha256_hash}")
        print(f"Row Count         : {version.row_count:,}")
        print(f"Storage URI       : {version.storage_uri}")

        # 3. Validate registered snapshot
        print("Validating snapshot against MetroTrafficContractV1...")
        val_run = service.validate_version(
            version_id=version.id,
            contract_version="v1",
        )
        print(f"ValidationRun ID  : {val_run.id}")
        print(f"Validation Status : {val_run.status}")
        print(f"Blocking Errors   : {val_run.error_count}")
        print(f"Quality Warnings  : {val_run.warning_count}")

        return {
            "dataset_id": dataset.id,
            "version_id": version.id,
            "version_tag": version.version_tag,
            "sha256": version.sha256_hash,
            "row_count": version.row_count,
            "storage_uri": version.storage_uri,
            "validation_run_id": val_run.id,
            "validation_status": val_run.status,
            "error_count": val_run.error_count,
            "warning_count": val_run.warning_count,
        }
    finally:
        db.close()


if __name__ == "__main__":
    register_metro_traffic()
