import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models
from app.schemas.training import TrainingJobConfig, TrainingJobCreate, TrainingJobResponse


def test_training_job_and_run_orm_relationship(db: Session):
    # 1. Setup prerequisite Dataset, DatasetVersion, ValidationRun
    dataset = models.Dataset(
        name="metro_traffic_test",
        task_type="regression",
        source_uri="https://archive.ics.uci.edu/dataset/492/urban+traffic",
        licence="CC BY 4.0",
    )
    db.add(dataset)
    db.commit()

    version = models.DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.0",
        storage_uri="storage/datasets/test.csv",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        byte_size=1024,
        row_count=50,
        columns_json=[{"name": "temp", "dtype": "float64"}],
        schema_hash="schema_hash_test",
    )
    db.add(version)
    db.commit()

    val_run = models.ValidationRun(
        dataset_version_id=version.id,
        status="PASSED",
        error_count=0,
        warning_count=0,
        details_json={"findings": []},
    )
    db.add(val_run)
    db.commit()

    # 2. Create TrainingJob
    job = models.TrainingJob(
        dataset_version_id=version.id,
        validation_run_id=val_run.id,
        config_json={"model": "random_forest", "seed": 42},
        status="QUEUED",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    assert job.id is not None
    assert job.status == "QUEUED"
    assert job.dataset_version.id == version.id
    assert job.validation_run.id == val_run.id

    # Verify Pydantic serialization from ORM attributes
    job_resp = TrainingJobResponse.model_validate(job)
    assert job_resp.id == job.id
    assert job_resp.status == "QUEUED"

    # 3. Create TrainingRun linked to TrainingJob
    run = models.TrainingRun(
        training_job_id=job.id,
        dataset_version_id=version.id,
        git_sha="c185645a8b79234b455b89a31a9862955f1a9a88",
        dependency_lock_hash="lockfile_sha256_hash",
        python_version="3.13.7",
        random_seed=42,
        status="RUNNING",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    db.refresh(job)

    # 4. Verify bidirectional one-to-one relationship
    assert job.training_run is not None
    assert job.training_run.id == run.id
    assert run.training_job.id == job.id
    assert run.dataset_version.id == version.id

    # 5. Enforce uniqueness: another TrainingRun cannot share the same training_job_id
    duplicate_run = models.TrainingRun(
        training_job_id=job.id,
        dataset_version_id=version.id,
        python_version="3.13.7",
        random_seed=42,
        status="QUEUED",
    )
    db.add(duplicate_run)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    # 6. Test Cascade deletion: deleting training_job deletes associated training_run
    db.delete(job)
    db.commit()

    remaining_runs = db.query(models.TrainingRun).filter(models.TrainingRun.id == run.id).first()
    assert remaining_runs is None


def test_training_schemas():
    create_req = TrainingJobCreate(
        dataset_version_id="ver-123",
        config=TrainingJobConfig(
            model_candidate="random_forest",
            hyperparameters={"n_estimators": 100, "max_depth": 10},
        ),
    )
    assert create_req.dataset_version_id == "ver-123"
    assert create_req.config.split.train_ratio == 0.70
    assert create_req.config.hyperparameters["n_estimators"] == 100
