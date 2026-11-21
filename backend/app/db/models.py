import uuid
from datetime import UTC, datetime

from app.db.session import Base
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class Dataset(Base):
    """Logical dataset identity."""

    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), unique=True, nullable=False, index=True)
    task_type = Column(String(50), nullable=False)  # regression, classification
    source_uri = Column(String(255), nullable=False)
    licence = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    contracts = relationship("DataContract", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    """Immutable dataset snapshot."""

    __tablename__ = "dataset_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_tag = Column(String(50), nullable=False)  # e.g., 'v1.0.0'
    storage_uri = Column(String(500), nullable=False)
    sha256_hash = Column(String(64), nullable=False, index=True)
    byte_size = Column(BigInteger, nullable=False)
    row_count = Column(Integer, nullable=False)
    columns_json = Column(JSON, nullable=False)  # list of column names & dtypes
    schema_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    dataset = relationship("Dataset", back_populates="versions")
    validation_runs = relationship("ValidationRun", back_populates="dataset_version", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="dataset_version", cascade="all, delete-orphan")
    training_runs = relationship("TrainingRun", back_populates="dataset_version", cascade="all, delete-orphan")


class DataContract(Base):
    """Versioned schema contract for validation."""

    __tablename__ = "data_contracts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_tag = Column(String(50), nullable=False)  # e.g., 'v1'
    contract_schema_json = Column(JSON, nullable=False)  # schema description, checks, bounds
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    dataset = relationship("Dataset", back_populates="contracts")
    validation_runs = relationship("ValidationRun", back_populates="contract")


class ValidationRun(Base):
    """Auditable evidence of dataset contract validation execution."""

    __tablename__ = "validation_runs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_version_id = Column(
        String(36),
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    contract_id = Column(String(36), ForeignKey("data_contracts.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(20), nullable=False)  # PASSED, FAILED, WARNING
    error_count = Column(Integer, default=0, nullable=False)
    warning_count = Column(Integer, default=0, nullable=False)
    details_json = Column(JSON, nullable=False)  # Detailed check-by-check breakdown & sample failures
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    dataset_version = relationship("DatasetVersion", back_populates="validation_runs")
    contract = relationship("DataContract", back_populates="validation_runs")
    training_jobs = relationship("TrainingJob", back_populates="validation_run")


class AuditEvent(Base):
    """Immutable audit trail for all control plane actions."""

    __tablename__ = "audit_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    actor = Column(String(100), nullable=False)  # system, user, or role
    action = Column(String(100), nullable=False)  # REGISTER_DATASET, VALIDATE_DATASET, etc.
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(36), nullable=False, index=True)
    before_state_json = Column(JSON, nullable=True)
    after_state_json = Column(JSON, nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class TrainingJob(Base):
    """Durable training job state."""

    __tablename__ = "training_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_version_id = Column(
        String(36),
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    validation_run_id = Column(
        String(36),
        ForeignKey("validation_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    config_json = Column(JSON, nullable=False)
    status = Column(String(20), default="QUEUED", nullable=False)  # QUEUED, RUNNING, SUCCEEDED, FAILED
    queue_job_id = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    dataset_version = relationship("DatasetVersion", back_populates="training_jobs")
    validation_run = relationship("ValidationRun", back_populates="training_jobs")
    training_run = relationship(
        "TrainingRun",
        back_populates="training_job",
        uselist=False,
        cascade="all, delete-orphan",
    )


class TrainingRun(Base):
    """Reproducible training execution metadata."""

    __tablename__ = "training_runs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    training_job_id = Column(
        String(36),
        ForeignKey("training_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    mlflow_run_id = Column(String(100), nullable=True)
    dataset_version_id = Column(
        String(36),
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    git_sha = Column(String(40), nullable=True)
    dependency_lock_hash = Column(String(64), nullable=True)
    python_version = Column(String(50), nullable=False)
    random_seed = Column(Integer, default=42, nullable=False)
    split_manifest_uri = Column(String(500), nullable=True)
    metrics_json = Column(JSON, nullable=True)
    artifact_uri = Column(String(500), nullable=True)
    status = Column(String(20), default="QUEUED", nullable=False)  # QUEUED, RUNNING, SUCCEEDED, FAILED
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    training_job = relationship("TrainingJob", back_populates="training_run")
    dataset_version = relationship("DatasetVersion", back_populates="training_runs")
