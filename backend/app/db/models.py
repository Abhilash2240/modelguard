import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.db.session import Base


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
    model_candidates = relationship("ModelCandidate", back_populates="dataset_version", cascade="all, delete-orphan")


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
    model_candidates = relationship("ModelCandidate", back_populates="training_run", cascade="all, delete-orphan")


class ModelCandidate(Base):
    """Registered ML candidate linked to an immutable MLflow model version."""

    __tablename__ = "model_candidates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    training_run_id = Column(
        String(36),
        ForeignKey("training_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_version_id = Column(
        String(36),
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mlflow_model_name = Column(String(100), nullable=False, index=True)
    mlflow_model_version = Column(String(50), nullable=False)
    model_uri = Column(String(500), nullable=False)
    artifact_uri = Column(String(500), nullable=False)
    signature_json = Column(JSON, nullable=True)
    input_contract_json = Column(JSON, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    gate_evaluation_json = Column(JSON, nullable=True)
    status = Column(String(30), default="REGISTERED", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    training_run = relationship("TrainingRun", back_populates="model_candidates")
    dataset_version = relationship("DatasetVersion", back_populates="model_candidates")
    deployments = relationship(
        "Deployment",
        foreign_keys="[Deployment.candidate_id]",
        back_populates="candidate",
        cascade="all, delete-orphan",
    )
    prediction_events = relationship("PredictionEvent", back_populates="candidate")


class DeploymentEnvironment(Base):
    """Deployment target environment pointer (e.g. staging, production)."""

    __tablename__ = "deployment_environments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(50), unique=True, nullable=False, index=True)  # staging, production
    active_candidate_id = Column(String(36), ForeignKey("model_candidates.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    active_candidate = relationship("ModelCandidate", foreign_keys=[active_candidate_id])
    deployments = relationship("Deployment", back_populates="environment", cascade="all, delete-orphan")


class Deployment(Base):
    """Audited staging and production deployment record."""

    __tablename__ = "deployments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    environment_id = Column(
        String(36),
        ForeignKey("deployment_environments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id = Column(
        String(36),
        ForeignKey("model_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    previous_candidate_id = Column(
        String(36),
        ForeignKey("model_candidates.id", ondelete="SET NULL"),
        nullable=True,
    )
    status = Column(String(30), default="STAGED", nullable=False, index=True)
    smoke_test_results_json = Column(JSON, nullable=True)
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approval_rationale = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    environment = relationship("DeploymentEnvironment", back_populates="deployments")
    candidate = relationship("ModelCandidate", foreign_keys=[candidate_id], back_populates="deployments")
    previous_candidate = relationship("ModelCandidate", foreign_keys=[previous_candidate_id])


class PredictionEvent(Base):
    """Privacy-safe inference telemetry record."""

    __tablename__ = "prediction_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    inference_id = Column(String(36), unique=True, nullable=False, index=True)
    candidate_id = Column(
        String(36),
        ForeignKey("model_candidates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    request_time = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    latency_ms = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)  # SUCCESS, ERROR
    prediction_json = Column(JSON, nullable=True)
    input_summary_json = Column(JSON, nullable=True)

    candidate = relationship("ModelCandidate", back_populates="prediction_events")
    feedback = relationship(
        "PredictionFeedback", uselist=False, back_populates="prediction_event", cascade="all, delete-orphan"
    )


class PredictionFeedback(Base):
    """Ground-truth feedback associated with an immutable inference record."""

    __tablename__ = "prediction_feedbacks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    inference_id = Column(
        String(36),
        ForeignKey("prediction_events.inference_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    actual_value = Column(Float, nullable=False)
    submitted_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    submitted_by = Column(String(100), default="system", nullable=False)
    notes = Column(Text, nullable=True)

    prediction_event = relationship("PredictionEvent", back_populates="feedback")


class ModelQualityReport(Base):
    """Persistent rolling model accuracy and error assessment from delayed feedback."""

    __tablename__ = "model_quality_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    sample_count = Column(Integer, nullable=False)
    labeled_sample_count = Column(Integer, nullable=False)
    label_coverage = Column(Float, nullable=False)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    status = Column(String(30), nullable=False)  # SUFFICIENT_LABELS, INSUFFICIENT_LABELS
    details_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class MonitoringReport(Base):
    """Persistent statistical drift or health monitoring report."""

    __tablename__ = "monitoring_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    report_type = Column(
        String(50), nullable=False, index=True
    )  # DATA_DRIFT, PREDICTION_DRIFT, MODEL_QUALITY, SYSTEM_HEALTH
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    reference_window_start = Column(DateTime(timezone=True), nullable=True)
    reference_window_end = Column(DateTime(timezone=True), nullable=True)
    current_window_start = Column(DateTime(timezone=True), nullable=True)
    current_window_end = Column(DateTime(timezone=True), nullable=True)
    reference_sample_count = Column(Integer, nullable=False)
    current_sample_count = Column(Integer, nullable=False)
    detection_method = Column(String(50), nullable=False)  # KS_TEST, PSI, CHI_SQUARE, TOTAL_VARIATION_DISTANCE
    policy_version = Column(String(20), default="v1.0.0", nullable=False)
    is_drift_detected = Column(Boolean, default=False, nullable=False)
    summary_json = Column(JSON, nullable=False)
    metrics_json = Column(JSON, nullable=True)
    limitations = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    alerts = relationship("MonitoringAlert", back_populates="report", cascade="all, delete-orphan")


class MonitoringAlert(Base):
    """Actionable threshold violation alert linking to evidence and model provenance."""

    __tablename__ = "monitoring_alerts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    report_id = Column(
        String(36),
        ForeignKey("monitoring_reports.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    alert_type = Column(
        String(50), nullable=False, index=True
    )  # DATA_DRIFT, PREDICTION_DRIFT, QUALITY_DEGRADATION, SERVING_FAILURE
    severity = Column(String(20), nullable=False)  # INFO, WARNING, CRITICAL
    message = Column(Text, nullable=False)
    triggering_evidence_json = Column(JSON, nullable=False)
    status = Column(String(20), default="OPEN", nullable=False, index=True)  # OPEN, ACKNOWLEDGED, RESOLVED
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    report = relationship("MonitoringReport", back_populates="alerts")


class UserRole(enum.StrEnum):
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"
    APPROVER = "APPROVER"
    ADMIN = "ADMIN"


class User(Base):
    """Persistent user account with role-based access control."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default=UserRole.VIEWER.value, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
