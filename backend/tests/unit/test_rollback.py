from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.errors import InvalidOperationError, RollbackError
from app.db import models
from app.services.deployment_service import DeploymentService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    prod_env = models.DeploymentEnvironment(name="production")
    staging_env = models.DeploymentEnvironment(name="staging")
    session.add_all([prod_env, staging_env])
    session.commit()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_candidates(db_session):
    cand1 = models.ModelCandidate(
        id="cand-uuid-1",
        training_run_id="run-1",
        dataset_version_id="ds-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="models:/metro-traffic-volume/1",
        artifact_uri="storage/models/1",
        status="ARCHIVED",
    )
    cand2 = models.ModelCandidate(
        id="cand-uuid-2",
        training_run_id="run-2",
        dataset_version_id="ds-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="2",
        model_uri="models:/metro-traffic-volume/2",
        artifact_uri="storage/models/2",
        status="PROMOTED",
    )
    db_session.add_all([cand1, cand2])
    db_session.flush()

    prod_env = (
        db_session.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == "production").first()
    )
    prod_env.active_candidate_id = cand2.id

    # Add historical deployment for cand1
    dep1 = models.Deployment(
        id="dep-1",
        environment_id=prod_env.id,
        candidate_id=cand1.id,
        status="PROMOTED",
        approved_by="admin",
        approval_rationale="Initial deployment v1",
    )
    # Add active deployment for cand2
    dep2 = models.Deployment(
        id="dep-2",
        environment_id=prod_env.id,
        candidate_id=cand2.id,
        previous_candidate_id=cand1.id,
        status="PROMOTED",
        approved_by="admin",
        approval_rationale="Promoted v2",
    )
    db_session.add_all([dep1, dep2])
    db_session.commit()

    return cand1, cand2, dep1, dep2


def test_rollback_success(db_session, sample_candidates):
    """Test successful rollback from v2 back to v1 with MLflow alias update and audit recording."""
    cand1, cand2, dep1, dep2 = sample_candidates
    service = DeploymentService(db_session)

    mock_client = mock.MagicMock()
    mock_model = mock.MagicMock()
    mock_model.predict.return_value = [3400.0]

    with (
        mock.patch.object(service, "_get_client", return_value=mock_client),
        mock.patch("mlflow.sklearn.load_model", return_value=mock_model),
    ):
        new_dep = service.rollback_deployment(
            deployment_id=dep2.id,
            actor="ops_engineer",
            reason="High regression observed on morning traffic patterns",
        )

        assert new_dep.candidate_id == cand1.id
        assert new_dep.previous_candidate_id == cand2.id
        assert new_dep.status == "PROMOTED"
        assert "ROLLBACK from v2 to v1" in new_dep.approval_rationale

        # Verify MLflow alias update was called for v1
        mock_client.set_registered_model_alias.assert_called_with(
            name="metro-traffic-volume",
            alias="champion",
            version="1",
        )

        # Verify production environment active candidate updated to cand1
        prod_env = (
            db_session.query(models.DeploymentEnvironment)
            .filter(models.DeploymentEnvironment.name == "production")
            .first()
        )
        assert prod_env.active_candidate_id == cand1.id

        # Verify candidate statuses
        db_session.refresh(cand1)
        db_session.refresh(cand2)
        assert cand1.status == "PROMOTED"
        assert cand2.status == "ROLLED_BACK"

        # Verify audit trail
        audit = (
            db_session.query(models.AuditEvent).filter(models.AuditEvent.action == "ROLLBACK_MODEL_CANDIDATE").first()
        )
        assert audit is not None
        assert audit.actor == "ops_engineer"
        assert audit.before_state_json["active_version"] == "2"
        assert audit.after_state_json["active_version"] == "1"

        # Verify historical deployments were NOT deleted
        all_deps = db_session.query(models.Deployment).all()
        assert len(all_deps) == 3


def test_rollback_to_already_active_model_rejected(db_session, sample_candidates):
    """Verify rollback cannot target a model version that is already active in production."""
    cand1, cand2, dep1, dep2 = sample_candidates
    service = DeploymentService(db_session)

    with pytest.raises(InvalidOperationError) as exc_info:
        service.rollback_deployment(
            deployment_id=dep2.id,
            actor="ops_engineer",
            reason="Invalid target",
            target_candidate_id=cand2.id,  # cand2 is already active
        )
    assert "already active in production" in str(exc_info.value)


def test_rollback_fails_if_artifact_corrupt(db_session, sample_candidates):
    """Verify pre-switch smoke check aborts rollback if target model cannot be loaded."""
    cand1, cand2, dep1, dep2 = sample_candidates
    service = DeploymentService(db_session)

    mock_client = mock.MagicMock()

    with (
        mock.patch.object(service, "_get_client", return_value=mock_client),
        mock.patch("mlflow.sklearn.load_model", side_effect=RuntimeError("Corrupted pickle")),
    ):
        with pytest.raises(RollbackError) as exc_info:
            service.rollback_deployment(
                deployment_id=dep2.id,
                actor="ops_engineer",
                reason="Corrupted candidate test",
            )
        assert "Pre-rollback verification failed" in str(exc_info.value)

        # Ensure MLflow alias was NOT changed
        mock_client.set_registered_model_alias.assert_not_called()


def test_rollback_compensation_on_db_failure(db_session, sample_candidates):
    """Verify that if database transaction fails after MLflow alias update, MLflow alias is compensated."""
    cand1, cand2, dep1, dep2 = sample_candidates
    service = DeploymentService(db_session)

    mock_client = mock.MagicMock()
    mock_model = mock.MagicMock()
    mock_model.predict.return_value = [3000.0]

    with (
        mock.patch.object(service, "_get_client", return_value=mock_client),
        mock.patch("mlflow.sklearn.load_model", return_value=mock_model),
        mock.patch.object(db_session, "commit", side_effect=RuntimeError("DB Lock Error")),
    ):
        with pytest.raises(RollbackError) as exc_info:
            service.rollback_deployment(
                deployment_id=dep2.id,
                actor="ops_engineer",
                reason="DB failure test",
            )
        assert "Rollback aborted and compensated" in str(exc_info.value)

        # Check call sequence: alias set to "1" then compensated back to "2"
        mock_client.set_registered_model_alias.assert_any_call(
            name="metro-traffic-volume",
            alias="champion",
            version="1",
        )
        mock_client.set_registered_model_alias.assert_called_with(
            name="metro-traffic-volume",
            alias="champion",
            version="2",
        )
