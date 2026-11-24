import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token, get_password_hash
from app.db import models
from app.main import app


@pytest.fixture
def clean_client(db: Session):
    """TestClient without any auth dependency overrides to test genuine authentication."""
    app.dependency_overrides.clear()
    from app.db.session import get_db

    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def seeded_users(db: Session):
    """Seed users for all four RBAC roles."""
    users = {}
    for role in [models.UserRole.VIEWER, models.UserRole.OPERATOR, models.UserRole.APPROVER, models.UserRole.ADMIN]:
        user = models.User(
            username=f"user_{role.value.lower()}",
            hashed_password=get_password_hash("Password123!"),
            role=role.value,
            is_active=True,
        )
        db.add(user)
        users[role.value] = user

    # Seed inactive user
    inactive = models.User(
        username="user_inactive",
        hashed_password=get_password_hash("Password123!"),
        role=models.UserRole.VIEWER.value,
        is_active=False,
    )
    db.add(inactive)
    users["INACTIVE"] = inactive

    db.commit()
    return users


def test_unauthenticated_request_rejected(clean_client: TestClient):
    """Unauthenticated requests to protected endpoints return 401 Unauthorized."""
    res = clean_client.get("/api/v1/datasets")
    assert res.status_code == 401
    assert "detail" in res.json()


def test_login_successful_and_cookie_set(clean_client: TestClient, seeded_users):
    """Valid credentials return JWT token and set HttpOnly session cookie."""
    res = clean_client.post(
        "/api/v1/auth/login",
        json={"username": "user_operator", "password": "Password123!"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "user_operator"
    assert data["user"]["role"] == "OPERATOR"
    assert "mg_access_token" in res.cookies


def test_login_invalid_password_rejected(clean_client: TestClient, seeded_users):
    """Invalid credentials return 401 Unauthorized."""
    res = clean_client.post(
        "/api/v1/auth/login",
        json={"username": "user_operator", "password": "WrongPassword!"},
    )
    assert res.status_code == 401


def test_login_inactive_user_rejected(clean_client: TestClient, seeded_users):
    """Inactive users are rejected with 403 Forbidden."""
    res = clean_client.post(
        "/api/v1/auth/login",
        json={"username": "user_inactive", "password": "Password123!"},
    )
    assert res.status_code == 403


def test_auth_me_endpoint(clean_client: TestClient, seeded_users):
    """Authenticated user can inspect own profile and role."""
    token = create_access_token("user_approver", "APPROVER")
    res = clean_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["username"] == "user_approver"
    assert res.json()["role"] == "APPROVER"


def test_rbac_viewer_forbidden_on_write(clean_client: TestClient, seeded_users):
    """VIEWER role is permitted to read datasets but forbidden from registering datasets."""
    token = create_access_token("user_viewer", "VIEWER")
    headers = {"Authorization": f"Bearer {token}"}

    # Reading is allowed
    read_res = clean_client.get("/api/v1/datasets", headers=headers)
    assert read_res.status_code == 200

    # Writing is forbidden (403)
    write_res = clean_client.post(
        "/api/v1/datasets",
        headers=headers,
        json={"name": "test_ds", "task_type": "regression", "source_uri": "s3://test", "licence": "MIT"},
    )
    assert write_res.status_code == 403


def test_rbac_operator_forbidden_from_promotion(clean_client: TestClient, seeded_users, db: Session):
    """OPERATOR role cannot approve or promote deployments; requires APPROVER or ADMIN."""
    token = create_access_token("user_operator", "OPERATOR")
    headers = {"Authorization": f"Bearer {token}"}

    # Attempt to promote
    res = clean_client.post("/api/v1/deployments/fake-id/promote", headers=headers)
    assert res.status_code == 403


def test_actor_impersonation_prevented(clean_client: TestClient, seeded_users, db: Session):
    """Passing a forged actor in request payload is overridden by authenticated user identity."""
    # Seed deployment environment and candidate
    env = models.DeploymentEnvironment(id="env-staging", name="staging")
    db.add(env)
    candidate = models.ModelCandidate(
        id="cand-impersonation-test",
        training_run_id="run-1",
        dataset_version_id="ver-1",
        mlflow_model_name="metro-traffic-volume",
        mlflow_model_version="1",
        model_uri="runs:/fake/model",
        artifact_uri="runs:/fake/artifact",
        status="STAGED",
    )
    db.add(candidate)
    dep = models.Deployment(
        id="dep-impersonation-test",
        environment_id=env.id,
        candidate_id=candidate.id,
        status="AWAITING_APPROVAL",
        smoke_test_results_json={"passed": True},
    )
    db.add(dep)
    db.commit()

    token = create_access_token("user_approver", "APPROVER")
    headers = {"Authorization": f"Bearer {token}"}

    # Caller attempts to impersonate "superadmin" in body
    res = clean_client.post(
        f"/api/v1/deployments/{dep.id}/approve",
        headers=headers,
        json={"actor": "forged_superadmin", "rationale": "Verified"},
    )
    assert res.status_code == 200

    # Verify audit event and deployment records the authenticated user, NOT the forged actor
    db.refresh(dep)
    events = db.query(models.AuditEvent).filter(models.AuditEvent.resource_id == dep.id).all()
    assert any(ev.actor == "user_approver" for ev in events)
    assert not any(ev.actor == "forged_superadmin" for ev in events)
