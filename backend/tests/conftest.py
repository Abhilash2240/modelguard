from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user, get_db
from app.db import models
from app.db.session import Base
from app.main import app

# Use in-memory SQLite with StaticPool so all connections share the same database
TEST_DB_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    future=True,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine, future=True)


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db: Session) -> Generator[TestClient, None, None]:
    def override_get_db():
        try:
            yield db
        finally:
            pass

    admin_user = models.User(
        id="test-admin-uuid",
        username="test_admin",
        hashed_password="hashed_admin_pass",
        role=models.UserRole.ADMIN.value,
        is_active=True,
    )

    def override_get_current_user():
        return admin_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "datasets" / "fixtures"


@pytest.fixture
def valid_csv_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_metro_valid.csv"


@pytest.fixture
def invalid_csv_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_metro_invalid.csv"
