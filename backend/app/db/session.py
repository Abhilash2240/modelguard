from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import settings

# Engine configuration (compatible with SQLite for tests & PostgreSQL for production)
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args=connect_args,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables, seed default environments, and bootstrap admin if configured."""
    from app.core.security import get_password_hash
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for env_name in ("staging", "production"):
            env = db.query(models.DeploymentEnvironment).filter(models.DeploymentEnvironment.name == env_name).first()
            if not env:
                db.add(models.DeploymentEnvironment(name=env_name))

        # Bootstrap initial administrator only if explicitly configured in environment
        if settings.BOOTSTRAP_ADMIN_USERNAME and settings.BOOTSTRAP_ADMIN_PASSWORD:
            existing = db.query(models.User).filter(models.User.username == settings.BOOTSTRAP_ADMIN_USERNAME).first()
            if not existing:
                admin_user = models.User(
                    username=settings.BOOTSTRAP_ADMIN_USERNAME,
                    hashed_password=get_password_hash(settings.BOOTSTRAP_ADMIN_PASSWORD),
                    role=models.UserRole.ADMIN.value,
                    is_active=True,
                )
                db.add(admin_user)
        db.commit()
    finally:
        db.close()
