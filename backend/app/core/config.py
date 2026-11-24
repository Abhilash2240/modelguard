import json
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ModelGuard Control Plane"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "0.1.0"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "sqlite:///./modelguard.db"

    # Storage & Uploads
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    DATASETS_DIR: Path = STORAGE_DIR / "datasets"
    MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024  # 100MB

    # MLflow Tracking & Artifact Storage (Absolute & Cross-Platform Normalized)
    MLFLOW_ENABLED: bool = True
    MLFLOW_EXPERIMENT_NAME: str = "metro-interstate-traffic-volume"
    MLFLOW_MODEL_NAME: str = "metro-traffic-volume"
    MLFLOW_DIR: Path = STORAGE_DIR / "mlflow"
    MLFLOW_DB_PATH: Path = STORAGE_DIR / "mlflow" / "mlflow.db"
    MLFLOW_ARTIFACTS_DIR: Path = STORAGE_DIR / "mlflow" / "artifacts"
    MLFLOW_TRACKING_URI: str = f"sqlite:///{(STORAGE_DIR / 'mlflow' / 'mlflow.db').resolve().as_posix()}"
    MLFLOW_ARTIFACT_ROOT: str = (STORAGE_DIR / "mlflow" / "artifacts").resolve().as_uri()

    # Security & Authentication
    ALLOWED_ORIGINS: list[str] = ["*"]
    SECRET_KEY: str = "modelguard-insecure-development-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    BOOTSTRAP_ADMIN_USERNAME: str | None = None
    BOOTSTRAP_ADMIN_PASSWORD: str | None = None

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    return json.loads(v_stripped)
                except Exception:
                    pass
            return [i.strip() for i in v_stripped.split(",") if i.strip()]
        elif isinstance(v, list):
            return [str(i) for i in v]
        return [str(v)]

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.ENVIRONMENT.lower() == "production":
            insecure_markers = [
                "change-in-production",
                "insecure",
                "development",
                "placeholder",
                "change_this",
            ]
            if len(self.SECRET_KEY) < 32 or any(m in self.SECRET_KEY.lower() for m in insecure_markers):
                raise ValueError(
                    "In production environment, SECRET_KEY must be a cryptographically secure key of at least 32 characters."
                )
            if self.BOOTSTRAP_ADMIN_PASSWORD:
                weak_passwords = [
                    "admin",
                    "password",
                    "modelguard",
                    "admin123",
                    "root",
                    "change_this_to_a_secure_admin_password",
                ]
                if len(self.BOOTSTRAP_ADMIN_PASSWORD) < 8 or self.BOOTSTRAP_ADMIN_PASSWORD.lower() in weak_passwords:
                    raise ValueError(
                        "In production environment, BOOTSTRAP_ADMIN_PASSWORD must be at least 8 characters and not a default or placeholder password."
                    )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Ensure required storage directories exist
settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
