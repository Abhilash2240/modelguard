from pathlib import Path

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

    # Security
    ALLOWED_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Ensure required storage directories exist
settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
