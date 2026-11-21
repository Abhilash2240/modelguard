from app.core.config import settings
from fastapi import APIRouter, status

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
def readiness_check():
    """Readiness probe checking storage and service status."""
    storage_accessible = settings.DATASETS_DIR.exists()
    return {
        "status": "ready" if storage_accessible else "degraded",
        "storage_accessible": storage_accessible,
    }
