from fastapi import APIRouter

from app.api.v1.routes import alerts, datasets, deployments, drift, feedback, health, inference, metrics, models, training

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(training.router)
api_router.include_router(models.router)
api_router.include_router(deployments.router)
api_router.include_router(inference.router)
api_router.include_router(metrics.router)
api_router.include_router(feedback.router)
api_router.include_router(drift.router)
api_router.include_router(alerts.router)
