from fastapi import APIRouter

from app.api.v1.routes import datasets, deployments, health, inference, models

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(models.router)
api_router.include_router(deployments.router)
api_router.include_router(inference.router)
