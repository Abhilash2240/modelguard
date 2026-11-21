from app.api.v1.routes import datasets, health
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router)
