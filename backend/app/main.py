from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core import metrics
from app.core.config import settings
from app.core.logging import logger
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and directories
    logger.info("Initializing ModelGuard Control Plane...")
    init_db()
    settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.set_component_health("database", True)
    metrics.set_component_health("storage", True)
    logger.info("ModelGuard startup complete.")
    yield
    # Shutdown
    logger.info("Shutting down ModelGuard Control Plane.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ModelGuard — Production ML Reliability Control Plane",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        status_code = response.status_code
        metrics.track_request(request.method, request.url.path, status_code)
        if status_code >= 400:
            metrics.track_error(f"HTTP_{status_code}", request.url.path)
        return response
    except Exception as exc:
        metrics.track_request(request.method, request.url.path, 500)
        metrics.track_error(type(exc).__name__, request.url.path)
        raise exc


@app.get("/metrics", tags=["Metrics"])
def metrics_endpoint() -> Response:
    """Root Prometheus metrics exposition endpoint for standard scrape configurations."""
    content, content_type = metrics.get_prometheus_metrics()
    return Response(content=content, media_type=content_type)


# Register API v1 routes
app.include_router(api_router, prefix=settings.API_V1_STR)
