from fastapi import APIRouter, Response

from app.core.metrics import get_prometheus_metrics

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
def get_metrics() -> Response:
    """Expose application and model reliability metrics in Prometheus text format."""
    metrics_bytes, content_type = get_prometheus_metrics()
    return Response(content=metrics_bytes, media_type=content_type)
