from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Standard metrics
HTTP_REQUESTS_TOTAL = Counter(
    "modelguard_http_requests_total",
    "Total HTTP requests received by endpoint and status code",
    ["method", "endpoint", "status_code"],
)

HTTP_ERRORS_TOTAL = Counter(
    "modelguard_errors_total",
    "Total operational and domain errors encountered",
    ["error_type", "endpoint"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "modelguard_inference_latency_seconds",
    "Inference execution latency in seconds",
    ["model_name", "model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ACTIVE_MODEL_VERSION = Gauge(
    "modelguard_active_model_version",
    "Currently active model version per environment",
    ["model_name", "environment", "model_version"],
)

MODEL_LOAD_FAILURES_TOTAL = Counter(
    "modelguard_model_load_failures_total",
    "Total failures encountered while loading model artifacts",
    ["model_name", "environment"],
)

COMPONENT_HEALTH = Gauge(
    "modelguard_component_health",
    "Health status of control plane components (1=healthy, 0=unhealthy)",
    ["component"],
)

DRIFT_DETECTED = Gauge(
    "modelguard_drift_detected",
    "Drift indicator status (1=drift detected, 0=normal)",
    ["model_name", "model_version", "drift_type"],
)

ROLLING_QUALITY_METRIC = Gauge(
    "modelguard_model_quality_metric",
    "Rolling evaluated quality score against delayed feedback",
    ["model_name", "model_version", "metric"],
)


def normalize_endpoint(path: str) -> str:
    """Normalize parameterized URL paths to prevent high-cardinality metric labels."""
    if not path:
        return "/"
    parts = path.strip("/").split("/")
    normalized = []
    for part in parts:
        # Check if part looks like a UUID or ID
        if len(part) >= 32 and ("-" in part or part.isalnum()):
            normalized.append("{id}")
        elif part.isdigit():
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized)


def track_request(method: str, endpoint: str, status_code: int) -> None:
    norm_ep = normalize_endpoint(endpoint)
    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=norm_ep, status_code=str(status_code)).inc()


def track_error(error_type: str, endpoint: str) -> None:
    norm_ep = normalize_endpoint(endpoint)
    HTTP_ERRORS_TOTAL.labels(error_type=error_type, endpoint=norm_ep).inc()


def track_inference(model_name: str, model_version: str, latency_seconds: float) -> None:
    INFERENCE_LATENCY_SECONDS.labels(model_name=model_name, model_version=model_version).observe(latency_seconds)


def set_active_model(model_name: str, environment: str, model_version: str) -> None:
    ACTIVE_MODEL_VERSION.labels(model_name=model_name, environment=environment, model_version=model_version).set(1.0)


def record_model_load_failure(model_name: str, environment: str) -> None:
    MODEL_LOAD_FAILURES_TOTAL.labels(model_name=model_name, environment=environment).inc()


def set_component_health(component: str, is_healthy: bool) -> None:
    COMPONENT_HEALTH.labels(component=component).set(1.0 if is_healthy else 0.0)


def set_drift_status(model_name: str, model_version: str, drift_type: str, detected: bool) -> None:
    DRIFT_DETECTED.labels(model_name=model_name, model_version=model_version, drift_type=drift_type).set(
        1.0 if detected else 0.0
    )


def record_quality_metric(model_name: str, model_version: str, metric_name: str, value: float) -> None:
    ROLLING_QUALITY_METRIC.labels(model_name=model_name, model_version=model_version, metric=metric_name).set(value)


def get_prometheus_metrics() -> tuple[bytes, str]:
    """Return raw Prometheus metrics bytes and appropriate Content-Type header."""
    return generate_latest(), CONTENT_TYPE_LATEST
