from app.core import metrics


def test_endpoint_normalization():
    """Verify endpoint normalization removes dynamic UUIDs to avoid high cardinality."""
    assert metrics.normalize_endpoint("/api/v1/models") == "/api/v1/models"
    assert (
        metrics.normalize_endpoint("/api/v1/deployments/550e8400-e29b-41d4-a716-446655440000/approve")
        == "/api/v1/deployments/{id}/approve"
    )
    assert metrics.normalize_endpoint("/api/v1/alerts/123/resolve") == "/api/v1/alerts/{id}/resolve"


def test_prometheus_metrics_generation():
    """Verify prometheus metric exposition output contains expected metric series."""
    metrics.track_request("GET", "/api/v1/models", 200)
    metrics.track_error("ValidationError", "/api/v1/predict")
    metrics.track_inference("metro-traffic-volume", "1", 0.025)
    metrics.set_active_model("metro-traffic-volume", "production", "1")
    metrics.record_model_load_failure("metro-traffic-volume", "production")
    metrics.set_component_health("database", True)
    metrics.set_drift_status("metro-traffic-volume", "1", "data_drift", False)

    content, content_type = metrics.get_prometheus_metrics()
    assert isinstance(content, bytes)
    assert "text/plain" in content_type

    text = content.decode("utf-8")
    assert "modelguard_http_requests_total" in text
    assert "modelguard_errors_total" in text
    assert "modelguard_inference_latency_seconds" in text
    assert "modelguard_active_model_version" in text
    assert "modelguard_model_load_failures_total" in text
    assert "modelguard_component_health" in text


def test_no_sensitive_payload_in_metrics():
    """Confirm metrics output does not expose raw feature values or customer inputs."""
    content, _ = metrics.get_prometheus_metrics()
    text = content.decode("utf-8")
    # Verify no raw payload keys or sensitive feature values appear in metric labels
    assert "holiday=" not in text
    assert "date_time=" not in text
