from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import EntityNotFoundError, InvalidOperationError
from app.core.logging import logger
from app.db import models


def utc_now() -> datetime:
    return datetime.now(UTC)


class AlertService:
    """Service managing persistent monitoring alerts, acknowledgements, and resolutions."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_alert(
        self,
        model_name: str,
        model_version: str,
        alert_type: str,
        severity: str,
        message: str,
        triggering_evidence: dict[str, Any],
        report_id: str | None = None,
    ) -> models.MonitoringAlert:
        """Create and persist a new monitoring alert."""
        alert = models.MonitoringAlert(
            report_id=report_id,
            model_name=model_name,
            model_version=model_version,
            alert_type=alert_type,
            severity=severity,
            message=message,
            triggering_evidence_json=triggering_evidence,
            status="OPEN",
            created_at=utc_now(),
        )
        self.db.add(alert)
        self.db.flush()
        logger.warning(f"Monitoring Alert [{severity}] for '{model_name}:v{model_version}' [{alert_type}]: {message}")
        return alert

    def get_alert(self, alert_id: str) -> models.MonitoringAlert:
        alert = self.db.query(models.MonitoringAlert).filter(models.MonitoringAlert.id == alert_id).first()
        if not alert:
            raise EntityNotFoundError(f"MonitoringAlert '{alert_id}' not found.")
        return alert

    def list_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        model_version: str | None = None,
    ) -> list[models.MonitoringAlert]:
        query = self.db.query(models.MonitoringAlert)
        if status:
            query = query.filter(models.MonitoringAlert.status == status)
        if severity:
            query = query.filter(models.MonitoringAlert.severity == severity)
        if model_version:
            query = query.filter(models.MonitoringAlert.model_version == model_version)
        return query.order_by(models.MonitoringAlert.created_at.desc()).all()

    def acknowledge_alert(self, alert_id: str, actor: str) -> models.MonitoringAlert:
        alert = self.get_alert(alert_id)
        if alert.status != "OPEN":
            raise InvalidOperationError(
                f"Alert '{alert_id}' with status '{alert.status}' cannot be acknowledged. Must be 'OPEN'."
            )
        now = utc_now()
        alert.status = "ACKNOWLEDGED"
        alert.acknowledged_by = actor
        alert.acknowledged_at = now

        audit = models.AuditEvent(
            actor=actor,
            action="ACKNOWLEDGE_ALERT",
            resource_type="monitoring_alert",
            resource_id=alert.id,
            before_state_json={"status": "OPEN"},
            after_state_json={"status": "ACKNOWLEDGED", "acknowledged_by": actor},
            reason="Operator acknowledged alert",
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(alert)
        logger.info(f"Alert '{alert.id}' ACKNOWLEDGED by '{actor}'.")
        return alert

    def resolve_alert(self, alert_id: str, actor: str, notes: str = "") -> models.MonitoringAlert:
        alert = self.get_alert(alert_id)
        if alert.status == "RESOLVED":
            raise InvalidOperationError(f"Alert '{alert_id}' is already RESOLVED.")
        now = utc_now()
        prev_status = alert.status
        alert.status = "RESOLVED"
        alert.resolved_by = actor
        alert.resolved_at = now
        alert.resolution_notes = notes

        audit = models.AuditEvent(
            actor=actor,
            action="RESOLVE_ALERT",
            resource_type="monitoring_alert",
            resource_id=alert.id,
            before_state_json={"status": prev_status},
            after_state_json={"status": "RESOLVED", "resolved_by": actor, "notes": notes},
            reason=notes or "Operator resolved alert",
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(alert)
        logger.info(f"Alert '{alert.id}' RESOLVED by '{actor}'.")
        return alert
