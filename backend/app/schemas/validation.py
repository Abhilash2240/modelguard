from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class CheckFinding(BaseModel):
    column: str | None = None
    check: str
    severity: Literal["BLOCKING", "WARNING"]
    failure_count: int
    sample_cases: list[Any] = []
    message: str


class ValidationSummary(BaseModel):
    status: Literal["PASSED", "FAILED", "WARNING"]
    dataset_version_id: str
    contract_version: str
    total_records: int
    error_count: int
    warning_count: int
    findings: list[CheckFinding] = []
    validated_at: datetime = Field(default_factory=utc_now)


class DataContractCreate(BaseModel):
    version_tag: str = Field(..., description="Contract version tag, e.g. v1")
    contract_schema_json: dict[str, Any] = Field(
        ..., description="Machine-readable contract definitions"
    )


class DataContractResponse(BaseModel):
    id: str
    dataset_id: str
    version_tag: str
    contract_schema_json: dict[str, Any]
    is_active: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ValidationRunResponse(BaseModel):
    id: str
    dataset_version_id: str
    contract_id: str | None
    status: str
    error_count: int
    warning_count: int
    details_json: dict[str, Any]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
