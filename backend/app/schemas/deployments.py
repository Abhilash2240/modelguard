from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SmokeTestResult(BaseModel):
    passed: bool
    model_loaded: bool
    sample_input_valid: bool
    sample_output_finite: bool
    latency_ms: float
    output_sample: list[float] | None = None
    error: str | None = None


class ApprovalRequest(BaseModel):
    actor: str = Field(..., min_length=1, description="Approver identity or role")
    rationale: str = Field(..., min_length=1, description="Justification for approving model version")


class RejectionRequest(BaseModel):
    actor: str = Field(..., min_length=1, description="Reviewer identity or role")
    reason: str = Field(..., min_length=1, description="Justification for rejecting model version")


class DeploymentResponse(BaseModel):
    id: str
    environment_id: str
    candidate_id: str
    previous_candidate_id: str | None = None
    status: str
    smoke_test_results_json: dict[str, Any] | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    approval_rationale: str | None = None
    created_at: datetime
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
