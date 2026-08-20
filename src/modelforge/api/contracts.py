"""API-only contracts; domain prediction schemas live in modelforge.schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorDetail


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "degraded"]
    version: str
    small_model_ready: bool
    frontier_configured: bool
    routing_policy_status: Literal["uncalibrated", "calibrated"]


class ModelDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    role: Literal["small", "frontier", "classical", "demo"]
    ready: bool
    revision: str | None = None
    adapter: str | None = None
    note: str = ""


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_./:-]+$")
    split: Literal["validation", "test", "adversarial"] = "validation"
    limit: int | None = Field(default=None, ge=1, le=200)


class EvaluationRunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: UUID
    status: Literal["queued", "running", "complete", "failed"]
    model_id: str
    split: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    report: dict[str, Any] | None = None
    error_code: str | None = None
