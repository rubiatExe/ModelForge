"""Strict request and result schemas for IAM support-ticket triage."""

from __future__ import annotations

import math
import unicodedata
from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, StrictStr, StringConstraints, field_validator

from modelforge.schemas._base import StrictBaseModel
from modelforge.schemas._normalization import normalize_unicode_whitespace
from modelforge.schemas.enums import (
    AffectedScope,
    IssueType,
    RecommendedAction,
    RoutingTeam,
    Severity,
)

TicketId = Annotated[
    StrictStr,
    StringConstraints(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"),
]
Subject = Annotated[StrictStr, StringConstraints(min_length=3, max_length=200)]
TicketBody = Annotated[StrictStr, StringConstraints(min_length=10, max_length=4_000)]
Department = Annotated[StrictStr, StringConstraints(min_length=1, max_length=100)]
EvidenceText = Annotated[StrictStr, StringConstraints(min_length=1, max_length=300)]


def _reject_unsafe_controls(value: str) -> str:
    for character in value:
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t":
            raise ValueError("text contains a disallowed control character")
    return value


class TicketInput(StrictBaseModel):
    """The only ticket fields models may use to make a triage decision."""

    ticket_id: TicketId
    subject: Subject
    body: TicketBody
    employee_department: Department
    submitted_at: datetime

    @field_validator("subject", "body", "employee_department")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        return _reject_unsafe_controls(value)

    @field_validator("submitted_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("submitted_at must include a UTC offset")
        return value


class TriageLabel(StrictBaseModel):
    """Gold decision fields; confidence is intentionally not a gold label."""

    issue_type: IssueType
    severity: Severity
    routing_team: RoutingTeam
    affected_scope: AffectedScope
    recommended_action: RecommendedAction
    evidence: Annotated[list[EvidenceText], Field(min_length=1, max_length=5)]

    @field_validator("evidence")
    @classmethod
    def require_unique_evidence(cls, evidence: list[str]) -> list[str]:
        normalized = [normalize_unicode_whitespace(item, casefold=True) for item in evidence]
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence snippets must be unique")
        return evidence


class TriageResult(TriageLabel):
    """A model prediction, including its separately evaluated self-confidence."""

    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_coerced_confidence(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a JSON number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("confidence must be finite")
        return converted
