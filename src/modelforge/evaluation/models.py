"""Typed, provider-neutral records used by the evaluation subsystem.

The evaluator deliberately consumes cached values rather than model clients.  A
provider outage is represented by :class:`PredictionError`; it is never
silently converted into a wrong model answer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

EVALUATOR_VERSION = "deterministic-iam-v1"


class FrozenModel(BaseModel):
    """Strict immutable base for artifacts that must remain reproducible."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PredictionError(FrozenModel):
    """A generation/runtime failure, kept separate from model quality."""

    kind: str = Field(min_length=1)
    message: str = ""
    retriable: bool = False


class EvaluationRecord(FrozenModel):
    """One expected value paired with either a prediction or an error.

    ``raw_prediction`` is intentionally retained as a distinct input.  It lets
    schema evaluation reject markdown fences, surrounding prose, duplicate JSON
    documents, and malformed JSON before any structured comparison occurs.
    """

    case_id: str = Field(min_length=1)
    expected: dict[str, Any]
    prediction: dict[str, Any] | None = None
    raw_prediction: str | None = None
    error: PredictionError | None = None
    slice: str = Field(default="default", min_length=1)
    input_text: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_outcome(self) -> EvaluationRecord:
        values = int(self.prediction is not None) + int(self.raw_prediction is not None)
        if self.error is not None:
            if values:
                raise ValueError("an errored record cannot also contain a prediction")
        elif values != 1:
            raise ValueError("a non-error record must contain exactly one prediction form")
        return self


class EvaluationStatus(StrEnum):
    ERROR = "error"
    SCHEMA_INVALID = "schema_invalid"
    SCORED = "scored"


class ClassMetrics(FrozenModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: float = Field(ge=0.0)
    predicted: float = Field(ge=0.0)


class ClassificationMetrics(FrozenModel):
    labels: tuple[str, ...]
    count: float = Field(ge=0.0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    macro_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    macro_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, float]]


class EvaluationItemResult(FrozenModel):
    case_id: str
    slice: str
    status: EvaluationStatus
    expected: dict[str, Any]
    prediction: dict[str, Any] | None = None
    error: PredictionError | None = None
    schema_valid: bool | None = None
    schema_error: str | None = None
    exact_match: bool | None = None
    field_matches: dict[str, bool] = Field(default_factory=dict)
    evidence_present: bool | None = None
    evidence_count: int = Field(default=0, ge=0)
    grounded_evidence_count: int = Field(default=0, ge=0)
    unsupported_evidence_count: int = Field(default=0, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    cost_usd: float | None = Field(default=None, ge=0.0)


class EvaluationSliceMetrics(FrozenModel):
    """Aggregate metrics with explicit, documented denominators.

    ``error_rate`` uses all cases.  Schema validity and exact match use only
    returned predictions, so a provider outage cannot masquerade as poor model
    quality.  ``wrong_predictions`` includes invalid-schema returns but excludes
    runtime errors.
    """

    total_cases: int = Field(ge=0)
    returned_predictions: int = Field(ge=0)
    error_count: int = Field(ge=0)
    wrong_predictions: int = Field(ge=0)
    schema_valid_count: int = Field(ge=0)
    schema_invalid_count: int = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    schema_validity_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    exact_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    issue: ClassificationMetrics
    issue_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    routing_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    severity_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    scope_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    action_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    field_accuracies: dict[str, float | None]
    evidence_presence_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    literal_evidence_grounding_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    unsupported_evidence_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class EvaluationReport(FrozenModel):
    evaluator_version: str = EVALUATOR_VERSION
    schema_name: str
    aggregate: EvaluationSliceMetrics
    slices: dict[str, EvaluationSliceMetrics]
    items: tuple[EvaluationItemResult, ...]


class EvaluationConfig(FrozenModel):
    """Names of the domain fields measured by deterministic evaluation."""

    issue_field: str = "issue_type"
    routing_field: str = "routing_team"
    severity_field: str = "severity"
    scope_field: str = "affected_scope"
    action_field: str = "recommended_action"
    evidence_field: str = "evidence"
    issue_labels: tuple[str, ...] = ()
    allowed_prediction_only_fields: tuple[str, ...] = ("confidence",)

    @property
    def categorical_fields(self) -> tuple[str, ...]:
        return (
            self.issue_field,
            self.routing_field,
            self.severity_field,
            self.scope_field,
            self.action_field,
        )
