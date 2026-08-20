"""Versioned dataset example, manifest, and test-lock schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from modelforge.schemas._base import StrictBaseModel
from modelforge.schemas._normalization import contains_normalized_literal
from modelforge.schemas.enums import (
    AffectedScope,
    AnnotationStatus,
    DatasetSource,
    DatasetSplit,
    IssueType,
    RecommendedAction,
    ReviewStatus,
    RobustnessCategory,
    RoutingTeam,
    Severity,
)
from modelforge.schemas.triage import TicketInput, TriageLabel

Sha256 = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Version = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]
StrictNonNegativeInt = Annotated[StrictInt, Field(ge=0)]
Identifier = Annotated[
    StrictStr,
    StringConstraints(min_length=3, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]


class ExampleProvenance(StrictBaseModel):
    source: DatasetSource
    generator: Annotated[StrictStr, StringConstraints(min_length=1, max_length=200)]
    generator_version: Version
    template_id: Identifier
    seed: StrictNonNegativeInt


class LabeledExample(StrictBaseModel):
    example_id: Identifier
    split: DatasetSplit
    ticket: TicketInput
    expected: TriageLabel
    provenance: ExampleProvenance
    annotation_status: AnnotationStatus
    review_status: ReviewStatus
    robustness_category: RobustnessCategory | None = None

    @model_validator(mode="after")
    def validate_example_semantics(self) -> LabeledExample:
        if self.split is DatasetSplit.ADVERSARIAL and self.robustness_category is None:
            raise ValueError("adversarial examples require a robustness_category")
        if self.split is not DatasetSplit.ADVERSARIAL and self.robustness_category is not None:
            raise ValueError("robustness_category is only valid for adversarial examples")

        ticket_text = f"{self.ticket.subject}\n{self.ticket.body}"
        unsupported = [
            snippet
            for snippet in self.expected.evidence
            if not contains_normalized_literal(ticket_text, snippet)
        ]
        if unsupported:
            raise ValueError(f"evidence is not a literal ticket substring: {unsupported!r}")
        return self


def _require_complete_distribution(
    mapping: dict[Any, int], enum_type: type[Any], field: str
) -> None:
    expected = set(enum_type)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(item.value for item in expected - actual)
        extra = sorted(str(item) for item in actual - expected)
        raise ValueError(
            f"{field} must contain every allowed label; missing={missing}, extra={extra}"
        )


class LabelDistribution(StrictBaseModel):
    issue_type: dict[IssueType, StrictNonNegativeInt]
    severity: dict[Severity, StrictNonNegativeInt]
    routing_team: dict[RoutingTeam, StrictNonNegativeInt]
    affected_scope: dict[AffectedScope, StrictNonNegativeInt]
    recommended_action: dict[RecommendedAction, StrictNonNegativeInt]

    @model_validator(mode="after")
    def require_all_labels(self) -> LabelDistribution:
        _require_complete_distribution(self.issue_type, IssueType, "issue_type")
        _require_complete_distribution(self.severity, Severity, "severity")
        _require_complete_distribution(self.routing_team, RoutingTeam, "routing_team")
        _require_complete_distribution(self.affected_scope, AffectedScope, "affected_scope")
        _require_complete_distribution(
            self.recommended_action,
            RecommendedAction,
            "recommended_action",
        )
        return self


def _validate_relative_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("artifact paths must be safe relative POSIX paths")
    return value


class SplitManifest(StrictBaseModel):
    split: DatasetSplit
    file: Annotated[StrictStr, StringConstraints(min_length=1, max_length=200)]
    count: StrictNonNegativeInt
    sha256: Sha256
    fingerprint_sha256: Sha256
    label_distribution: LabelDistribution
    robustness_distribution: dict[RobustnessCategory, StrictNonNegativeInt] = Field(
        default_factory=dict
    )

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        return _validate_relative_artifact_path(value)

    @model_validator(mode="after")
    def validate_counts(self) -> SplitManifest:
        for field_name in (
            "issue_type",
            "severity",
            "routing_team",
            "affected_scope",
            "recommended_action",
        ):
            if sum(getattr(self.label_distribution, field_name).values()) != self.count:
                raise ValueError(f"{field_name} distribution does not sum to split count")

        if self.split is DatasetSplit.ADVERSARIAL:
            _require_complete_distribution(
                self.robustness_distribution,
                RobustnessCategory,
                "robustness_distribution",
            )
            if sum(self.robustness_distribution.values()) != self.count:
                raise ValueError("robustness_distribution does not sum to split count")
        elif self.robustness_distribution:
            raise ValueError("only the adversarial split may have a robustness distribution")
        return self


class DatasetManifest(StrictBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Identifier
    dataset_version: Version
    generated_at: datetime
    generator: Annotated[StrictStr, StringConstraints(min_length=1, max_length=200)]
    generator_version: Version
    seed: StrictNonNegativeInt
    source: DatasetSource
    annotation_status: AnnotationStatus
    review_status: ReviewStatus
    total_count: StrictNonNegativeInt
    splits: Annotated[list[SplitManifest], Field(min_length=4, max_length=4)]
    test_lock_file: Annotated[StrictStr, StringConstraints(min_length=1, max_length=200)]

    @field_validator("generated_at")
    @classmethod
    def require_generated_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a UTC offset")
        return value

    @field_validator("test_lock_file")
    @classmethod
    def validate_lock_file(cls, value: str) -> str:
        return _validate_relative_artifact_path(value)

    @model_validator(mode="after")
    def validate_manifest_totals(self) -> DatasetManifest:
        splits = [item.split for item in self.splits]
        if len(splits) != len(set(splits)) or set(splits) != set(DatasetSplit):
            raise ValueError("manifest must contain each dataset split exactly once")
        if sum(item.count for item in self.splits) != self.total_count:
            raise ValueError("split counts do not sum to total_count")
        return self


class TestSetLock(StrictBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Identifier
    dataset_version: Version
    split: DatasetSplit
    file: Annotated[StrictStr, StringConstraints(min_length=1, max_length=200)]
    count: StrictNonNegativeInt
    sha256: Sha256
    fingerprint_sha256: Sha256

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        return _validate_relative_artifact_path(value)

    @model_validator(mode="after")
    def require_test_split(self) -> TestSetLock:
        if self.split is not DatasetSplit.TEST:
            raise ValueError("a test-set lock may only describe the test split")
        return self
