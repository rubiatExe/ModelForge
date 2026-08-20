"""Failure taxonomy and deterministic representative examples."""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum

from pydantic import Field

from modelforge.evaluation.models import (
    EvaluationConfig,
    EvaluationItemResult,
    EvaluationReport,
    EvaluationStatus,
    FrozenModel,
)


class FailureCategory(StrEnum):
    PREDICTION_ERROR = "prediction_error"
    TIMEOUT = "timeout"
    SCHEMA_INVALID = "schema_invalid"
    ISSUE_TYPE = "issue_type_mismatch"
    ROUTING_TEAM = "routing_team_mismatch"
    SEVERITY = "severity_mismatch"
    SCOPE = "scope_mismatch"
    ACTION = "action_mismatch"
    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    EVIDENCE_UNSUPPORTED = "evidence_not_literal"
    MULTIPLE_FIELDS = "multiple_field_mismatch"
    OTHER = "other_structured_mismatch"


class FailureExample(FrozenModel):
    case_id: str
    slice: str
    category: FailureCategory
    detail: str
    expected: dict
    prediction: dict | None = None


class FailureBucket(FrozenModel):
    category: FailureCategory
    count: int = Field(ge=0)
    examples: tuple[FailureExample, ...]


class FailureTaxonomyReport(FrozenModel):
    evaluated_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    buckets: tuple[FailureBucket, ...]


def _categories(
    item: EvaluationItemResult,
    config: EvaluationConfig,
) -> list[tuple[FailureCategory, str]]:
    if item.status == EvaluationStatus.ERROR:
        kind = item.error.kind if item.error is not None else "unknown"
        category = (
            FailureCategory.TIMEOUT
            if "timeout" in kind.casefold()
            else FailureCategory.PREDICTION_ERROR
        )
        return [(category, kind)]
    if item.status == EvaluationStatus.SCHEMA_INVALID:
        return [(FailureCategory.SCHEMA_INVALID, item.schema_error or "schema invalid")]

    mapping = {
        config.issue_field: FailureCategory.ISSUE_TYPE,
        config.routing_field: FailureCategory.ROUTING_TEAM,
        config.severity_field: FailureCategory.SEVERITY,
        config.scope_field: FailureCategory.SCOPE,
        config.action_field: FailureCategory.ACTION,
    }
    mismatches = [field for field, matched in item.field_matches.items() if not matched]
    categories: list[tuple[FailureCategory, str]] = [
        (mapping[field], field) for field in mismatches if field in mapping
    ]
    if config.evidence_field in mismatches:
        categories.append((FailureCategory.EVIDENCE_MISMATCH, config.evidence_field))
    if item.evidence_present is False:
        categories.append((FailureCategory.EVIDENCE_MISSING, config.evidence_field))
    if item.unsupported_evidence_count:
        categories.append(
            (
                FailureCategory.EVIDENCE_UNSUPPORTED,
                f"{item.unsupported_evidence_count}/{item.evidence_count} snippets are not literal",
            )
        )
    if len(mismatches) > 1:
        categories.append((FailureCategory.MULTIPLE_FIELDS, ",".join(sorted(mismatches))))
    if item.exact_match is False and not categories:
        categories.append((FailureCategory.OTHER, "structured values differ"))
    return categories


def build_failure_taxonomy(
    report: EvaluationReport,
    *,
    config: EvaluationConfig | None = None,
    max_examples_per_category: int = 3,
) -> FailureTaxonomyReport:
    """Classify failures without including raw ticket text in the artifact."""

    if max_examples_per_category < 0:
        raise ValueError("max_examples_per_category must be non-negative")
    resolved_config = config or EvaluationConfig()
    buckets: defaultdict[FailureCategory, list[FailureExample]] = defaultdict(list)
    failed_case_ids: set[str] = set()
    for item in sorted(report.items, key=lambda value: value.case_id):
        categories = _categories(item, resolved_config)
        if categories:
            failed_case_ids.add(item.case_id)
        for category, detail in categories:
            buckets[category].append(
                FailureExample(
                    case_id=item.case_id,
                    slice=item.slice,
                    category=category,
                    detail=detail,
                    expected=item.expected,
                    prediction=item.prediction,
                )
            )

    return FailureTaxonomyReport(
        evaluated_cases=len(report.items),
        failed_cases=len(failed_case_ids),
        buckets=tuple(
            FailureBucket(
                category=category,
                count=len(buckets[category]),
                examples=tuple(buckets[category][:max_examples_per_category]),
            )
            for category in sorted(buckets, key=str)
        ),
    )
