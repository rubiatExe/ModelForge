"""Population-aware threshold sweeps over cached paired predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from modelforge.evaluation.metrics import (
    _jsonable,
    classification_metrics,
    evaluate_records,
    validate_schema,
)
from modelforge.evaluation.models import (
    ClassificationMetrics,
    EvaluationConfig,
    EvaluationRecord,
    EvaluationStatus,
    FrozenModel,
    PredictionError,
)

DEFAULT_THRESHOLDS = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


class CachedPrediction(FrozenModel):
    """One immutable branch outcome captured before a threshold sweep."""

    prediction: dict[str, Any] | None = None
    raw_prediction: str | None = None
    error: PredictionError | None = None
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _one_outcome(self) -> CachedPrediction:
        count = int(self.prediction is not None) + int(self.raw_prediction is not None)
        if self.error is not None:
            if count:
                raise ValueError("an errored cached prediction cannot also have output")
        elif count != 1:
            raise ValueError("a cached prediction requires exactly one output form")
        return self

    @classmethod
    def from_model_prediction(cls, prediction: Any) -> CachedPrediction:
        result = getattr(prediction, "result", getattr(prediction, "output", None))
        if result is None:
            raise TypeError("model prediction does not expose a result")
        return cls(
            prediction=_jsonable(result),
            latency_ms=float(prediction.latency_ms),
            cost_usd=float(getattr(prediction, "estimated_cost_usd", 0.0)),
        )


class CachedRoutingCase(FrozenModel):
    case_id: str = Field(min_length=1)
    split: str = "validation"
    slice: str = "default"
    expected: dict[str, Any]
    input_text: str = ""
    student: CachedPrediction
    frontier: CachedPrediction
    student_score: float | None = Field(default=None, ge=0.0, le=1.0)
    population_weight: float = Field(default=1.0, gt=0.0)

    @classmethod
    def from_domain(
        cls,
        *,
        example: Any,
        student: CachedPrediction | Any,
        frontier: CachedPrediction | Any,
        student_score: float | None,
        population_weight: float = 1.0,
        slice_name: str | None = None,
    ) -> CachedRoutingCase:
        """Adapt a labeled example and two cached ``ModelPrediction`` values."""

        split = getattr(example.split, "value", str(example.split))
        ticket = example.ticket
        return cls(
            case_id=str(example.example_id),
            split=str(split),
            slice=slice_name or str(split),
            expected=_jsonable(example.expected),
            input_text="\n".join(part for part in (ticket.subject, ticket.body) if part),
            student=(
                student
                if isinstance(student, CachedPrediction)
                else CachedPrediction.from_model_prediction(student)
            ),
            frontier=(
                frontier
                if isinstance(frontier, CachedPrediction)
                else CachedPrediction.from_model_prediction(frontier)
            ),
            student_score=student_score,
            population_weight=population_weight,
        )


class RoutingMetrics(FrozenModel):
    case_count: int = Field(ge=0)
    population_weight: float = Field(ge=0.0)
    selected_error_weight: float = Field(ge=0.0)
    selected_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    schema_validity_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    exact_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    issue: ClassificationMetrics
    issue_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    routing_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    severity_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    scope_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    action_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    field_accuracies: dict[str, float | None]
    frontier_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    frontier_calls_per_100: float | None = Field(default=None, ge=0.0, le=100.0)
    average_latency_ms: float | None = Field(default=None, ge=0.0)
    p95_latency_ms: float | None = Field(default=None, ge=0.0)
    average_cost_usd: float | None = Field(default=None, ge=0.0)
    overconfident_error_weight: float = Field(ge=0.0)
    overconfident_errors_per_100: float | None = Field(default=None, ge=0.0, le=100.0)


class RoutingThresholdResult(FrozenModel):
    threshold: float = Field(ge=0.0, le=1.0)
    aggregate: RoutingMetrics
    slices: dict[str, RoutingMetrics]


class RoutingSweepReport(FrozenModel):
    split: str
    route_rule: str = "frontier when student score < threshold or student is invalid/error"
    thresholds: tuple[float, ...]
    results: tuple[RoutingThresholdResult, ...]


class _RoutedItem:
    __slots__ = ("case", "escalated", "record")

    def __init__(self, case: CachedRoutingCase, escalated: bool, record: EvaluationRecord) -> None:
        self.case = case
        self.escalated = escalated
        self.record = record


def _branch_record(
    case: CachedRoutingCase,
    branch: CachedPrediction,
    *,
    latency_ms: float,
    cost_usd: float,
) -> EvaluationRecord:
    return EvaluationRecord(
        case_id=case.case_id,
        expected=case.expected,
        prediction=branch.prediction,
        raw_prediction=branch.raw_prediction,
        error=branch.error,
        slice=case.slice,
        input_text=case.input_text,
        confidence=case.student_score,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def _student_is_usable(
    case: CachedRoutingCase,
    *,
    schema_model: type[BaseModel] | None,
    config: EvaluationConfig,
) -> bool:
    if case.student.error is not None:
        return False
    record = _branch_record(
        case,
        case.student,
        latency_ms=case.student.latency_ms,
        cost_usd=case.student.cost_usd,
    )
    _, problem = validate_schema(record, schema_model=schema_model, config=config)
    return problem is None


def _route_cases(
    cases: Sequence[CachedRoutingCase],
    threshold: float,
    *,
    schema_model: type[BaseModel] | None,
    config: EvaluationConfig,
) -> list[_RoutedItem]:
    routed: list[_RoutedItem] = []
    for case in cases:
        usable = _student_is_usable(case, schema_model=schema_model, config=config)
        escalated = not usable or case.student_score is None or case.student_score < threshold
        if escalated:
            branch = case.frontier
            latency = case.student.latency_ms + branch.latency_ms
            cost = case.student.cost_usd + branch.cost_usd
        else:
            branch = case.student
            latency = branch.latency_ms
            cost = branch.cost_usd
        routed.append(
            _RoutedItem(
                case,
                escalated,
                _branch_record(case, branch, latency_ms=latency, cost_usd=cost),
            )
        )
    return routed


def weighted_percentile(values: Iterable[tuple[float, float]], q: float) -> float | None:
    """Nearest-rank percentile for positive population weights."""

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    supplied = [(float(value), float(weight)) for value, weight in values]
    if any(not math.isfinite(value) or not math.isfinite(weight) for value, weight in supplied):
        raise ValueError("weighted percentile inputs must be finite")
    if any(weight < 0 for _, weight in supplied):
        raise ValueError("weighted percentile weights must be non-negative")
    rows = sorted((value, weight) for value, weight in supplied if weight > 0)
    if not rows:
        return None
    total = sum(weight for _, weight in rows)
    target = q * total
    cumulative = 0.0
    for value, weight in rows:
        cumulative += weight
        if cumulative >= target:
            return value
    return rows[-1][0]


def _summarize_routed(
    routed: Sequence[_RoutedItem],
    *,
    schema_model: type[BaseModel] | None,
    config: EvaluationConfig,
) -> RoutingMetrics:
    report = evaluate_records(
        [item.record for item in routed],
        schema_model=schema_model,
        config=config,
    )
    by_id = {item.case_id: item for item in report.items}
    population = sum(item.case.population_weight for item in routed)
    error_weight = valid_weight = returned_weight = exact_weight = frontier_weight = 0.0
    overconfident = 0.0
    issue_observations: list[tuple[str, str, float]] = []
    field_totals: defaultdict[str, float] = defaultdict(float)
    field_hits: defaultdict[str, float] = defaultdict(float)
    for routed_item in routed:
        case = routed_item.case
        weight = case.population_weight
        result = by_id[case.case_id]
        if routed_item.escalated:
            frontier_weight += weight
        if result.status == EvaluationStatus.ERROR:
            error_weight += weight
            continue
        returned_weight += weight
        if result.status == EvaluationStatus.SCORED:
            valid_weight += weight
            if result.exact_match:
                exact_weight += weight
            assert result.prediction is not None
            expected_issue = result.expected.get(config.issue_field)
            predicted_issue = result.prediction.get(config.issue_field)
            if expected_issue is not None and predicted_issue is not None:
                issue_observations.append((str(expected_issue), str(predicted_issue), weight))
            for field in config.categorical_fields:
                if field in result.field_matches:
                    field_totals[field] += weight
                    if result.field_matches[field]:
                        field_hits[field] += weight
        if not routed_item.escalated and result.exact_match is not True:
            overconfident += weight

    field_accuracies = {
        field: field_hits[field] / field_totals[field] if field_totals[field] else None
        for field in config.categorical_fields
    }
    latency_values = [
        (float(item.record.latency_ms), item.case.population_weight)
        for item in routed
        if item.record.latency_ms is not None
    ]
    average_latency = (
        sum(value * weight for value, weight in latency_values)
        / sum(weight for _, weight in latency_values)
        if latency_values
        else None
    )
    cost_values = [
        (float(item.record.cost_usd), item.case.population_weight)
        for item in routed
        if item.record.cost_usd is not None
    ]
    average_cost = (
        sum(value * weight for value, weight in cost_values)
        / sum(weight for _, weight in cost_values)
        if cost_values
        else None
    )
    issue_metrics = classification_metrics(issue_observations, labels=config.issue_labels)
    return RoutingMetrics(
        case_count=len(routed),
        population_weight=population,
        selected_error_weight=error_weight,
        selected_error_rate=error_weight / population if population else None,
        schema_validity_rate=valid_weight / returned_weight if returned_weight else None,
        exact_match_rate=exact_weight / returned_weight if returned_weight else None,
        issue=issue_metrics,
        issue_accuracy=issue_metrics.accuracy,
        routing_accuracy=field_accuracies[config.routing_field],
        severity_accuracy=field_accuracies[config.severity_field],
        scope_accuracy=field_accuracies[config.scope_field],
        action_accuracy=field_accuracies[config.action_field],
        field_accuracies=field_accuracies,
        frontier_rate=frontier_weight / population if population else None,
        frontier_calls_per_100=100.0 * frontier_weight / population if population else None,
        average_latency_ms=average_latency,
        p95_latency_ms=weighted_percentile(latency_values, 0.95),
        average_cost_usd=average_cost,
        overconfident_error_weight=overconfident,
        overconfident_errors_per_100=100.0 * overconfident / population if population else None,
    )


def evaluate_routing_threshold(
    cases: Sequence[CachedRoutingCase],
    threshold: float,
    *,
    schema_model: type[BaseModel] | None = None,
    config: EvaluationConfig | None = None,
) -> RoutingThresholdResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    resolved_config = config or EvaluationConfig()
    routed = _route_cases(
        cases,
        threshold,
        schema_model=schema_model,
        config=resolved_config,
    )
    grouped: defaultdict[str, list[_RoutedItem]] = defaultdict(list)
    for item in routed:
        grouped[item.case.slice].append(item)
    return RoutingThresholdResult(
        threshold=threshold,
        aggregate=_summarize_routed(routed, schema_model=schema_model, config=resolved_config),
        slices={
            name: _summarize_routed(
                grouped[name], schema_model=schema_model, config=resolved_config
            )
            for name in sorted(grouped)
        },
    )


def sweep_routing_thresholds(
    cases: Sequence[CachedRoutingCase],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    schema_model: type[BaseModel] | None = None,
    config: EvaluationConfig | None = None,
    calibration_split: str = "validation",
) -> RoutingSweepReport:
    """Sweep fixed thresholds on one calibration split using cached outputs."""

    if not cases:
        raise ValueError("routing sweep requires at least one cached case")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("routing sweep case ids must be unique")
    splits = {
        case.split.value if isinstance(case.split, Enum) else str(case.split) for case in cases
    }
    if splits != {calibration_split}:
        raise ValueError(
            f"threshold calibration requires only split {calibration_split!r}; found {sorted(splits)}"
        )
    ordered = tuple(sorted(float(value) for value in thresholds))
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("thresholds must be non-empty and unique")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in ordered):
        raise ValueError("thresholds must be finite values in [0, 1]")
    resolved_config = config or EvaluationConfig()
    return RoutingSweepReport(
        split=calibration_split,
        thresholds=ordered,
        results=tuple(
            evaluate_routing_threshold(
                cases,
                threshold,
                schema_model=schema_model,
                config=resolved_config,
            )
            for threshold in ordered
        ),
    )
