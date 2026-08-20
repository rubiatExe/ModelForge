"""Deterministic IAM triage metrics.

Closed labels and literal evidence do not need an LLM judge.  This module keeps
the parse/schema boundary, structured comparison, and aggregate denominators
explicit so callers cannot confuse provider errors with wrong predictions.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from modelforge.evaluation.models import (
    ClassificationMetrics,
    ClassMetrics,
    EvaluationConfig,
    EvaluationItemResult,
    EvaluationRecord,
    EvaluationReport,
    EvaluationSliceMetrics,
    EvaluationStatus,
    PredictionError,
)


class _DuplicateKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_raw_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateKey) as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "prediction JSON must be an object"
    return parsed, None


def _same_json_shape(actual: Any, expected: Any, path: str = "$") -> str | None:
    """Return the first strict JSON type mismatch, if one exists."""

    if expected is None:
        return None if actual is None else f"{path}: expected null"
    if isinstance(expected, bool):
        return None if isinstance(actual, bool) else f"{path}: expected boolean"
    if isinstance(expected, str):
        return None if isinstance(actual, str) else f"{path}: expected string"
    if isinstance(expected, int):
        return (
            None
            if isinstance(actual, int) and not isinstance(actual, bool)
            else f"{path}: expected integer"
        )
    if isinstance(expected, float):
        if (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
        ):
            return None
        return f"{path}: expected finite number"
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}: expected array"
        if expected:
            exemplar = expected[0]
            for index, item in enumerate(actual):
                problem = _same_json_shape(item, exemplar, f"{path}[{index}]")
                if problem:
                    return problem
        return None
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected object"
        if set(actual) != set(expected):
            return f"{path}: object keys differ"
        for key, expected_value in expected.items():
            problem = _same_json_shape(actual[key], expected_value, f"{path}.{key}")
            if problem:
                return problem
        return None
    return None if type(actual) is type(expected) else f"{path}: incompatible value type"


def validate_schema(
    record: EvaluationRecord,
    *,
    schema_model: type[BaseModel] | None,
    config: EvaluationConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse and strictly validate one returned prediction.

    When a Pydantic result schema is supplied it is authoritative.  Without
    one, the gold record supplies the required key/type shape and only declared
    result-only metadata (``confidence`` by default) may be additional.
    """

    if record.error is not None:
        raise ValueError("errored records do not have a schema outcome")

    if record.raw_prediction is not None:
        prediction, problem = _parse_raw_json(record.raw_prediction)
        if problem:
            return prediction, problem
    else:
        prediction = _jsonable(record.prediction)
        problem = None
    if prediction is None or not isinstance(prediction, dict):
        return None, problem or "prediction must be an object"

    expected = _jsonable(record.expected)
    if schema_model is not None:
        try:
            # JSON-mode strictness accepts JSON enum strings but still rejects
            # coercions such as strings to numbers and respects ``extra=forbid``.
            serialized = json.dumps(prediction, ensure_ascii=False, allow_nan=False)
            validated = schema_model.model_validate_json(serialized, strict=True)
        except (TypeError, ValueError, ValidationError) as exc:
            return prediction, str(exc)
        normalized = validated.model_dump(mode="json")
        missing_gold = set(expected) - set(normalized)
        if missing_gold:
            return normalized, f"prediction missing gold fields: {sorted(missing_gold)}"
        return normalized, None

    required = set(expected)
    allowed = required | set(config.allowed_prediction_only_fields)
    if missing := required - set(prediction):
        return prediction, f"prediction missing fields: {sorted(missing)}"
    if extra := set(prediction) - allowed:
        return prediction, f"prediction has unexpected fields: {sorted(extra)}"
    for key, expected_value in expected.items():
        if problem := _same_json_shape(prediction[key], expected_value, f"$.{key}"):
            return prediction, problem
    confidence = prediction.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return prediction, "$.confidence: expected finite number in [0, 1]"
    return prediction, None


def normalize_literal(text: str) -> str:
    """Canonicalize text for conservative literal-evidence containment."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def evidence_items(value: Any) -> tuple[str, ...]:
    """Return non-empty evidence strings; invalid types yield no evidence."""

    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return ()
    return tuple(item for item in values if isinstance(item, str) and item.strip())


def literal_evidence_counts(value: Any, input_text: str) -> tuple[int, int, int]:
    """Return ``(total, grounded, unsupported)`` evidence snippet counts."""

    items = evidence_items(value)
    source = normalize_literal(input_text)
    grounded = sum(
        bool(normalize_literal(item)) and normalize_literal(item) in source for item in items
    )
    return len(items), grounded, len(items) - grounded


def classification_metrics(
    observations: Iterable[tuple[str, str, float]],
    *,
    labels: Iterable[str] = (),
) -> ClassificationMetrics:
    """Compute deterministic weighted multiclass metrics and confusion matrix."""

    rows = [
        (str(expected), str(predicted), float(weight))
        for expected, predicted, weight in observations
    ]
    if any(not math.isfinite(weight) or weight < 0 for _, _, weight in rows):
        raise ValueError("classification weights must be finite and non-negative")

    configured = list(dict.fromkeys(str(label) for label in labels))
    observed = sorted({label for expected, predicted, _ in rows for label in (expected, predicted)})
    ordered_labels = tuple(configured + [label for label in observed if label not in configured])
    matrix = {
        expected: {predicted: 0.0 for predicted in ordered_labels} for expected in ordered_labels
    }
    for expected, predicted, weight in rows:
        matrix[expected][predicted] += weight

    total = sum(weight for _, _, weight in rows)
    correct = sum(weight for expected, predicted, weight in rows if expected == predicted)
    per_class: dict[str, ClassMetrics] = {}
    for label in ordered_labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in ordered_labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            predicted=predicted_count,
        )

    if ordered_labels:
        macro_precision = sum(item.precision for item in per_class.values()) / len(ordered_labels)
        macro_recall = sum(item.recall for item in per_class.values()) / len(ordered_labels)
        macro_f1 = sum(item.f1 for item in per_class.values()) / len(ordered_labels)
    else:
        macro_precision = macro_recall = macro_f1 = None
    return ClassificationMetrics(
        labels=ordered_labels,
        count=total,
        accuracy=correct / total if total else None,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=matrix,
    )


def _score_record(
    record: EvaluationRecord,
    *,
    schema_model: type[BaseModel] | None,
    config: EvaluationConfig,
) -> EvaluationItemResult:
    expected = _jsonable(record.expected)
    if record.error is not None:
        return EvaluationItemResult(
            case_id=record.case_id,
            slice=record.slice,
            status=EvaluationStatus.ERROR,
            expected=expected,
            error=record.error,
            confidence=record.confidence,
            latency_ms=record.latency_ms,
            cost_usd=record.cost_usd,
        )

    prediction, schema_error = validate_schema(record, schema_model=schema_model, config=config)
    if schema_error is not None:
        return EvaluationItemResult(
            case_id=record.case_id,
            slice=record.slice,
            status=EvaluationStatus.SCHEMA_INVALID,
            expected=expected,
            prediction=prediction,
            schema_valid=False,
            schema_error=schema_error,
            exact_match=False,
            confidence=record.confidence,
            latency_ms=record.latency_ms,
            cost_usd=record.cost_usd,
        )

    assert prediction is not None
    field_matches = {field: prediction.get(field) == value for field, value in expected.items()}
    exact_match = all(field_matches.values())
    evidence = prediction.get(config.evidence_field)
    total, grounded, unsupported = literal_evidence_counts(evidence, record.input_text)
    return EvaluationItemResult(
        case_id=record.case_id,
        slice=record.slice,
        status=EvaluationStatus.SCORED,
        expected=expected,
        prediction=prediction,
        schema_valid=True,
        exact_match=exact_match,
        field_matches=field_matches,
        evidence_present=bool(total),
        evidence_count=total,
        grounded_evidence_count=grounded,
        unsupported_evidence_count=unsupported,
        confidence=record.confidence,
        latency_ms=record.latency_ms,
        cost_usd=record.cost_usd,
    )


def summarize_items(
    items: Sequence[EvaluationItemResult],
    *,
    config: EvaluationConfig,
) -> EvaluationSliceMetrics:
    total = len(items)
    errors = [item for item in items if item.status == EvaluationStatus.ERROR]
    predictions = [item for item in items if item.status != EvaluationStatus.ERROR]
    valid = [item for item in items if item.status == EvaluationStatus.SCORED]
    invalid = [item for item in items if item.status == EvaluationStatus.SCHEMA_INVALID]
    exact = sum(item.exact_match is True for item in valid)

    issue_observations: list[tuple[str, str, float]] = []
    for item in valid:
        assert item.prediction is not None
        expected = item.expected.get(config.issue_field)
        predicted = item.prediction.get(config.issue_field)
        if expected is not None and predicted is not None:
            issue_observations.append((str(expected), str(predicted), 1.0))

    field_accuracies: dict[str, float | None] = {}
    for field in config.categorical_fields:
        eligible = [
            item for item in valid if field in item.expected and field in item.field_matches
        ]
        field_accuracies[field] = (
            sum(item.field_matches[field] for item in eligible) / len(eligible)
            if eligible
            else None
        )

    evidence_eligible = [item for item in valid if config.evidence_field in item.expected]
    evidence_total = sum(item.evidence_count for item in evidence_eligible)
    grounded_total = sum(item.grounded_evidence_count for item in evidence_eligible)
    unsupported_total = sum(item.unsupported_evidence_count for item in evidence_eligible)
    issue_metrics = classification_metrics(issue_observations, labels=config.issue_labels)
    return EvaluationSliceMetrics(
        total_cases=total,
        returned_predictions=len(predictions),
        error_count=len(errors),
        wrong_predictions=len(predictions) - exact,
        schema_valid_count=len(valid),
        schema_invalid_count=len(invalid),
        exact_match_count=exact,
        error_rate=len(errors) / total if total else None,
        schema_validity_rate=len(valid) / len(predictions) if predictions else None,
        exact_match_rate=exact / len(predictions) if predictions else None,
        issue=issue_metrics,
        issue_accuracy=issue_metrics.accuracy,
        routing_accuracy=field_accuracies[config.routing_field],
        severity_accuracy=field_accuracies[config.severity_field],
        scope_accuracy=field_accuracies[config.scope_field],
        action_accuracy=field_accuracies[config.action_field],
        field_accuracies=field_accuracies,
        evidence_presence_rate=(
            sum(item.evidence_present is True for item in evidence_eligible)
            / len(evidence_eligible)
            if evidence_eligible
            else None
        ),
        literal_evidence_grounding_rate=grounded_total / evidence_total if evidence_total else None,
        unsupported_evidence_rate=unsupported_total / evidence_total if evidence_total else None,
    )


def evaluate_records(
    records: Iterable[EvaluationRecord],
    *,
    schema_model: type[BaseModel] | None = None,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Evaluate cached records once, returning aggregate and per-slice metrics."""

    resolved_config = config or EvaluationConfig()
    rows = list(records)
    case_ids = [record.case_id for record in rows]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted(case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1)
        raise ValueError(f"duplicate evaluation case ids: {duplicates}")

    items = tuple(
        _score_record(record, schema_model=schema_model, config=resolved_config) for record in rows
    )
    grouped: defaultdict[str, list[EvaluationItemResult]] = defaultdict(list)
    for item in items:
        grouped[item.slice].append(item)
    slices = {
        name: summarize_items(grouped[name], config=resolved_config) for name in sorted(grouped)
    }
    return EvaluationReport(
        schema_name=schema_model.__name__ if schema_model is not None else "inferred-from-gold",
        aggregate=summarize_items(items, config=resolved_config),
        slices=slices,
        items=items,
    )


def evaluation_record_from_domain(
    *,
    example: Any,
    prediction: Any | None = None,
    error: PredictionError | None = None,
    slice_name: str | None = None,
) -> EvaluationRecord:
    """Adapt ``LabeledExample`` plus ``TriageResult``/``ModelPrediction``.

    The function uses the documented domain attributes without importing model
    adapters.  This keeps deterministic evaluation usable in the lightweight
    core environment and avoids a circular dependency on optional providers.
    """

    expected = _jsonable(example.expected)
    ticket = example.ticket
    input_text = "\n".join(part for part in (ticket.subject, ticket.body) if part)
    robustness = getattr(getattr(example, "robustness_category", None), "value", None)
    case_slice = slice_name or robustness or getattr(example, "split", "default")
    if isinstance(case_slice, Enum):
        case_slice = str(case_slice.value)

    if error is not None:
        return EvaluationRecord(
            case_id=str(example.example_id),
            expected=expected,
            error=error,
            slice=str(case_slice),
            input_text=input_text,
            metadata={"robustness_category": robustness},
        )

    output = getattr(prediction, "result", getattr(prediction, "output", prediction))
    # Only the result's explicit self-confidence belongs here. A composite
    # routing score and label logprob remain separately visible signals.
    confidence = getattr(output, "confidence", None)
    return EvaluationRecord(
        case_id=str(example.example_id),
        expected=expected,
        prediction=_jsonable(output),
        slice=str(case_slice),
        input_text=input_text,
        confidence=confidence,
        latency_ms=getattr(prediction, "latency_ms", None),
        cost_usd=getattr(prediction, "estimated_cost_usd", None),
        metadata={"robustness_category": robustness},
    )
