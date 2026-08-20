"""Dataset quality checks that fail closed before training or evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from modelforge.datasets.fingerprint import (
    canonical_fingerprint,
    canonical_label_fingerprint,
    fingerprint_set_digest,
)
from modelforge.schemas import (
    AffectedScope,
    DatasetSplit,
    IssueType,
    LabelDistribution,
    LabeledExample,
    RecommendedAction,
    RobustnessCategory,
    RoutingTeam,
    Severity,
)


class DatasetValidationError(ValueError):
    """Base class for deterministic dataset-integrity failures."""


class DuplicateExampleError(DatasetValidationError):
    pass


class ConflictingLabelError(DatasetValidationError):
    pass


class CrossSplitLeakageError(DatasetValidationError):
    pass


def validate_no_conflicting_labels(examples: Sequence[LabeledExample]) -> None:
    seen: dict[str, tuple[str, str]] = {}
    for example in examples:
        input_fingerprint = canonical_fingerprint(example.ticket)
        label_fingerprint = canonical_label_fingerprint(example.expected)
        previous = seen.get(input_fingerprint)
        if previous is not None and previous[1] != label_fingerprint:
            raise ConflictingLabelError(
                "conflicting labels for canonical input "
                f"{input_fingerprint}: {previous[0]} and {example.example_id}"
            )
        seen[input_fingerprint] = (example.example_id, label_fingerprint)


def validate_no_duplicates(examples: Sequence[LabeledExample]) -> None:
    seen_ids: set[str] = set()
    seen_inputs: dict[str, str] = {}
    for example in examples:
        if example.example_id in seen_ids:
            raise DuplicateExampleError(f"duplicate example_id: {example.example_id}")
        seen_ids.add(example.example_id)

        fingerprint = canonical_fingerprint(example.ticket)
        if fingerprint in seen_inputs:
            raise DuplicateExampleError(
                "duplicate canonical input "
                f"{fingerprint}: {seen_inputs[fingerprint]} and {example.example_id}"
            )
        seen_inputs[fingerprint] = example.example_id


def validate_no_cross_split_leakage(
    splits: Mapping[DatasetSplit, Sequence[LabeledExample]],
) -> None:
    seen: dict[str, tuple[DatasetSplit, str]] = {}
    seen_ids: dict[str, DatasetSplit] = {}
    for split, examples in splits.items():
        for example in examples:
            if example.split is not split:
                raise DatasetValidationError(
                    f"{example.example_id} declares {example.split.value}, loaded as {split.value}"
                )
            prior_id_split = seen_ids.get(example.example_id)
            if prior_id_split is not None and prior_id_split is not split:
                raise CrossSplitLeakageError(
                    f"example_id {example.example_id} appears in {prior_id_split.value} and {split.value}"
                )
            seen_ids[example.example_id] = split

            fingerprint = canonical_fingerprint(example.ticket)
            previous = seen.get(fingerprint)
            if previous is not None and previous[0] is not split:
                raise CrossSplitLeakageError(
                    "canonical input leaks across splits "
                    f"{previous[0].value}/{previous[1]} and {split.value}/{example.example_id}"
                )
            seen[fingerprint] = (split, example.example_id)


def validate_dataset_splits(
    splits: Mapping[DatasetSplit, Sequence[LabeledExample]],
) -> None:
    if set(splits) != set(DatasetSplit):
        missing = sorted(split.value for split in set(DatasetSplit) - set(splits))
        raise DatasetValidationError(f"dataset must contain every split; missing={missing}")

    all_examples: list[LabeledExample] = []
    for split, examples in splits.items():
        if not examples:
            raise DatasetValidationError(f"{split.value} split is empty")
        validate_no_conflicting_labels(examples)
        validate_no_duplicates(examples)
        all_examples.extend(examples)

    validate_no_conflicting_labels(all_examples)
    validate_no_cross_split_leakage(splits)


def _complete_counts(enum_type: type, values: Sequence) -> dict:
    counts = Counter(values)
    return {member: counts[member] for member in enum_type}


def label_distribution(examples: Sequence[LabeledExample]) -> LabelDistribution:
    return LabelDistribution(
        issue_type=_complete_counts(IssueType, [item.expected.issue_type for item in examples]),
        severity=_complete_counts(Severity, [item.expected.severity for item in examples]),
        routing_team=_complete_counts(
            RoutingTeam,
            [item.expected.routing_team for item in examples],
        ),
        affected_scope=_complete_counts(
            AffectedScope,
            [item.expected.affected_scope for item in examples],
        ),
        recommended_action=_complete_counts(
            RecommendedAction,
            [item.expected.recommended_action for item in examples],
        ),
    )


def robustness_distribution(
    examples: Sequence[LabeledExample],
) -> dict[RobustnessCategory, int]:
    counts = Counter(item.robustness_category for item in examples)
    return {category: counts[category] for category in RobustnessCategory}


def examples_fingerprint_digest(examples: Sequence[LabeledExample]) -> str:
    return fingerprint_set_digest(canonical_fingerprint(item.ticket) for item in examples)
