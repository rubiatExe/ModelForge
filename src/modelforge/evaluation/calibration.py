"""Calibration diagnostics for an inspectable routing score.

The score is called a score—not a probability.  ECE is reported as a useful
diagnostic against empirical correctness, while risk/coverage is the primary
selective-prediction view.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pydantic import Field

from modelforge.evaluation.models import EvaluationReport, FrozenModel


class CalibrationObservation(FrozenModel):
    case_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    correct: bool
    slice: str = "default"


class CalibrationBin(FrozenModel):
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    average_score: float | None = Field(default=None, ge=0.0, le=1.0)
    empirical_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    absolute_gap: float | None = Field(default=None, ge=0.0, le=1.0)
    ece_contribution: float = Field(ge=0.0, le=1.0)


class RiskCoveragePoint(FrozenModel):
    threshold: float = Field(ge=0.0, le=1.0)
    accepted: int = Field(ge=0)
    total: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    selective_accuracy: float = Field(ge=0.0, le=1.0)


class CalibrationReport(FrozenModel):
    scored_count: int = Field(ge=0)
    missing_score_count: int = Field(ge=0)
    bin_count: int = Field(ge=1)
    ece: float | None = Field(default=None, ge=0.0, le=1.0)
    bins: tuple[CalibrationBin, ...]
    risk_coverage: tuple[RiskCoveragePoint, ...]


def expected_calibration_error(
    observations: Iterable[CalibrationObservation],
    *,
    bin_count: int = 10,
    missing_score_count: int = 0,
) -> CalibrationReport:
    """Return fixed-width ECE bins and a tie-stable risk/coverage curve."""

    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    rows = list(observations)
    grouped: defaultdict[int, list[CalibrationObservation]] = defaultdict(list)
    for row in rows:
        index = min(int(row.confidence * bin_count), bin_count - 1)
        grouped[index].append(row)

    bins: list[CalibrationBin] = []
    ece = 0.0
    for index in range(bin_count):
        members = grouped[index]
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if members:
            average = sum(item.confidence for item in members) / len(members)
            accuracy = sum(item.correct for item in members) / len(members)
            gap = abs(average - accuracy)
            contribution = gap * len(members) / len(rows)
            ece += contribution
        else:
            average = accuracy = gap = None
            contribution = 0.0
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(members),
                average_score=average,
                empirical_accuracy=accuracy,
                absolute_gap=gap,
                ece_contribution=contribution,
            )
        )

    return CalibrationReport(
        scored_count=len(rows),
        missing_score_count=missing_score_count,
        bin_count=bin_count,
        ece=ece if rows else None,
        bins=tuple(bins),
        risk_coverage=risk_coverage_curve(rows),
    )


def risk_coverage_curve(
    observations: Iterable[CalibrationObservation],
) -> tuple[RiskCoveragePoint, ...]:
    """Accept high-score predictions first, grouping tied scores atomically."""

    rows = sorted(observations, key=lambda item: (-item.confidence, item.case_id))
    total = len(rows)
    if not rows:
        return ()

    accepted = correct = 0
    points: list[RiskCoveragePoint] = []
    index = 0
    while index < total:
        threshold = rows[index].confidence
        tied: list[CalibrationObservation] = []
        while index < total and rows[index].confidence == threshold:
            tied.append(rows[index])
            index += 1
        accepted += len(tied)
        correct += sum(item.correct for item in tied)
        accuracy = correct / accepted
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                accepted=accepted,
                total=total,
                coverage=accepted / total,
                risk=1.0 - accuracy,
                selective_accuracy=accuracy,
            )
        )
    return tuple(points)


def calibration_from_evaluation(
    report: EvaluationReport,
    *,
    bin_count: int = 10,
) -> CalibrationReport:
    """Build calibration observations from non-error evaluation outcomes."""

    observations: list[CalibrationObservation] = []
    missing = 0
    for item in report.items:
        if item.error is not None:
            continue
        if item.confidence is None:
            missing += 1
            continue
        observations.append(
            CalibrationObservation(
                case_id=item.case_id,
                confidence=item.confidence,
                correct=item.exact_match is True,
                slice=item.slice,
            )
        )
    return expected_calibration_error(
        observations,
        bin_count=bin_count,
        missing_score_count=missing,
    )
