"""Versioned narrow judge contract and judge-vs-human validation.

The judge sees only ticket text and evidence snippets.  It cannot override the
deterministic schema or closed-label metrics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from modelforge.evaluation.metrics import evidence_items
from modelforge.evaluation.models import (
    EvaluationConfig,
    EvaluationReport,
    EvaluationStatus,
    FrozenModel,
    PredictionError,
)


class JudgeSpec(FrozenModel):
    judge_name: str = Field(min_length=1)
    judge_version: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def identity(self) -> str:
        return f"{self.judge_name}:{self.judge_version}:{self.rubric_version}:{self.prompt_sha256}"


class EvidenceJudgeRequest(FrozenModel):
    case_id: str = Field(min_length=1)
    source_text: str
    predicted_evidence: tuple[str, ...]
    reference_evidence: tuple[str, ...] = ()


class EvidenceJudgment(FrozenModel):
    evidence_relevant: bool
    unsupported_claim: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2_000)


class JudgeStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class JudgeResult(FrozenModel):
    case_id: str
    spec: JudgeSpec
    status: JudgeStatus
    judgment: EvidenceJudgment | None = None
    error: PredictionError | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _consistent_status(self) -> JudgeResult:
        if self.status == JudgeStatus.OK and (self.judgment is None or self.error is not None):
            raise ValueError("successful judge results require a judgment and no error")
        if self.status == JudgeStatus.ERROR and (self.error is None or self.judgment is not None):
            raise ValueError("errored judge results require an error and no judgment")
        return self


@runtime_checkable
class Judge(Protocol):
    """Async, injectable judge limited to evidence semantics."""

    @property
    def spec(self) -> JudgeSpec: ...

    async def judge(self, request: EvidenceJudgeRequest) -> JudgeResult: ...


class HumanEvidenceReview(FrozenModel):
    """Portable review row; optional labels allow blank review queues."""

    schema_version: Literal["human-evidence-review-v1"] = "human-evidence-review-v1"
    case_id: str = Field(min_length=1)
    reviewer_id: str = ""
    source_text: str = ""
    predicted_evidence: tuple[str, ...] = ()
    reference_evidence: tuple[str, ...] = ()
    evidence_relevant: bool | None = None
    unsupported_claim: bool | None = None
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str = ""

    @property
    def complete(self) -> bool:
        return self.evidence_relevant is not None and self.unsupported_claim is not None


class JudgeDisagreement(FrozenModel):
    case_id: str
    human_evidence_relevant: bool
    judge_evidence_relevant: bool
    human_unsupported_claim: bool
    judge_unsupported_claim: bool
    judge_rationale: str


class JudgeValidationReport(FrozenModel):
    human_review_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    missing_judge_count: int = Field(ge=0)
    judge_error_count: int = Field(ge=0)
    relevance_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    unsupported_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    joint_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    unsupported_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    rank_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    false_accept_count: int = Field(ge=0)
    false_reject_count: int = Field(ge=0)
    false_accept_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_reject_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreements: tuple[JudgeDisagreement, ...]


class EvidenceComparisonRow(FrozenModel):
    """Aligned deterministic/model, human, and judge evidence outcomes."""

    case_id: str
    model_status: str
    model_exact_match: bool | None
    gold_evidence_count: int = Field(ge=0)
    model_evidence_count: int = Field(ge=0)
    model_evidence_all_literal: bool | None
    human_evidence_relevant: bool | None
    human_unsupported_claim: bool | None
    judge_status: str
    judge_evidence_relevant: bool | None
    judge_unsupported_claim: bool | None


class EvidenceComparisonReport(FrozenModel):
    judge_identity: str | None
    rows: tuple[EvidenceComparisonRow, ...]


def cohen_kappa(left: Sequence[bool], right: Sequence[bool]) -> float | None:
    """Binary Cohen's kappa with a deterministic degenerate-case convention."""

    if len(left) != len(right):
        raise ValueError("kappa inputs must have equal length")
    if not left:
        return None
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / count
    left_true = sum(left) / count
    right_true = sum(right) / count
    expected = left_true * right_true + (1.0 - left_true) * (1.0 - right_true)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1.0 - expected)))


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for original_index, _ in indexed[start:end]:
            ranks[original_index] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties."""

    if len(left) != len(right):
        raise ValueError("rank inputs must have equal length")
    if len(left) < 2:
        return None
    x = _average_ranks(left)
    y = _average_ranks(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_scale = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if math.isclose(x_scale, 0.0) or math.isclose(y_scale, 0.0):
        return None
    return max(-1.0, min(1.0, numerator / (x_scale * y_scale)))


def validate_judge(
    human_reviews: Sequence[HumanEvidenceReview],
    judge_results: Sequence[JudgeResult],
    *,
    max_disagreements: int = 10,
) -> JudgeValidationReport:
    """Compare one versioned judge with completed human reviews."""

    if max_disagreements < 0:
        raise ValueError("max_disagreements must be non-negative")
    human_ids = [row.case_id for row in human_reviews]
    judge_ids = [row.case_id for row in judge_results]
    if len(human_ids) != len(set(human_ids)):
        raise ValueError("duplicate human review case ids")
    if len(judge_ids) != len(set(judge_ids)):
        raise ValueError("duplicate judge result case ids")
    judge_identities = {row.spec.identity for row in judge_results}
    if len(judge_identities) > 1:
        raise ValueError(f"mixed judge identities are not comparable: {sorted(judge_identities)}")

    judge_by_id = {row.case_id: row for row in judge_results}
    complete = sorted((row for row in human_reviews if row.complete), key=lambda row: row.case_id)
    matched: list[tuple[HumanEvidenceReview, EvidenceJudgment]] = []
    missing = errors = 0
    for human in complete:
        judge = judge_by_id.get(human.case_id)
        if judge is None:
            missing += 1
        elif judge.status == JudgeStatus.ERROR:
            errors += 1
        else:
            assert judge.judgment is not None
            matched.append((human, judge.judgment))

    human_relevance = [bool(h.evidence_relevant) for h, _ in matched]
    judge_relevance = [j.evidence_relevant for _, j in matched]
    human_unsupported = [bool(h.unsupported_claim) for h, _ in matched]
    judge_unsupported = [j.unsupported_claim for _, j in matched]
    count = len(matched)
    relevance_agreement = (
        sum(a == b for a, b in zip(human_relevance, judge_relevance, strict=True)) / count
        if count
        else None
    )
    unsupported_agreement = (
        sum(a == b for a, b in zip(human_unsupported, judge_unsupported, strict=True)) / count
        if count
        else None
    )
    joint_agreement = (
        sum(
            hr == jr and hu == ju
            for hr, jr, hu, ju in zip(
                human_relevance,
                judge_relevance,
                human_unsupported,
                judge_unsupported,
                strict=True,
            )
        )
        / count
        if count
        else None
    )

    rank_pairs = [
        (human.relevance_score, judgment.relevance_score)
        for human, judgment in matched
        if human.relevance_score is not None
    ]
    rank_correlation = spearman_rank_correlation(
        [float(left) for left, _ in rank_pairs],
        [right for _, right in rank_pairs],
    )

    human_accepts = [
        relevant and not unsupported
        for relevant, unsupported in zip(human_relevance, human_unsupported, strict=True)
    ]
    judge_accepts = [
        relevant and not unsupported
        for relevant, unsupported in zip(judge_relevance, judge_unsupported, strict=True)
    ]
    false_accept = sum(
        not human and judge for human, judge in zip(human_accepts, judge_accepts, strict=True)
    )
    false_reject = sum(
        human and not judge for human, judge in zip(human_accepts, judge_accepts, strict=True)
    )
    human_reject_count = sum(not value for value in human_accepts)
    human_accept_count = sum(human_accepts)

    disagreements = tuple(
        JudgeDisagreement(
            case_id=human.case_id,
            human_evidence_relevant=bool(human.evidence_relevant),
            judge_evidence_relevant=judgment.evidence_relevant,
            human_unsupported_claim=bool(human.unsupported_claim),
            judge_unsupported_claim=judgment.unsupported_claim,
            judge_rationale=judgment.rationale,
        )
        for human, judgment in matched
        if human.evidence_relevant != judgment.evidence_relevant
        or human.unsupported_claim != judgment.unsupported_claim
    )[:max_disagreements]

    return JudgeValidationReport(
        human_review_count=len(complete),
        matched_count=count,
        missing_judge_count=missing,
        judge_error_count=errors,
        relevance_agreement=relevance_agreement,
        unsupported_agreement=unsupported_agreement,
        joint_agreement=joint_agreement,
        relevance_kappa=cohen_kappa(human_relevance, judge_relevance),
        unsupported_kappa=cohen_kappa(human_unsupported, judge_unsupported),
        rank_correlation=rank_correlation,
        false_accept_count=false_accept,
        false_reject_count=false_reject,
        false_accept_rate=false_accept / human_reject_count if human_reject_count else None,
        false_reject_rate=false_reject / human_accept_count if human_accept_count else None,
        disagreements=disagreements,
    )


def build_evidence_comparison(
    evaluation: EvaluationReport,
    human_reviews: Sequence[HumanEvidenceReview],
    judge_results: Sequence[JudgeResult],
    *,
    config: EvaluationConfig | None = None,
) -> EvidenceComparisonReport:
    """Align gold/model deterministic results with optional human and judge labels."""

    resolved_config = config or EvaluationConfig()
    human_ids = [row.case_id for row in human_reviews]
    judge_ids = [row.case_id for row in judge_results]
    if len(human_ids) != len(set(human_ids)):
        raise ValueError("duplicate human review case ids")
    if len(judge_ids) != len(set(judge_ids)):
        raise ValueError("duplicate judge result case ids")
    identities = {row.spec.identity for row in judge_results}
    if len(identities) > 1:
        raise ValueError(f"mixed judge identities are not comparable: {sorted(identities)}")

    human_by_id = {row.case_id: row for row in human_reviews}
    judge_by_id = {row.case_id: row for row in judge_results}
    rows: list[EvidenceComparisonRow] = []
    for item in sorted(evaluation.items, key=lambda value: value.case_id):
        human = human_by_id.get(item.case_id)
        judge = judge_by_id.get(item.case_id)
        judgment = judge.judgment if judge is not None and judge.status == JudgeStatus.OK else None
        gold_count = len(evidence_items(item.expected.get(resolved_config.evidence_field)))
        model_count = item.evidence_count
        if item.status != EvaluationStatus.SCORED or not model_count:
            all_literal = None
        else:
            all_literal = item.unsupported_evidence_count == 0
        rows.append(
            EvidenceComparisonRow(
                case_id=item.case_id,
                model_status=item.status.value,
                model_exact_match=item.exact_match,
                gold_evidence_count=gold_count,
                model_evidence_count=model_count,
                model_evidence_all_literal=all_literal,
                human_evidence_relevant=(human.evidence_relevant if human is not None else None),
                human_unsupported_claim=(human.unsupported_claim if human is not None else None),
                judge_status=(judge.status.value if judge is not None else "missing"),
                judge_evidence_relevant=(
                    judgment.evidence_relevant if judgment is not None else None
                ),
                judge_unsupported_claim=(
                    judgment.unsupported_claim if judgment is not None else None
                ),
            )
        )
    return EvidenceComparisonReport(
        judge_identity=next(iter(identities), None),
        rows=tuple(rows),
    )
