from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelforge.evaluation.human_review import (
    export_human_review_csv,
    export_human_review_jsonl,
    import_human_review_csv,
    import_human_review_jsonl,
)
from modelforge.evaluation.judge import (
    EvidenceComparisonReport,
    EvidenceJudgeRequest,
    EvidenceJudgment,
    HumanEvidenceReview,
    Judge,
    JudgeResult,
    JudgeSpec,
    JudgeStatus,
    build_evidence_comparison,
    cohen_kappa,
    validate_judge,
)
from modelforge.evaluation.metrics import evaluate_records
from modelforge.evaluation.models import EvaluationRecord, PredictionError
from modelforge.schemas import TriageResult

SPEC = JudgeSpec(
    judge_name="evidence-only",
    judge_version="1.0.0",
    rubric_version="1.0.0",
    model_name="fake-judge",
    prompt_sha256="a" * 64,
)


def judgment(
    *, relevant: bool, unsupported: bool, score: float, rationale: str = "checked"
) -> EvidenceJudgment:
    return EvidenceJudgment(
        evidence_relevant=relevant,
        unsupported_claim=unsupported,
        relevance_score=score,
        rationale=rationale,
    )


class FakeJudge:
    @property
    def spec(self) -> JudgeSpec:
        return SPEC

    async def judge(self, request: EvidenceJudgeRequest) -> JudgeResult:
        return JudgeResult(
            case_id=request.case_id,
            spec=self.spec,
            status=JudgeStatus.OK,
            judgment=judgment(relevant=True, unsupported=False, score=0.9),
        )


def test_judge_protocol_is_narrow_versioned_and_async() -> None:
    fake = FakeJudge()
    assert isinstance(fake, Judge)
    request = EvidenceJudgeRequest(
        case_id="case-1",
        source_text="MFA prompts loop",
        predicted_evidence=("MFA prompts loop",),
    )
    response = asyncio.run(fake.judge(request))
    assert response.spec.identity.startswith("evidence-only:1.0.0")
    assert response.judgment is not None
    assert response.judgment.evidence_relevant is True


def test_judge_result_status_invariants() -> None:
    with pytest.raises(ValidationError):
        JudgeResult(case_id="bad", spec=SPEC, status=JudgeStatus.OK)
    with pytest.raises(ValidationError):
        JudgeResult(
            case_id="bad",
            spec=SPEC,
            status=JudgeStatus.ERROR,
            judgment=judgment(relevant=True, unsupported=False, score=1.0),
        )


def review(
    case_id: str,
    *,
    relevant: bool | None,
    unsupported: bool | None,
    score: float | None,
) -> HumanEvidenceReview:
    return HumanEvidenceReview(
        case_id=case_id,
        reviewer_id="reviewer-1",
        source_text="literal ticket text",
        predicted_evidence=("literal ticket text",),
        reference_evidence=("literal ticket text",),
        evidence_relevant=relevant,
        unsupported_claim=unsupported,
        relevance_score=score,
    )


@pytest.mark.parametrize("suffix", ["jsonl", "csv"])
def test_human_review_export_import_round_trip(tmp_path: Path, suffix: str) -> None:
    rows = (
        review("b", relevant=None, unsupported=None, score=None),
        review("a", relevant=True, unsupported=False, score=0.9),
    )
    path = tmp_path / f"review.{suffix}"
    if suffix == "jsonl":
        export_human_review_jsonl(rows, path)
        loaded = import_human_review_jsonl(path)
    else:
        export_human_review_csv(rows, path)
        loaded = import_human_review_csv(path)
    assert [row.case_id for row in loaded] == ["a", "b"]
    assert loaded[0].predicted_evidence == ("literal ticket text",)
    assert loaded[1].complete is False

    importer = import_human_review_jsonl if suffix == "jsonl" else import_human_review_csv
    with pytest.raises(ValueError, match="incomplete"):
        importer(path, require_complete=True)


def test_human_review_csv_rejects_ambiguous_boolean(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    export_human_review_csv((review("a", relevant=True, unsupported=False, score=0.8),), path)
    path.write_text(path.read_text().replace(",true,false,", ",yes,false,"), encoding="utf-8")
    with pytest.raises(ValueError, match="must be true, false, or blank"):
        import_human_review_csv(path)


def test_judge_validation_reports_kappa_rank_and_false_acceptance() -> None:
    humans = (
        review("a", relevant=True, unsupported=False, score=0.9),
        review("b", relevant=False, unsupported=False, score=0.2),
        review("c", relevant=True, unsupported=True, score=0.7),
        review("missing", relevant=True, unsupported=False, score=0.5),
        review("judge-error", relevant=True, unsupported=False, score=0.5),
        review("incomplete", relevant=None, unsupported=None, score=None),
    )
    judges = (
        JudgeResult(
            case_id="a",
            spec=SPEC,
            status=JudgeStatus.OK,
            judgment=judgment(relevant=True, unsupported=False, score=0.8),
        ),
        JudgeResult(
            case_id="b",
            spec=SPEC,
            status=JudgeStatus.OK,
            judgment=judgment(relevant=True, unsupported=False, score=0.7, rationale="disagree"),
        ),
        JudgeResult(
            case_id="c",
            spec=SPEC,
            status=JudgeStatus.OK,
            judgment=judgment(relevant=True, unsupported=True, score=0.6),
        ),
        JudgeResult(
            case_id="judge-error",
            spec=SPEC,
            status=JudgeStatus.ERROR,
            error=PredictionError(kind="provider_error"),
        ),
    )

    report = validate_judge(humans, judges)
    assert report.human_review_count == 5
    assert report.matched_count == 3
    assert report.missing_judge_count == 1
    assert report.judge_error_count == 1
    assert report.relevance_agreement == pytest.approx(2 / 3)
    assert report.unsupported_agreement == 1.0
    assert report.joint_agreement == pytest.approx(2 / 3)
    assert report.unsupported_kappa == 1.0
    assert report.rank_correlation == pytest.approx(0.5)
    assert report.false_accept_count == 1
    assert report.false_accept_rate == pytest.approx(0.5)
    assert report.false_reject_count == 0
    assert report.disagreements[0].case_id == "b"


def test_kappa_degenerate_perfect_agreement_is_one() -> None:
    assert cohen_kappa([True, True], [True, True]) == 1.0
    assert cohen_kappa([], []) is None


def test_judge_validation_rejects_mixed_versions() -> None:
    alternate = SPEC.model_copy(update={"judge_version": "2.0.0"})
    humans = (
        review("a", relevant=True, unsupported=False, score=0.9),
        review("b", relevant=True, unsupported=False, score=0.8),
    )
    results = (
        JudgeResult(
            case_id="a",
            spec=SPEC,
            status=JudgeStatus.OK,
            judgment=judgment(relevant=True, unsupported=False, score=0.9),
        ),
        JudgeResult(
            case_id="b",
            spec=alternate,
            status=JudgeStatus.OK,
            judgment=judgment(relevant=True, unsupported=False, score=0.8),
        ),
    )
    with pytest.raises(ValueError, match="mixed judge identities"):
        validate_judge(humans, results)


def test_gold_model_human_judge_comparison_aligns_by_case_id() -> None:
    expected = {
        "issue_type": "MFA_FAILURE",
        "severity": "P2",
        "routing_team": "IDENTITY_PLATFORM",
        "affected_scope": "MULTIPLE_USERS",
        "recommended_action": "INVESTIGATE_MFA_CONFIGURATION",
        "evidence": ["literal ticket text"],
    }
    evaluation = evaluate_records(
        [
            EvaluationRecord(
                case_id="a",
                expected=expected,
                prediction={**expected, "confidence": 0.9},
                input_text="literal ticket text",
            )
        ],
        schema_model=TriageResult,
    )
    comparison = build_evidence_comparison(
        evaluation,
        (review("a", relevant=True, unsupported=False, score=1.0),),
        (
            JudgeResult(
                case_id="a",
                spec=SPEC,
                status=JudgeStatus.OK,
                judgment=judgment(relevant=True, unsupported=False, score=0.9),
            ),
        ),
    )
    assert isinstance(comparison, EvidenceComparisonReport)
    assert comparison.rows[0].model_exact_match is True
    assert comparison.rows[0].model_evidence_all_literal is True
    assert comparison.rows[0].human_evidence_relevant is True
    assert comparison.rows[0].judge_unsupported_claim is False
