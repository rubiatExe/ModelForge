from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelforge.evaluation.metrics import evaluate_records
from modelforge.evaluation.models import EvaluationRecord
from modelforge.evaluation.regression import (
    GateStatus,
    MetricFloor,
    RegressionPolicy,
    RunIdentity,
    case_ids_sha256,
    gate_regression,
)
from modelforge.schemas import TriageResult


def label(issue_type: str) -> dict:
    return {
        "issue_type": issue_type,
        "severity": "P2",
        "routing_team": "IDENTITY_PLATFORM",
        "affected_scope": "MULTIPLE_USERS",
        "recommended_action": "INVESTIGATE_IDP_OR_SSO_CONFIGURATION",
        "evidence": ["authentication failed"],
    }


def prediction(issue_type: str) -> dict:
    return {**label(issue_type), "confidence": 0.9}


def report(*, regress: bool = False):
    return evaluate_records(
        [
            EvaluationRecord(
                case_id="common-a",
                expected=label("SSO_AUTHENTICATION_FAILURE"),
                prediction=prediction("SSO_AUTHENTICATION_FAILURE"),
                input_text="authentication failed",
                slice="common",
            ),
            EvaluationRecord(
                case_id="adv-b",
                expected=label("MFA_FAILURE"),
                prediction=prediction("SSO_AUTHENTICATION_FAILURE" if regress else "MFA_FAILURE"),
                input_text="authentication failed",
                slice="adversarial",
            ),
        ],
        schema_model=TriageResult,
    )


def identity(
    evaluation_report, *, run_id: str, model: str, dataset_sha: str = "a" * 64
) -> RunIdentity:
    return RunIdentity(
        run_id=run_id,
        dataset_name="iam-seed",
        dataset_version="1.0.0",
        dataset_sha256=dataset_sha,
        split="test",
        case_ids_sha256=case_ids_sha256(evaluation_report),
        case_count=len(evaluation_report.items),
        evaluator_version=evaluation_report.evaluator_version,
        schema_version="triage-v1",
        model_identity=model,
        prompt_sha256=("b" if model == "base" else "c") * 64,
        generation_config_sha256="d" * 64,
    )


POLICY = RegressionPolicy(
    policy_name="ship-test",
    policy_version="1.0.0",
    overall=MetricFloor(
        max_error_rate=0.0,
        min_schema_validity_rate=1.0,
        min_exact_match_rate=0.9,
        min_issue_macro_f1=0.9,
        per_class_f1_floors={"SSO_AUTHENTICATION_FAILURE": 0.9, "MFA_FAILURE": 0.9},
    ),
    slice_floors={"adversarial": MetricFloor(min_exact_match_rate=1.0, min_issue_macro_f1=1.0)},
    max_exact_match_drop=0.0,
    max_issue_macro_f1_drop=0.0,
    max_per_class_f1_drop=0.0,
)


def test_regression_gate_passes_comparable_reports_and_allows_model_prompt_change() -> None:
    baseline = report()
    candidate = report()
    result = gate_regression(
        baseline,
        candidate,
        baseline_identity=identity(baseline, run_id="base", model="base"),
        candidate_identity=identity(candidate, run_id="candidate", model="candidate"),
        policy=POLICY,
    )
    assert result.status == GateStatus.PASS
    assert result.comparable is True
    assert result.exit_code == 0


def test_regression_gate_checks_comparability_before_quality() -> None:
    baseline = report()
    candidate = report(regress=True)
    result = gate_regression(
        baseline,
        candidate,
        baseline_identity=identity(baseline, run_id="base", model="base"),
        candidate_identity=identity(
            candidate,
            run_id="candidate",
            model="candidate",
            dataset_sha="f" * 64,
        ),
        policy=POLICY,
    )
    assert result.status == GateStatus.INCOMPARABLE
    assert result.exit_code == 1
    assert all(violation.code == "incomparable" for violation in result.violations)
    assert any("dataset_sha256 differs" in violation.detail for violation in result.violations)


def test_regression_gate_enforces_aggregate_per_class_and_adversarial_floors() -> None:
    baseline = report()
    candidate = report(regress=True)
    result = gate_regression(
        baseline,
        candidate,
        baseline_identity=identity(baseline, run_id="base", model="base"),
        candidate_identity=identity(candidate, run_id="candidate", model="candidate"),
        policy=POLICY,
    )
    metrics = {violation.metric for violation in result.violations}
    assert result.status == GateStatus.FAIL
    assert result.comparable is True
    assert result.exit_code == 1
    assert "exact_match_rate" in metrics
    assert "issue_macro_f1" in metrics
    assert "issue_f1.MFA_FAILURE" in metrics
    assert any(violation.slice == "adversarial" for violation in result.violations)


def test_metric_floor_rejects_out_of_range_mapping_values() -> None:
    with pytest.raises(ValidationError, match="metric floors must be in"):
        MetricFloor(field_accuracy_floors={"severity": 1.1})
