"""Small fixed cases that guard the evaluator's high-risk semantics.

These are harness fixtures, not claims about any real model.
"""

from __future__ import annotations

from modelforge.evaluation.failures import FailureCategory, build_failure_taxonomy
from modelforge.evaluation.metrics import evaluate_records
from modelforge.evaluation.models import EvaluationRecord, PredictionError
from modelforge.schemas import TriageResult


def label(
    *,
    issue_type: str,
    severity: str,
    action: str,
    evidence: list[str],
) -> dict:
    return {
        "issue_type": issue_type,
        "severity": severity,
        "routing_team": "IDENTITY_PLATFORM",
        "affected_scope": "MULTIPLE_USERS",
        "recommended_action": action,
        "evidence": evidence,
    }


def test_golden_difficult_cases_preserve_error_and_failure_semantics() -> None:
    sso = label(
        issue_type="SSO_AUTHENTICATION_FAILURE",
        severity="P2",
        action="INVESTIGATE_IDP_OR_SSO_CONFIGURATION",
        evidence=["2,000 users cannot sign in"],
    )
    mfa = label(
        issue_type="MFA_FAILURE",
        severity="P4",
        action="INVESTIGATE_MFA_CONFIGURATION",
        evidence=["my MFA prompt loops"],
    )
    records = [
        EvaluationRecord(
            case_id="calm-large-outage",
            expected=sso,
            prediction={**sso, "confidence": 0.9},
            input_text="A calm update: 2,000 users cannot sign in through SSO.",
            slice="scope-vs-tone",
        ),
        EvaluationRecord(
            case_id="angry-single-user",
            expected=mfa,
            prediction={**mfa, "severity": "P1", "confidence": 0.99},
            input_text="THIS IS TERRIBLE — my MFA prompt loops on one account.",
            slice="severity-vs-tone",
        ),
        EvaluationRecord(
            case_id="unsupported-evidence",
            expected=sso,
            prediction={
                **sso,
                "evidence": ["the identity provider is compromised"],
                "confidence": 0.95,
            },
            input_text="2,000 users cannot sign in; root cause is not known.",
            slice="unsupported-claim",
        ),
        EvaluationRecord(
            case_id="provider-outage",
            expected=sso,
            error=PredictionError(kind="provider_timeout", message="deadline", retriable=True),
            slice="operations",
        ),
    ]
    report = evaluate_records(records, schema_model=TriageResult)
    taxonomy = build_failure_taxonomy(report)
    categories = {bucket.category for bucket in taxonomy.buckets}

    assert report.aggregate.error_count == 1
    assert report.aggregate.wrong_predictions == 2
    assert report.aggregate.exact_match_count == 1
    assert report.slices["severity-vs-tone"].field_accuracies["severity"] == 0.0
    assert report.slices["unsupported-claim"].unsupported_evidence_rate == 1.0
    assert FailureCategory.SEVERITY in categories
    assert FailureCategory.EVIDENCE_UNSUPPORTED in categories
    assert FailureCategory.TIMEOUT in categories
