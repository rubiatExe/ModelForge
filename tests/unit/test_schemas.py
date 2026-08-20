from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from modelforge.schemas import (
    AffectedScope,
    AnnotationStatus,
    DatasetSplit,
    ExampleProvenance,
    IssueType,
    LabeledExample,
    RecommendedAction,
    ReviewStatus,
    RobustnessCategory,
    RoutingTeam,
    Severity,
    SplitManifest,
    TicketInput,
    TriageLabel,
    TriageResult,
)


def valid_ticket(**updates: object) -> TicketInput:
    values: dict[str, object] = {
        "ticket_id": "TKT-00421",
        "subject": "SSO keeps redirecting after migration",
        "body": (
            "I approve MFA, then get sent back to the login screen. "
            "Four people on my team see the same behavior."
        ),
        "employee_department": "Sales",
        "submitted_at": datetime(2026, 8, 20, 14, 32, tzinfo=UTC),
    }
    values.update(updates)
    return TicketInput.model_validate(values)


def valid_label(**updates: object) -> TriageLabel:
    values: dict[str, object] = {
        "issue_type": IssueType.SSO_AUTHENTICATION_FAILURE,
        "severity": Severity.P2,
        "routing_team": RoutingTeam.IDENTITY_PLATFORM,
        "affected_scope": AffectedScope.MULTIPLE_USERS,
        "recommended_action": RecommendedAction.INVESTIGATE_IDP_OR_SSO_CONFIGURATION,
        "evidence": [
            "approve MFA",
            "Four people on my team see the same behavior",
        ],
    }
    values.update(updates)
    return TriageLabel.model_validate(values)


def valid_example(**updates: object) -> LabeledExample:
    values: dict[str, object] = {
        "example_id": "MF-TR-0001",
        "split": DatasetSplit.TRAIN,
        "ticket": valid_ticket(),
        "expected": valid_label(),
        "provenance": ExampleProvenance(
            source="SYNTHETIC_TEMPLATE_GENERATOR",
            generator="scripts/generate_dataset.py",
            generator_version="1.0.0",
            template_id="sso-loop-core",
            seed=20260820,
        ),
        "annotation_status": AnnotationStatus.SYNTHETICALLY_LABELED,
        "review_status": ReviewStatus.PENDING_HUMAN_REVIEW,
        "robustness_category": None,
    }
    values.update(updates)
    return LabeledExample.model_validate(values)


def test_allowed_iam_enums_are_closed_and_complete() -> None:
    assert {item.value for item in IssueType} == {
        "SSO_AUTHENTICATION_FAILURE",
        "MFA_FAILURE",
        "ACCOUNT_LOCKED_OR_DISABLED",
        "ACCESS_DENIED_AFTER_LOGIN",
        "PROVISIONING_OR_SYNC_FAILURE",
        "OTHER_IAM",
    }
    assert {item.value for item in Severity} == {"P1", "P2", "P3", "P4"}
    assert {item.value for item in RoutingTeam} == {
        "IDENTITY_PLATFORM",
        "ACCESS_MANAGEMENT",
        "SERVICE_DESK",
    }
    assert {item.value for item in AffectedScope} == {
        "SINGLE_USER",
        "MULTIPLE_USERS",
        "DEPARTMENT",
        "ORGANIZATION",
        "UNKNOWN",
    }
    assert {item.value for item in RecommendedAction} == {
        "RESET_OR_UNLOCK_ACCOUNT",
        "INVESTIGATE_MFA_CONFIGURATION",
        "INVESTIGATE_IDP_OR_SSO_CONFIGURATION",
        "REVIEW_ACCESS_ENTITLEMENTS",
        "INVESTIGATE_PROVISIONING_SYNC",
        "COLLECT_MORE_INFORMATION",
        "ESCALATE_SECURITY_REVIEW",
    }


@pytest.mark.parametrize("model", [TicketInput, TriageLabel, TriageResult, LabeledExample])
def test_domain_schemas_forbid_extra_fields(model: type) -> None:
    payloads = {
        TicketInput: valid_ticket().model_dump(mode="json"),
        TriageLabel: valid_label().model_dump(mode="json"),
        TriageResult: {
            **valid_label().model_dump(mode="json"),
            "confidence": 0.91,
        },
        LabeledExample: valid_example().model_dump(mode="json"),
    }
    payload = payloads[model]
    payload["unexpected"] = "forbidden"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(payload)


def test_ticket_rejects_naive_timestamp_and_control_characters() -> None:
    with pytest.raises(ValidationError, match="UTC offset"):
        valid_ticket(submitted_at=datetime(2026, 8, 20, 14, 32))
    with pytest.raises(ValidationError, match="control character"):
        valid_ticket(body="A sufficiently long ticket body with a null byte \x00 here")


def test_ticket_accepts_rfc3339_timestamp_from_json() -> None:
    payload = valid_ticket().model_dump(mode="json")
    payload["submitted_at"] = "2026-08-20T14:32:00Z"
    assert TicketInput.model_validate(payload).submitted_at.utcoffset() is not None


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_result_rejects_out_of_range_or_nonfinite_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        TriageResult.model_validate(
            {**valid_label().model_dump(mode="json"), "confidence": confidence}
        )


@pytest.mark.parametrize("confidence", [True, "0.9", None])
def test_result_does_not_coerce_confidence(confidence: object) -> None:
    with pytest.raises(ValidationError, match="JSON number"):
        TriageResult.model_validate(
            {**valid_label().model_dump(mode="json"), "confidence": confidence}
        )


def test_invalid_enum_and_duplicate_evidence_are_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_label(issue_type="PASSWORD_RESET")
    with pytest.raises(ValidationError, match="must be unique"):
        valid_label(evidence=["approve MFA", "  APPROVE\tMFA  "])


def test_labeled_example_requires_literal_grounded_evidence() -> None:
    unsupported = valid_label(evidence=["The identity provider configuration is broken"])
    with pytest.raises(ValidationError, match="not a literal ticket substring"):
        valid_example(expected=unsupported)


def test_adversarial_category_is_required_only_for_adversarial_split() -> None:
    with pytest.raises(ValidationError, match="require a robustness_category"):
        valid_example(split=DatasetSplit.ADVERSARIAL)
    with pytest.raises(ValidationError, match="only valid for adversarial"):
        valid_example(robustness_category=RobustnessCategory.PROMPT_INJECTION)

    result = valid_example(
        split=DatasetSplit.ADVERSARIAL,
        robustness_category=RobustnessCategory.PROMPT_INJECTION,
    )
    assert result.robustness_category is RobustnessCategory.PROMPT_INJECTION


def test_schema_json_explicitly_forbids_additional_properties() -> None:
    assert TicketInput.model_json_schema()["additionalProperties"] is False
    assert TriageResult.model_json_schema()["additionalProperties"] is False


def test_manifest_counts_do_not_coerce_strings_or_booleans() -> None:
    zero_distribution = {
        "issue_type": {item.value: 0 for item in IssueType},
        "severity": {item.value: 0 for item in Severity},
        "routing_team": {item.value: 0 for item in RoutingTeam},
        "affected_scope": {item.value: 0 for item in AffectedScope},
        "recommended_action": {item.value: 0 for item in RecommendedAction},
    }
    base = {
        "split": "train",
        "file": "train.jsonl",
        "count": 0,
        "sha256": "0" * 64,
        "fingerprint_sha256": "1" * 64,
        "label_distribution": zero_distribution,
        "robustness_distribution": {},
    }
    for invalid in ("0", False):
        with pytest.raises(ValidationError):
            SplitManifest.model_validate({**base, "count": invalid})
