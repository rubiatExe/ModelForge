from __future__ import annotations

from pathlib import Path

from modelforge.evaluation.failures import build_failure_taxonomy
from modelforge.evaluation.metrics import evaluate_records
from modelforge.evaluation.models import EvaluationRecord, EvaluationReport
from modelforge.evaluation.reporting import evaluation_markdown, failure_taxonomy_markdown
from modelforge.evaluation.serialization import (
    artifact_json,
    read_json_artifact,
    write_json_artifact,
)
from modelforge.schemas import TriageResult


def label() -> dict:
    return {
        "issue_type": "MFA_FAILURE",
        "severity": "P2",
        "routing_team": "IDENTITY_PLATFORM",
        "affected_scope": "MULTIPLE_USERS",
        "recommended_action": "INVESTIGATE_MFA_CONFIGURATION",
        "evidence": ["MFA prompts loop"],
    }


def test_evaluation_json_artifact_round_trip_is_deterministic(tmp_path: Path) -> None:
    report = evaluate_records(
        [
            EvaluationRecord(
                case_id="case-1",
                expected=label(),
                prediction={**label(), "confidence": 0.8},
                input_text="MFA prompts loop for every user",
            )
        ],
        schema_model=TriageResult,
    )
    path = tmp_path / "evaluation.json"
    write_json_artifact(path, report)
    loaded = read_json_artifact(path, EvaluationReport)
    assert loaded == report
    assert path.read_text() == artifact_json(report)
    assert path.read_text().endswith("\n")


def test_markdown_reports_are_readable_and_do_not_include_ticket_text() -> None:
    report = evaluate_records(
        [
            EvaluationRecord(
                case_id="case|unsafe",
                expected=label(),
                prediction={**label(), "severity": "P1", "confidence": 0.9},
                input_text="private ticket body MFA prompts loop",
            )
        ],
        schema_model=TriageResult,
    )
    summary = evaluation_markdown(report)
    failures = failure_taxonomy_markdown(build_failure_taxonomy(report))
    assert "issue macro F1" in summary
    assert "Errors are operational failures" in summary
    assert "severity_mismatch" in failures
    assert "case\\|unsafe" in failures
    assert "private ticket body" not in summary + failures
