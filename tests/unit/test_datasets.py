from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import modelforge.datasets.io as dataset_io
from modelforge.datasets import (
    ConflictingLabelError,
    CrossSplitLeakageError,
    DatasetIOError,
    DigestMismatchError,
    DuplicateExampleError,
    JSONLLineError,
    PurposeViolationError,
    atomic_write_bytes,
    canonical_fingerprint,
    canonical_json,
    load_dataset_split,
    load_manifest,
    load_test_examples,
    load_training_examples,
    normalize_for_fingerprint,
    normalize_text,
    read_jsonl,
    sha256_file,
    validate_dataset_splits,
    validate_no_conflicting_labels,
    validate_no_cross_split_leakage,
    validate_no_duplicates,
    verify_test_lock,
)
from modelforge.schemas import (
    AffectedScope,
    AnnotationStatus,
    DatasetPurpose,
    DatasetSource,
    DatasetSplit,
    ExampleProvenance,
    IssueType,
    LabeledExample,
    RecommendedAction,
    ReviewStatus,
    RobustnessCategory,
    RoutingTeam,
    Severity,
    TicketInput,
    TriageLabel,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = REPOSITORY_ROOT / "data" / "iam_triage_v1"


def make_example(
    *,
    example_id: str = "MF-TR-9001",
    split: DatasetSplit = DatasetSplit.TRAIN,
    ticket_id: str = "TKT-9001",
    subject: str = "SSO redirects to login",
    body: str = "I approve MFA, then the browser returns to the sign-in screen.",
    issue_type: IssueType = IssueType.SSO_AUTHENTICATION_FAILURE,
    action: RecommendedAction = RecommendedAction.INVESTIGATE_IDP_OR_SSO_CONFIGURATION,
    robustness_category: RobustnessCategory | None = None,
    submitted_at: datetime | None = None,
) -> LabeledExample:
    return LabeledExample(
        example_id=example_id,
        split=split,
        ticket=TicketInput(
            ticket_id=ticket_id,
            subject=subject,
            body=body,
            employee_department="Engineering",
            submitted_at=submitted_at or datetime(2026, 8, 20, tzinfo=UTC),
        ),
        expected=TriageLabel(
            issue_type=issue_type,
            severity=Severity.P3,
            routing_team=RoutingTeam.IDENTITY_PLATFORM,
            affected_scope=AffectedScope.SINGLE_USER,
            recommended_action=action,
            evidence=["returns to the sign-in screen"],
        ),
        provenance=ExampleProvenance(
            source=DatasetSource.SYNTHETIC_TEMPLATE_GENERATOR,
            generator="tests",
            generator_version="1.0.0",
            template_id="test-template",
            seed=1,
        ),
        annotation_status=AnnotationStatus.SYNTHETICALLY_LABELED,
        review_status=ReviewStatus.PENDING_HUMAN_REVIEW,
        robustness_category=robustness_category,
    )


def test_unicode_and_whitespace_normalization_is_canonical() -> None:
    assert normalize_text("  ＳＳＯ\t\n  failed  ") == "SSO failed"
    assert normalize_for_fingerprint("  Straße\t") == "strasse"


def test_fingerprint_ignores_record_metadata_and_normalizes_content() -> None:
    first = make_example()
    second = make_example(
        example_id="MF-TR-9002",
        ticket_id="TKT-9002",
        subject="  ＳＳＯ\tredirects to login ",
        body="I approve MFA, then the browser  returns to the sign-in screen.",
        submitted_at=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert canonical_fingerprint(first.ticket) == canonical_fingerprint(second.ticket)


def test_duplicate_and_conflicting_label_detection() -> None:
    first = make_example()
    duplicate = make_example(example_id="MF-TR-9002", ticket_id="TKT-9002")
    with pytest.raises(DuplicateExampleError, match="duplicate canonical input"):
        validate_no_duplicates([first, duplicate])

    conflicting = make_example(
        example_id="MF-TR-9003",
        ticket_id="TKT-9003",
        issue_type=IssueType.OTHER_IAM,
        action=RecommendedAction.COLLECT_MORE_INFORMATION,
    )
    with pytest.raises(ConflictingLabelError, match="conflicting labels"):
        validate_no_conflicting_labels([first, conflicting])


def test_cross_split_leakage_uses_content_not_ticket_id() -> None:
    train = make_example()
    test = make_example(
        example_id="MF-TE-9001",
        split=DatasetSplit.TEST,
        ticket_id="TKT-TE-9001",
        submitted_at=datetime(2027, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(CrossSplitLeakageError, match="leaks across splits"):
        validate_no_cross_split_leakage({DatasetSplit.TRAIN: [train], DatasetSplit.TEST: [test]})


def test_line_aware_jsonl_reports_the_exact_bad_line(tmp_path: Path) -> None:
    valid = canonical_json(make_example().model_dump(mode="json"))
    path = tmp_path / "examples.jsonl"
    path.write_text(f"{valid}\n[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(JSONLLineError) as captured:
        read_jsonl(path, LabeledExample)
    assert captured.value.line_number == 2
    assert "must be an object" in str(captured.value)


@pytest.mark.parametrize("bad_line", ["", "NaN", '{"unexpected":true}'])
def test_jsonl_rejects_blank_nonstandard_and_invalid_records(
    tmp_path: Path,
    bad_line: str,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(f"{bad_line}\n", encoding="utf-8")
    with pytest.raises(JSONLLineError):
        read_jsonl(path, LabeledExample)


def test_jsonl_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    payload = make_example().model_dump(mode="json")
    valid = canonical_json(payload)
    duplicate_id = valid[:-1] + ',"example_id":"MF-TR-9999"}'
    path = tmp_path / "duplicate-key.jsonl"
    path.write_text(f"{duplicate_id}\n", encoding="utf-8")
    with pytest.raises(JSONLLineError, match="duplicate JSON object key"):
        read_jsonl(path, LabeledExample)


def test_jsonl_requires_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "missing-newline.jsonl"
    path.write_text(canonical_json(make_example().model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(JSONLLineError, match="must end with a newline"):
        read_jsonl(path, LabeledExample)


def test_jsonl_refuses_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text(
        f"{canonical_json(make_example().model_dump(mode='json'))}\n",
        encoding="utf-8",
    )
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(DatasetIOError, match="refusing to read symlink"):
        read_jsonl(link, LabeledExample)


def test_atomic_write_preserves_previous_file_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"old\n")

    def fail_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        raise OSError(f"cannot replace {source} with {target}")

    monkeypatch.setattr(dataset_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        atomic_write_bytes(destination, b"new\n")
    assert destination.read_bytes() == b"old\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_training_purpose_cannot_load_test_or_adversarial() -> None:
    for forbidden in (DatasetSplit.TEST, DatasetSplit.ADVERSARIAL):
        with pytest.raises(PurposeViolationError, match="may not load"):
            load_dataset_split(
                DATASET_DIRECTORY,
                forbidden,
                purpose=DatasetPurpose.TRAINING,
            )


def test_committed_dataset_counts_hashes_labels_and_isolation() -> None:
    manifest = load_manifest(DATASET_DIRECTORY)
    lock = verify_test_lock(DATASET_DIRECTORY)
    splits = {
        DatasetSplit.TRAIN: load_training_examples(DATASET_DIRECTORY),
        DatasetSplit.VALIDATION: load_dataset_split(
            DATASET_DIRECTORY,
            DatasetSplit.VALIDATION,
            purpose=DatasetPurpose.VALIDATION,
        ),
        DatasetSplit.TEST: load_test_examples(DATASET_DIRECTORY),
        DatasetSplit.ADVERSARIAL: load_dataset_split(
            DATASET_DIRECTORY,
            DatasetSplit.ADVERSARIAL,
            purpose=DatasetPurpose.ADVERSARIAL_EVALUATION,
        ),
    }
    assert {split: len(examples) for split, examples in splits.items()} == {
        DatasetSplit.TRAIN: 500,
        DatasetSplit.VALIDATION: 100,
        DatasetSplit.TEST: 200,
        DatasetSplit.ADVERSARIAL: 100,
    }
    assert manifest.total_count == 900
    assert lock.count == 200
    validate_dataset_splits(splits)

    for split_manifest in manifest.splits:
        assert set(split_manifest.label_distribution.issue_type) == set(IssueType)
        assert set(split_manifest.label_distribution.severity) == set(Severity)
        assert set(split_manifest.label_distribution.routing_team) == set(RoutingTeam)
        assert set(split_manifest.label_distribution.affected_scope) == set(AffectedScope)
        assert set(split_manifest.label_distribution.recommended_action) == set(RecommendedAction)

    adversarial = splits[DatasetSplit.ADVERSARIAL]
    assert {item.robustness_category for item in adversarial} == set(RobustnessCategory)
    for category in RobustnessCategory:
        covered_issues = {
            item.expected.issue_type for item in adversarial if item.robustness_category is category
        }
        assert len(covered_issues) >= 5
    security_review_examples = [
        item
        for item in sum(splits.values(), [])
        if item.expected.recommended_action is RecommendedAction.ESCALATE_SECURITY_REVIEW
    ]
    assert security_review_examples
    assert all(
        item.expected.severity in {Severity.P1, Severity.P2} for item in security_review_examples
    )
    assert all(
        evidence.casefold() in f"{item.ticket.subject}\n{item.ticket.body}".casefold()
        for item in sum(splits.values(), [])
        for evidence in item.expected.evidence
    )


def test_test_lock_detects_file_tampering(tmp_path: Path) -> None:
    copy = tmp_path / "dataset"
    copy.mkdir()
    for source in DATASET_DIRECTORY.iterdir():
        (copy / source.name).write_bytes(source.read_bytes())
    with (copy / "test.jsonl").open("ab") as handle:
        handle.write(b" \n")
    with pytest.raises(DigestMismatchError, match="SHA-256 mismatch"):
        verify_test_lock(copy)


def test_generator_is_deterministic_across_fresh_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")}
    for destination in (first, second):
        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "generate_dataset.py"),
                "--output-dir",
                str(destination),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

    assert {item.name for item in first.iterdir()} == {item.name for item in second.iterdir()}
    assert {item.name: sha256_file(item) for item in first.iterdir()} == {
        item.name: sha256_file(item) for item in second.iterdir()
    }


def test_generator_is_idempotent_for_an_identical_version(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")}
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "generate_dataset.py"),
        "--output-dir",
        str(destination),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    before = {item.name: item.read_bytes() for item in destination.iterdir()}
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    after = {item.name: item.read_bytes() for item in destination.iterdir()}
    assert after == before
