"""Portable JSONL/CSV queues for narrow human evidence review."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from modelforge.evaluation.judge import HumanEvidenceReview

CSV_FIELDS = (
    "schema_version",
    "case_id",
    "reviewer_id",
    "source_text",
    "predicted_evidence",
    "reference_evidence",
    "evidence_relevant",
    "unsupported_claim",
    "relevance_score",
    "notes",
)


def _validate_unique(rows: Sequence[HumanEvidenceReview]) -> None:
    ids = [row.case_id for row in rows]
    duplicates = sorted(case_id for case_id in set(ids) if ids.count(case_id) > 1)
    if duplicates:
        raise ValueError(f"duplicate human-review case ids: {duplicates}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as sink:
            sink.write(text)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_human_review_jsonl(rows: Sequence[HumanEvidenceReview], path: Path) -> None:
    """Write a deterministic, atomic JSONL review queue.

    Review queues contain ticket text by design and must be handled as sensitive
    artifacts; operational telemetry must never reuse this format.
    """

    _validate_unique(rows)
    ordered = sorted(rows, key=lambda row: row.case_id)
    text = "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"
        for row in ordered
    )
    _atomic_write(path, text)


def import_human_review_jsonl(
    path: Path,
    *,
    require_complete: bool = False,
) -> tuple[HumanEvidenceReview, ...]:
    rows: list[HumanEvidenceReview] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(HumanEvidenceReview.model_validate_json(line, strict=True))
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid human review: {exc}") from exc
    _validate_unique(rows)
    if require_complete and (incomplete := [row.case_id for row in rows if not row.complete]):
        raise ValueError(f"incomplete human reviews: {incomplete}")
    return tuple(rows)


def _bool_cell(value: bool | None) -> str:
    return "" if value is None else ("true" if value else "false")


def _parse_bool_cell(value: str, *, field: str, line_number: int) -> bool | None:
    normalized = value.strip().casefold()
    if not normalized:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"CSV row {line_number}: {field} must be true, false, or blank")


def export_human_review_csv(rows: Sequence[HumanEvidenceReview], path: Path) -> None:
    _validate_unique(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=CSV_FIELDS, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in sorted(rows, key=lambda item: item.case_id):
        writer.writerow(
            {
                "schema_version": row.schema_version,
                "case_id": row.case_id,
                "reviewer_id": row.reviewer_id,
                "source_text": row.source_text,
                "predicted_evidence": json.dumps(row.predicted_evidence, ensure_ascii=False),
                "reference_evidence": json.dumps(row.reference_evidence, ensure_ascii=False),
                "evidence_relevant": _bool_cell(row.evidence_relevant),
                "unsupported_claim": _bool_cell(row.unsupported_claim),
                "relevance_score": "" if row.relevance_score is None else repr(row.relevance_score),
                "notes": row.notes,
            }
        )
    _atomic_write(path, stream.getvalue())


def _parse_evidence_cell(value: str, *, field: str, line_number: int) -> tuple[str, ...]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"CSV row {line_number}: {field} is not JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"CSV row {line_number}: {field} must be a JSON string array")
    return tuple(parsed)


def import_human_review_csv(
    path: Path,
    *,
    require_complete: bool = False,
) -> tuple[HumanEvidenceReview, ...]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                f"unexpected CSV columns: {reader.fieldnames}; expected {list(CSV_FIELDS)}"
            )
        rows: list[HumanEvidenceReview] = []
        for line_number, raw in enumerate(reader, start=2):
            score_cell = raw["relevance_score"].strip()
            try:
                score = None if not score_cell else float(score_cell)
                row = HumanEvidenceReview(
                    schema_version=raw["schema_version"],
                    case_id=raw["case_id"],
                    reviewer_id=raw["reviewer_id"],
                    source_text=raw["source_text"],
                    predicted_evidence=_parse_evidence_cell(
                        raw["predicted_evidence"],
                        field="predicted_evidence",
                        line_number=line_number,
                    ),
                    reference_evidence=_parse_evidence_cell(
                        raw["reference_evidence"],
                        field="reference_evidence",
                        line_number=line_number,
                    ),
                    evidence_relevant=_parse_bool_cell(
                        raw["evidence_relevant"], field="evidence_relevant", line_number=line_number
                    ),
                    unsupported_claim=_parse_bool_cell(
                        raw["unsupported_claim"], field="unsupported_claim", line_number=line_number
                    ),
                    relevance_score=score,
                    notes=raw["notes"],
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise ValueError(f"CSV row {line_number}: invalid human review: {exc}") from exc
            rows.append(row)
    _validate_unique(rows)
    if require_complete and (incomplete := [row.case_id for row in rows if not row.complete]):
        raise ValueError(f"incomplete human reviews: {incomplete}")
    return tuple(rows)
