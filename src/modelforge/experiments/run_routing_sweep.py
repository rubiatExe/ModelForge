"""Derive a reproducible routing sweep from paired measured validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from modelforge.datasets import (
    load_manifest,
    load_validation_examples,
    sha256_file,
)
from modelforge.evaluation import (
    DEFAULT_THRESHOLDS,
    CachedPrediction,
    CachedRoutingCase,
    EvaluationConfig,
    RoutingSweepReport,
    read_json_artifact,
    sweep_routing_thresholds,
    write_json_artifact,
)
from modelforge.evaluation.models import EvaluationItemResult, EvaluationStatus
from modelforge.evaluation.regression import case_ids_sha256
from modelforge.experiments._artifacts import (
    canonical_sha256,
    require_available_outputs,
    write_svg_artifact,
)
from modelforge.experiments.run_evaluation import ModelEvaluationArtifact
from modelforge.routing import ConfidenceAssessor
from modelforge.schemas import IssueType, LabeledExample, TriageResult
from modelforge.schemas._base import StrictBaseModel

CONFIDENCE_ASSESSOR_VERSION = "confidence-assessor-v1"
CONFIDENCE_COMPONENT_WEIGHTS = {
    "structural": 0.10,
    "evidence_grounding": 0.25,
    "ambiguity": 0.25,
    "self_reported": 0.15,
    "label_logprob": 0.15,
    "sample_agreement": 0.10,
}


class EvaluationSourceIdentity(StrictBaseModel):
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_id: str
    artifact_version: str
    result_status: Literal["measured"]
    backend: Literal["hf-base", "hf-adapter", "frontier"]
    model_id: str
    observed_model_versions: tuple[str, ...]
    model_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str | None
    dataset_name: str
    dataset_version: str
    split: Literal["validation"]
    split_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(gt=0)
    full_split_count: int = Field(gt=0)
    partial_run: bool
    evaluator_version: str
    schema_name: str
    result_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_usd_per_million: float | None = Field(default=None, ge=0.0)
    output_usd_per_million: float | None = Field(default=None, ge=0.0)
    pricing_source: str | None = None
    pricing_as_of: date | None = None


class ComparabilityProof(StrictBaseModel):
    source_roles_verified: Literal[True] = True
    measured_status_verified: Literal[True] = True
    dataset_identity_verified: Literal[True] = True
    current_dataset_digest_verified: Literal[True] = True
    validation_split_verified: Literal[True] = True
    evaluator_and_schema_verified: Literal[True] = True
    exact_case_population_verified: Literal[True] = True
    gold_labels_verified: Literal[True] = True
    per_case_cost_and_latency_verified: Literal[True] = True


class ConfidenceCase(StrictBaseModel):
    case_id: str
    score: float = Field(ge=0.0, le=1.0)
    is_probability: Literal[False] = False
    hard_failure: bool
    components: dict[str, float | None]
    reasons: tuple[str, ...]


class ConfidenceAssessorArtifact(StrictBaseModel):
    version: Literal["confidence-assessor-v1"] = CONFIDENCE_ASSESSOR_VERSION
    is_probability: Literal[False] = False
    method: str
    component_weights: dict[str, float]
    optional_model_signals_available: Literal[False] = False
    successful_student_cases: tuple[ConfidenceCase, ...]
    unscored_student_case_ids: tuple[str, ...]


class QualityCostPoint(StrictBaseModel):
    threshold: float = Field(ge=0.0, le=1.0)
    end_to_end_exact_match_rate: float = Field(ge=0.0, le=1.0)
    average_cost_usd_per_request: float = Field(ge=0.0)
    frontier_rate: float = Field(ge=0.0, le=1.0)
    selected_error_rate: float = Field(ge=0.0, le=1.0)


class RoutingSweepArtifact(StrictBaseModel):
    artifact_version: Literal["1.0"] = "1.0"
    result_status: Literal["derived_from_measured_sources"] = (
        "derived_from_measured_sources"
    )
    experiment_id: str = Field(min_length=1)
    completed_at: datetime
    student_source: EvaluationSourceIdentity
    frontier_source: EvaluationSourceIdentity
    comparability: ComparabilityProof
    confidence_assessor: ConfidenceAssessorArtifact
    thresholds: tuple[float, ...]
    sweep: RoutingSweepReport
    quality_cost_points: tuple[QualityCostPoint, ...]
    graph_filename: str
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    privacy: dict[str, str | bool]
    limitations: tuple[str, ...]


def _load_source(path: Path) -> ModelEvaluationArtifact:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evaluation source must be a regular non-symlink file: {path}")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError(f"evaluation source exceeds 256 MiB: {path}")
    return read_json_artifact(path, ModelEvaluationArtifact)


def _source_identity(
    path: Path,
    artifact: ModelEvaluationArtifact,
    *,
    expected_backend: Literal["student", "frontier"],
) -> EvaluationSourceIdentity:
    if artifact.result_status != "measured":
        raise ValueError(
            f"{expected_backend} source must be measured; found {artifact.result_status!r}"
        )
    allowed = {"hf-base", "hf-adapter"} if expected_backend == "student" else {"frontier"}
    if artifact.backend not in allowed:
        raise ValueError(
            f"{expected_backend} source backend must be one of {sorted(allowed)}; "
            f"found {artifact.backend!r}"
        )
    if artifact.split != "validation":
        raise ValueError("routing thresholds may only be calibrated on the validation split")
    if artifact.evaluated_count != len(artifact.evaluation.items):
        raise ValueError(f"{expected_backend} evaluated_count does not match report items")
    if artifact.operations.attempted_cases != artifact.evaluated_count:
        raise ValueError(f"{expected_backend} operations do not cover every evaluated case")
    if artifact.operations.cost_accounted_cases != artifact.evaluated_count:
        raise ValueError(f"{expected_backend} source has incomplete per-case cost accounting")
    if artifact.operations.missing_cost_accounting_cases:
        raise ValueError(f"{expected_backend} source has incomplete per-case cost accounting")
    pricing_source: str | None = None
    pricing_as_of: date | None = None
    input_rate: float | None = None
    output_rate: float | None = None
    if expected_backend == "frontier":
        pricing_source = artifact.model_configuration.get("pricing_source")
        pricing_as_of_value = artifact.model_configuration.get("pricing_as_of")
        if not isinstance(pricing_source, str) or not pricing_source.strip():
            raise ValueError("frontier source is missing pricing_source provenance")
        if not isinstance(pricing_as_of_value, str) or not pricing_as_of_value.strip():
            raise ValueError("frontier source is missing pricing_as_of provenance")
        try:
            pricing_as_of = date.fromisoformat(pricing_as_of_value)
        except ValueError as exc:
            raise ValueError("frontier source has invalid pricing_as_of provenance") from exc
        for field in ("input_usd_per_million", "output_usd_per_million"):
            value = artifact.model_configuration.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"frontier source has invalid {field}")
        input_rate = float(artifact.model_configuration["input_usd_per_million"])
        output_rate = float(artifact.model_configuration["output_usd_per_million"])
    if artifact.case_ids_sha256 != case_ids_sha256(artifact.evaluation):
        raise ValueError(f"{expected_backend} case_ids_sha256 does not match report items")
    if not artifact.observed_model_versions:
        raise ValueError(f"{expected_backend} source has no observed model version")
    return EvaluationSourceIdentity(
        artifact_sha256=sha256_file(path),
        experiment_id=artifact.experiment_id,
        artifact_version=artifact.artifact_version,
        result_status="measured",
        backend=artifact.backend,
        model_id=artifact.model_id,
        observed_model_versions=artifact.observed_model_versions,
        model_configuration_sha256=canonical_sha256(artifact.model_configuration),
        prompt_sha256=artifact.prompt_sha256,
        dataset_name=artifact.dataset_name,
        dataset_version=artifact.dataset_version,
        split="validation",
        split_file_sha256=artifact.split_file_sha256,
        case_ids_sha256=artifact.case_ids_sha256,
        case_count=artifact.evaluated_count,
        full_split_count=artifact.full_split_count,
        partial_run=artifact.partial_run,
        evaluator_version=artifact.evaluation.evaluator_version,
        schema_name=artifact.evaluation.schema_name,
        result_schema_sha256=artifact.result_schema_sha256,
        input_usd_per_million=input_rate,
        output_usd_per_million=output_rate,
        pricing_source=pricing_source,
        pricing_as_of=pricing_as_of,
    )


def _require_comparable_sources(
    student: EvaluationSourceIdentity,
    frontier: EvaluationSourceIdentity,
    *,
    current_dataset_name: str,
    current_dataset_version: str,
    current_split_sha256: str,
    current_split_count: int,
    allow_partial: bool,
) -> None:
    fields = (
        "dataset_name",
        "dataset_version",
        "split",
        "split_file_sha256",
        "case_ids_sha256",
        "case_count",
        "full_split_count",
        "partial_run",
        "evaluator_version",
        "schema_name",
        "result_schema_sha256",
    )
    mismatches = [name for name in fields if getattr(student, name) != getattr(frontier, name)]
    if mismatches:
        raise ValueError(f"student and frontier sources are incomparable: {mismatches}")
    if student.model_id == frontier.model_id:
        raise ValueError("student and frontier sources must have distinct model identities")
    if student.partial_run and not allow_partial:
        raise ValueError("partial routing calibration is disabled; supply full validation runs")
    if (
        student.dataset_name != current_dataset_name
        or student.dataset_version != current_dataset_version
        or student.split_file_sha256 != current_split_sha256
        or student.full_split_count != current_split_count
    ):
        raise ValueError("source artifacts do not match the currently loaded validation dataset")
    if not allow_partial and student.case_count != current_split_count:
        raise ValueError("full validation artifacts must cover every current validation case")


def _item_map(
    artifact: ModelEvaluationArtifact,
    *,
    source_name: str,
) -> dict[str, EvaluationItemResult]:
    items = {item.case_id: item for item in artifact.evaluation.items}
    if len(items) != len(artifact.evaluation.items):
        raise ValueError(f"{source_name} source contains duplicate case ids")
    for item in items.values():
        if item.slice != "validation":
            raise ValueError(f"{source_name} report contains a non-validation item")
        if item.latency_ms is None or item.cost_usd is None:
            raise ValueError(f"{source_name} source lacks per-case latency or cost accounting")
    return items


def _cached_prediction(item: EvaluationItemResult) -> CachedPrediction:
    if item.status == EvaluationStatus.ERROR:
        if item.error is None:
            raise ValueError(f"errored case {item.case_id!r} has no typed error")
        return CachedPrediction(
            error=item.error,
            latency_ms=float(item.latency_ms),
            cost_usd=float(item.cost_usd),
        )
    if item.status == EvaluationStatus.SCHEMA_INVALID:
        # Invalid payloads are redacted by the evaluation runner. An empty JSON
        # object safely preserves the branch's invalid outcome for re-scoring.
        return CachedPrediction(
            raw_prediction="{}",
            latency_ms=float(item.latency_ms),
            cost_usd=float(item.cost_usd),
        )
    if item.prediction is None:
        raise ValueError(f"scored case {item.case_id!r} is missing its prediction")
    return CachedPrediction(
        prediction=item.prediction,
        latency_ms=float(item.latency_ms),
        cost_usd=float(item.cost_usd),
    )


def _build_cases(
    examples: list[LabeledExample],
    student_items: dict[str, EvaluationItemResult],
    frontier_items: dict[str, EvaluationItemResult],
) -> tuple[tuple[CachedRoutingCase, ...], ConfidenceAssessorArtifact]:
    examples_by_id = {example.example_id: example for example in examples}
    source_ids = set(student_items)
    if source_ids != set(frontier_items):
        raise ValueError("student and frontier sources contain different case populations")
    if not source_ids <= set(examples_by_id):
        unknown = sorted(source_ids - set(examples_by_id))
        raise ValueError(f"source artifacts contain unknown validation case ids: {unknown}")

    assessor = ConfidenceAssessor()
    assessments: list[ConfidenceCase] = []
    unscored: list[str] = []
    cases: list[CachedRoutingCase] = []
    for case_id in sorted(source_ids):
        example = examples_by_id[case_id]
        student_item = student_items[case_id]
        frontier_item = frontier_items[case_id]
        expected = example.expected.model_dump(mode="json")
        if student_item.expected != expected or frontier_item.expected != expected:
            raise ValueError(f"source gold label does not match the dataset for case {case_id!r}")

        score: float | None = None
        if student_item.status == EvaluationStatus.SCORED:
            if student_item.prediction is None:
                raise ValueError(f"scored student case {case_id!r} has no prediction")
            result = TriageResult.model_validate(student_item.prediction)
            assessment = assessor.assess(example.ticket, result)
            score = assessment.score
            assessments.append(
                ConfidenceCase(
                    case_id=case_id,
                    score=assessment.score,
                    hard_failure=assessment.hard_failure,
                    components=assessment.components,
                    reasons=assessment.reasons,
                )
            )
        else:
            unscored.append(case_id)

        cases.append(
            CachedRoutingCase.from_domain(
                example=example,
                student=_cached_prediction(student_item),
                frontier=_cached_prediction(frontier_item),
                student_score=score,
                slice_name="validation",
            )
        )
    return (
        tuple(cases),
        ConfidenceAssessorArtifact(
            method=(
                "Weighted geometric composite of structural validity, literal evidence grounding, "
                "ticket ambiguity, and model self-report, capped by ambiguity. The persisted score "
                "is an inspectable routing score, never a probability."
            ),
            component_weights=CONFIDENCE_COMPONENT_WEIGHTS,
            successful_student_cases=tuple(assessments),
            unscored_student_case_ids=tuple(unscored),
        ),
    )


def _quality_cost_points(sweep: RoutingSweepReport) -> tuple[QualityCostPoint, ...]:
    points: list[QualityCostPoint] = []
    for result in sweep.results:
        metrics = result.aggregate
        error_rate = metrics.selected_error_rate
        frontier_rate = metrics.frontier_rate
        cost = metrics.average_cost_usd
        if error_rate is None or frontier_rate is None or cost is None:
            raise ValueError("routing sweep produced incomplete aggregate accounting")
        exact_returned = metrics.exact_match_rate or 0.0
        points.append(
            QualityCostPoint(
                threshold=result.threshold,
                end_to_end_exact_match_rate=exact_returned * (1.0 - error_rate),
                average_cost_usd_per_request=cost,
                frontier_rate=frontier_rate,
                selected_error_rate=error_rate,
            )
        )
    return tuple(points)


def render_quality_cost_svg(points: tuple[QualityCostPoint, ...]) -> str:
    """Render a standalone SVG using numeric data only and no active content."""

    if not points:
        raise ValueError("quality-cost graph requires at least one point")
    width, height = 760, 500
    left, right, top, bottom = 92, 28, 42, 76
    plot_width = width - left - right
    plot_height = height - top - bottom
    costs = [point.average_cost_usd_per_request for point in points]
    cost_min, cost_max = min(costs), max(costs)
    if math.isclose(cost_min, cost_max):
        padding = max(abs(cost_min) * 0.1, 1e-6)
        cost_min -= padding
        cost_max += padding

    def x_position(cost: float) -> float:
        return left + (cost - cost_min) / (cost_max - cost_min) * plot_width

    def y_position(quality: float) -> float:
        return top + (1.0 - quality) * plot_height

    grid: list[str] = []
    for step in range(6):
        quality = step / 5
        y = y_position(quality)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#374151">{quality:.1f}</text>'
        )
    coordinates = " ".join(
        f"{x_position(point.average_cost_usd_per_request):.2f},"
        f"{y_position(point.end_to_end_exact_match_rate):.2f}"
        for point in points
    )
    marks: list[str] = []
    for point in points:
        x = x_position(point.average_cost_usd_per_request)
        y = y_position(point.end_to_end_exact_match_rate)
        marks.extend(
            (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2563eb"/>',
                f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-size="11" '
                f'fill="#111827">t={point.threshold:.2f}</text>',
            )
        )
    x_labels = (
        f'<text x="{left}" y="{height - bottom + 24}" text-anchor="middle" '
        f'font-size="11" fill="#374151">${cost_min:.6f}</text>',
        f'<text x="{width - right}" y="{height - bottom + 24}" text-anchor="middle" '
        f'font-size="11" fill="#374151">${cost_max:.6f}</text>',
    )
    return "\n".join(
        (
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title desc">',
            '<title id="title">Routing quality versus estimated cost</title>',
            '<desc id="desc">Validation exact-match quality including runtime errors, plotted '
            'against average sequential routing cost per request.</desc>',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            *grid,
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" '
            'stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
            f'y2="{height - bottom}" stroke="#111827" stroke-width="1.5"/>',
            f'<polyline points="{coordinates}" fill="none" stroke="#2563eb" '
            'stroke-width="2"/>',
            *marks,
            *x_labels,
            f'<text x="{width / 2:.2f}" y="{height - 18}" text-anchor="middle" '
            'font-size="13" fill="#111827">Average estimated cost (USD/request)</text>',
            f'<text x="20" y="{height / 2:.2f}" text-anchor="middle" font-size="13" '
            'fill="#111827" transform="rotate(-90 20 250)">End-to-end exact-match rate</text>',
            "</svg>",
        )
    )


def run_routing_sweep(
    *,
    student_artifact_path: Path,
    frontier_artifact_path: Path,
    dataset_root: Path,
    output_path: Path,
    graph_path: Path,
    experiment_id: str,
    allow_partial: bool = False,
) -> RoutingSweepArtifact:
    """Validate source provenance, sweep fixed thresholds, and persist derived evidence."""

    require_available_outputs((output_path, graph_path), overwrite=False)
    if student_artifact_path.resolve() == frontier_artifact_path.resolve():
        raise ValueError("student and frontier source paths must be distinct")
    student_artifact = _load_source(student_artifact_path)
    frontier_artifact = _load_source(frontier_artifact_path)
    student_source = _source_identity(
        student_artifact_path,
        student_artifact,
        expected_backend="student",
    )
    frontier_source = _source_identity(
        frontier_artifact_path,
        frontier_artifact,
        expected_backend="frontier",
    )

    manifest = load_manifest(dataset_root)
    examples = load_validation_examples(dataset_root)
    validation_manifest = next(
        item for item in manifest.splits if item.split.value == "validation"
    )
    _require_comparable_sources(
        student_source,
        frontier_source,
        current_dataset_name=manifest.dataset_name,
        current_dataset_version=manifest.dataset_version,
        current_split_sha256=validation_manifest.sha256,
        current_split_count=len(examples),
        allow_partial=allow_partial,
    )
    student_items = _item_map(student_artifact, source_name="student")
    frontier_items = _item_map(frontier_artifact, source_name="frontier")
    cases, confidence = _build_cases(examples, student_items, frontier_items)
    if {case.case_id for case in cases} != set(student_items):
        raise ValueError("cached routing cases do not preserve the source case population")

    config = EvaluationConfig(issue_labels=tuple(item.value for item in IssueType))
    sweep = sweep_routing_thresholds(
        cases,
        thresholds=DEFAULT_THRESHOLDS,
        schema_model=TriageResult,
        config=config,
        calibration_split="validation",
    )
    points = _quality_cost_points(sweep)
    svg = render_quality_cost_svg(points)
    svg_payload = svg.rstrip() + "\n"
    artifact = RoutingSweepArtifact(
        experiment_id=experiment_id,
        completed_at=datetime.now(UTC),
        student_source=student_source,
        frontier_source=frontier_source,
        comparability=ComparabilityProof(),
        confidence_assessor=confidence,
        thresholds=DEFAULT_THRESHOLDS,
        sweep=sweep,
        quality_cost_points=points,
        graph_filename=graph_path.name,
        graph_sha256=hashlib.sha256(svg_payload.encode("utf-8")).hexdigest(),
        privacy={
            "classification": "derived_evaluation_artifact",
            "contains_raw_ticket_text": False,
            "contains_predictions_or_gold_labels": False,
            "contains_case_ids": True,
        },
        limitations=(
            "Thresholds are calibrated on validation data and must not be re-selected on test data.",
            "Latency assumes sequential student-first fallback and is not a concurrent load test.",
            "Costs are estimates from source adapters, not reconciled provider invoices.",
            "The confidence score is inspectable but uncalibrated and is not a probability.",
            "Optional label-logprob and repeated-sample signals are unavailable in cached evaluation artifacts.",
        ),
    )
    write_svg_artifact(graph_path, svg)
    write_json_artifact(output_path, artifact)
    return artifact


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-artifact", type=Path, required=True)
    parser.add_argument("--frontier-artifact", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "data" / "iam_triage_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a matching subset of validation cases (never final-test data).",
    )
    args = parser.parse_args()
    result = run_routing_sweep(
        student_artifact_path=args.student_artifact,
        frontier_artifact_path=args.frontier_artifact,
        dataset_root=args.dataset_root,
        output_path=args.output,
        graph_path=args.graph_output,
        experiment_id=args.experiment_id,
        allow_partial=args.allow_partial,
    )
    print(
        f"{result.experiment_id}: derived {len(result.quality_cost_points)} validation "
        "threshold points from two measured source artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
