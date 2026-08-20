"""Measure how deterministic training-label corruption affects the classical baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from modelforge.datasets import (
    load_manifest,
    load_test_examples,
    load_training_examples,
    load_validation_examples,
    sha256_file,
)
from modelforge.evaluation.models import ClassificationMetrics
from modelforge.evaluation.serialization import write_json_artifact
from modelforge.experiments._artifacts import require_available_outputs
from modelforge.experiments.run_classical import SplitResult, measure_split
from modelforge.models.classical import TfidfIssueTypeModel
from modelforge.schemas import IssueType, LabeledExample
from modelforge.schemas._base import StrictBaseModel


class AblationSplitResult(StrictBaseModel):
    count: int = Field(ge=0)
    metrics: ClassificationMetrics


class IssueTypeErrorExample(StrictBaseModel):
    case_id: str
    expected_issue_type: IssueType
    predicted_issue_type: IssueType


class IssueTypeErrorAnalysis(StrictBaseModel):
    misclassified_count: int = Field(ge=0)
    representative_examples: tuple[IssueTypeErrorExample, ...]


class DataQualityAblationPoint(StrictBaseModel):
    label_noise_fraction: float = Field(ge=0.0, le=1.0)
    corrupted_label_count: int = Field(ge=0)
    corruption_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_seconds: float = Field(ge=0.0)
    model_version: str
    validation: AblationSplitResult
    test: AblationSplitResult
    validation_errors: IssueTypeErrorAnalysis | None = None
    test_errors: IssueTypeErrorAnalysis | None = None


class DataQualityAblationArtifact(StrictBaseModel):
    artifact_version: Literal["1.0", "1.1"] = "1.1"
    result_status: Literal["measured"] = "measured"
    experiment_id: str = "classical_label_quality_ablation_v1"
    completed_at: datetime
    hypothesis: str
    dataset_name: str
    dataset_version: str
    train_file_sha256: str
    validation_file_sha256: str
    test_file_sha256: str
    test_lock_sha256: str
    seed: int
    method: str
    points: tuple[DataQualityAblationPoint, ...]
    conclusion: str
    limitations: tuple[str, ...]


def corrupted_labels(
    examples: list[LabeledExample],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[IssueType], tuple[tuple[str, str, str], ...]]:
    """Replace an exact seeded subset with a different allowed label."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    count = round(len(examples) * fraction)
    selected = set(random.Random(seed).sample(range(len(examples)), count))
    labels = tuple(IssueType)
    output: list[IssueType] = []
    plan: list[tuple[str, str, str]] = []
    for index, example in enumerate(examples):
        original = example.expected.issue_type
        if index in selected:
            replacement = labels[(labels.index(original) + 1) % len(labels)]
            output.append(replacement)
            plan.append((example.example_id, original.value, replacement.value))
        else:
            output.append(original)
    return output, tuple(sorted(plan))


def _plan_sha256(plan: tuple[tuple[str, str, str], ...]) -> str:
    payload = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_result(result: SplitResult) -> AblationSplitResult:
    return AblationSplitResult(count=result.count, metrics=result.metrics)


def _issue_type_error_analysis(
    model: TfidfIssueTypeModel,
    examples: list[LabeledExample],
    *,
    max_examples: int = 10,
) -> IssueTypeErrorAnalysis:
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    errors = [
        IssueTypeErrorExample(
            case_id=example.example_id,
            expected_issue_type=example.expected.issue_type,
            predicted_issue_type=prediction.issue_type,
        )
        for example in examples
        if (prediction := model.predict_issue_type(example.ticket)).issue_type
        != example.expected.issue_type
    ]
    errors.sort(key=lambda item: item.case_id)
    return IssueTypeErrorAnalysis(
        misclassified_count=len(errors),
        representative_examples=tuple(errors[:max_examples]),
    )


def run_data_quality_ablation(
    *,
    dataset_root: Path,
    output_path: Path,
    noise_fractions: tuple[float, ...] = (0.0, 0.1, 0.3),
    seed: int = 20260820,
    overwrite: bool = False,
) -> DataQualityAblationArtifact:
    require_available_outputs((output_path,), overwrite=overwrite)
    if not noise_fractions or len(set(noise_fractions)) != len(noise_fractions):
        raise ValueError("noise fractions must be non-empty and unique")
    if any(not 0.0 <= value <= 1.0 for value in noise_fractions):
        raise ValueError("noise fractions must be in [0, 1]")
    if 0.0 not in noise_fractions:
        raise ValueError("noise fractions must include a 0.0 clean control")

    manifest = load_manifest(dataset_root)
    training = load_training_examples(dataset_root)
    validation = load_validation_examples(dataset_root)
    test = load_test_examples(dataset_root)
    points: list[DataQualityAblationPoint] = []
    for offset, fraction in enumerate(noise_fractions):
        labels, plan = corrupted_labels(training, fraction=fraction, seed=seed + offset)
        model = TfidfIssueTypeModel(
            model_id=f"tfidf-label-noise-{fraction:.2f}",
            random_state=seed,
        )
        started = time.perf_counter()
        model.fit([example.ticket for example in training], labels)
        fit_seconds = time.perf_counter() - started
        validation_result = measure_split(model, validation)
        test_result = measure_split(model, test)
        points.append(
            DataQualityAblationPoint(
                label_noise_fraction=fraction,
                corrupted_label_count=len(plan),
                corruption_plan_sha256=_plan_sha256(plan),
                fit_seconds=fit_seconds,
                model_version=model.model_version,
                validation=_split_result(validation_result),
                test=_split_result(test_result),
                validation_errors=_issue_type_error_analysis(model, validation),
                test_errors=_issue_type_error_analysis(model, test),
            )
        )

    clean_point = next(item for item in points if item.label_noise_fraction == 0.0)
    clean = clean_point.test.metrics.macro_f1
    noisiest = max(points, key=lambda item: item.label_noise_fraction).test.metrics.macro_f1
    if clean is None or noisiest is None:
        conclusion = "The held-out macro-F1 comparison was unavailable."
    elif noisiest < clean:
        conclusion = (
            f"Test macro F1 fell from {clean:.4f} to {noisiest:.4f} at the largest "
            "label-noise level; label quality mattered in this controlled stress test."
        )
    else:
        conclusion = (
            f"Test macro F1 remained {clean:.4f} through the largest label-noise level; "
            "the template benchmark is too separable for this ablation to expose sensitivity."
        )

    split_by_name = {item.split.value: item for item in manifest.splits}
    artifact = DataQualityAblationArtifact(
        completed_at=datetime.now(UTC),
        hypothesis="Held-out issue classification should degrade as training labels become less reliable.",
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        train_file_sha256=split_by_name["train"].sha256,
        validation_file_sha256=split_by_name["validation"].sha256,
        test_file_sha256=split_by_name["test"].sha256,
        test_lock_sha256=sha256_file(dataset_root / manifest.test_lock_file),
        seed=seed,
        method=(
            "For each declared fraction, choose an exact seeded subset of training rows and rotate "
            "its issue_type to the next allowed class. Validation and locked test labels are unchanged. "
            "Error examples are sorted by case id and capped at ten per split."
        ),
        points=tuple(points),
        conclusion=conclusion,
        limitations=(
            "Injected label noise is a controlled stressor, not a model of every real annotation error.",
            "The synthetic template benchmark contains unusually strong lexical class cues.",
            "The classical model predicts issue_type only; this ablation does not measure full triage quality.",
            "Representative failures contain case ids and issue labels only, never ticket text.",
        ),
    )
    write_json_artifact(output_path, artifact)
    return artifact


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "data" / "iam_triage_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "results" / "classical_label_quality_ablation_v1.json",
    )
    parser.add_argument("--noise-fractions", type=float, nargs="+", default=(0.0, 0.1, 0.3))
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_data_quality_ablation(
        dataset_root=args.dataset_root,
        output_path=args.output,
        noise_fractions=tuple(args.noise_fractions),
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
