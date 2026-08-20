"""Train and measure the required TF-IDF + logistic-regression baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from modelforge.datasets import (
    atomic_write_bytes,
    load_manifest,
    load_test_examples,
    load_training_examples,
    load_validation_examples,
    sha256_file,
)
from modelforge.evaluation.metrics import classification_metrics
from modelforge.evaluation.models import ClassificationMetrics
from modelforge.experiments._artifacts import require_available_outputs
from modelforge.models.classical import TfidfIssueTypeModel
from modelforge.schemas import IssueType, LabeledExample
from modelforge.schemas._base import StrictBaseModel


class SplitResult(StrictBaseModel):
    count: int = Field(ge=0)
    metrics: ClassificationMetrics
    average_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    estimated_cost_usd_per_request: float = 0.0


class ClassicalExperimentArtifact(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_version: Literal["1.0"] = "1.0"
    result_status: Literal["measured"] = "measured"
    experiment_id: str
    completed_at: datetime
    hypothesis: str
    dataset_name: str
    dataset_version: str
    train_file_sha256: str
    validation_file_sha256: str
    test_file_sha256: str
    test_lock_sha256: str
    train_count: int
    fit_seconds: float
    peak_rss_raw: int
    peak_rss_platform_note: str
    model_id: str
    model_version: str
    baseline_config: dict[str, int | str]
    model_artifact_sha256: str | None
    validation: SplitResult
    test: SplitResult
    environment: dict[str, str]
    limitations: tuple[str, ...]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def measure_split(
    model: TfidfIssueTypeModel,
    examples: list[LabeledExample],
) -> SplitResult:
    observations: list[tuple[str, str, float]] = []
    latencies: list[float] = []
    for example in examples:
        started = time.perf_counter()
        prediction = model.predict_issue_type(example.ticket)
        latencies.append((time.perf_counter() - started) * 1_000)
        observations.append(
            (example.expected.issue_type.value, prediction.issue_type.value, 1.0)
        )
    return SplitResult(
        count=len(examples),
        metrics=classification_metrics(
            observations,
            labels=tuple(label.value for label in IssueType),
        ),
        average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
    )


def run_experiment(
    *,
    dataset_root: Path,
    output_path: Path,
    model_path: Path | None = None,
    overwrite: bool = False,
) -> ClassicalExperimentArtifact:
    destinations = [output_path]
    if model_path is not None:
        destinations.append(model_path)
    require_available_outputs(destinations, overwrite=overwrite)
    manifest = load_manifest(dataset_root)
    training = load_training_examples(dataset_root)
    validation = load_validation_examples(dataset_root)
    # The lock is checked inside load_test_examples before final measurement.
    test = load_test_examples(dataset_root)

    model = TfidfIssueTypeModel(
        model_id="tfidf-logreg-issue-type-v1",
        random_state=20260820,
        max_features=20_000,
        max_iter=1_000,
    )
    started = time.perf_counter()
    model.fit(
        [example.ticket for example in training],
        [example.expected.issue_type for example in training],
    )
    fit_seconds = time.perf_counter() - started
    artifact_sha: str | None = None
    if model_path is not None:
        model.save(model_path)
        artifact_sha = sha256_file(model_path)

    validation_result = measure_split(model, validation)
    test_result = measure_split(model, test)
    split_by_name = {item.split.value: item for item in manifest.splits}
    result = ClassicalExperimentArtifact(
        experiment_id="classical_tfidf_logreg_v1",
        completed_at=datetime.now(UTC),
        hypothesis="A sparse lexical baseline establishes the minimum complexity IAM triage needs.",
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        train_file_sha256=split_by_name["train"].sha256,
        validation_file_sha256=split_by_name["validation"].sha256,
        test_file_sha256=split_by_name["test"].sha256,
        test_lock_sha256=sha256_file(dataset_root / manifest.test_lock_file),
        train_count=len(training),
        fit_seconds=fit_seconds,
        peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_platform_note=(
            "ru_maxrss is bytes on macOS and KiB on Linux; it is process peak RSS, not model-only memory"
        ),
        model_id=model.model_id,
        model_version=model.model_version,
        baseline_config={
            "random_state": model.random_state,
            "max_features": model.max_features,
            "max_iter": model.max_iter,
            "features": "word_unigram_bigram_tfidf",
            "classifier": "balanced_logistic_regression",
        },
        model_artifact_sha256=artifact_sha,
        validation=validation_result,
        test=test_result,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "numpy": importlib.metadata.version("numpy"),
        },
        limitations=(
            "The benchmark is template-generated and much easier than real enterprise tickets.",
            "The required classical baseline predicts issue_type only; non-issue fields are not scored here.",
            "Single-request latency is measured without concurrent load or model deserialization.",
        ),
    )
    atomic_write_bytes(
        output_path,
        (result.model_dump_json(indent=2) + "\n").encode(),
    )
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "data" / "iam_triage_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "results" / "classical_tfidf_logreg_v1.json",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=root / "artifacts" / "models" / "classical_tfidf_logreg_v1.joblib",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_experiment(
        dataset_root=args.dataset_root,
        output_path=args.output,
        model_path=args.model_output,
        overwrite=args.overwrite,
    )
    print(
        f"{result.experiment_id}: validation macro F1={result.validation.metrics.macro_f1:.4f}; "
        f"locked-test macro F1={result.test.metrics.macro_f1:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
