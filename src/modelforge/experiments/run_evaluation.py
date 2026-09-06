"""Run one model on an isolated ModelForge split and persist measured evidence."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import math
import platform
import re
import resource
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from modelforge.datasets import (
    load_adversarial_examples,
    load_manifest,
    load_test_examples,
    load_validation_examples,
    sha256_file,
)
from modelforge.evaluation import (
    EvaluationConfig,
    EvaluationRecord,
    EvaluationReport,
    FailureTaxonomyReport,
    PredictionError,
    build_failure_taxonomy,
    calibration_from_evaluation,
    evaluate_records,
    evaluation_record_from_domain,
    write_json_artifact,
)
from modelforge.evaluation.calibration import CalibrationReport
from modelforge.evaluation.models import EvaluationStatus
from modelforge.evaluation.regression import case_ids_sha256
from modelforge.experiments._artifacts import (
    canonical_sha256,
    require_available_outputs,
)
from modelforge.models import (
    FrontierTriageModel,
    HeuristicTriageModel,
    HuggingFaceModelConfig,
    HuggingFaceTriageModel,
    ModelAdapterError,
    ModelOutputError,
    OpenAICompatibleFrontierProvider,
    TriageModel,
)
from modelforge.schemas import IssueType, LabeledExample, TriageResult
from modelforge.schemas._base import StrictBaseModel
from modelforge.training.manifest import ExperimentManifest, artifact_hashes

BackendName = Literal["heuristic", "hf-base", "hf-adapter", "frontier"]
EvaluationSplit = Literal["validation", "test", "adversarial"]
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class OperationalSummary(StrictBaseModel):
    load_profile: str = "sequential_single_process"
    attempted_cases: int = Field(ge=0)
    valid_predictions: int = Field(ge=0)
    schema_invalid_outputs: int = Field(ge=0)
    operational_errors: int = Field(ge=0)
    average_latency_ms: float | None = Field(default=None, ge=0.0)
    p50_latency_ms: float | None = Field(default=None, ge=0.0)
    p95_latency_ms: float | None = Field(default=None, ge=0.0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    missing_token_usage_cases: int = Field(ge=0)
    total_estimated_cost_usd: float = Field(ge=0.0)
    average_estimated_cost_usd_per_accounted_case: float | None = Field(
        default=None,
        ge=0.0,
    )
    cost_accounted_cases: int = Field(ge=0)
    missing_cost_accounting_cases: int = Field(ge=0)
    cost_estimate_is_lower_bound: bool
    peak_rss_raw: int = Field(ge=0)
    peak_rss_platform_note: str

    @model_validator(mode="after")
    def validate_accounting(self) -> OperationalSummary:
        outcome_count = (
            self.valid_predictions
            + self.schema_invalid_outputs
            + self.operational_errors
        )
        if outcome_count != self.attempted_cases:
            raise ValueError("prediction outcome counts must sum to attempted_cases")
        if self.cost_accounted_cases + self.missing_cost_accounting_cases != self.attempted_cases:
            raise ValueError("cost accounting counts must sum to attempted_cases")
        if self.missing_token_usage_cases > self.attempted_cases:
            raise ValueError("missing_token_usage_cases cannot exceed attempted_cases")
        if self.cost_estimate_is_lower_bound != (self.missing_cost_accounting_cases > 0):
            raise ValueError("cost_estimate_is_lower_bound must reflect missing cost accounting")
        return self


class TrainingRunProvenance(StrictBaseModel):
    """Cryptographic chain of custody from a local LoRA adapter to its training run."""

    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_name: str = Field(min_length=1)
    resolved_model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    resolved_tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_artifacts_sha256: dict[str, str] = Field(min_length=1)


class ModelEvaluationArtifact(StrictBaseModel):
    artifact_version: Literal["1.0", "1.1"] = "1.1"
    result_status: Literal["measured", "fixture"] = "measured"
    experiment_id: str = Field(min_length=1)
    completed_at: datetime
    backend: BackendName
    model_id: str = Field(min_length=1)
    observed_model_versions: tuple[str, ...]
    model_configuration: dict[str, Any]
    training_run: TrainingRunProvenance | None = None
    prompt_version: str | None = None
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dataset_name: str
    dataset_version: str
    split: EvaluationSplit
    split_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    full_split_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    partial_run: bool
    case_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation: EvaluationReport
    failures: FailureTaxonomyReport
    confidence_diagnostic: CalibrationReport
    operations: OperationalSummary
    environment: dict[str, str]
    privacy: dict[str, str | bool]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_artifact_consistency(self) -> ModelEvaluationArtifact:
        if (self.backend == "heuristic") != (self.result_status == "fixture"):
            raise ValueError("heuristic artifacts must be fixtures and model artifacts must be measured")
        if self.backend == "frontier":
            for field in ("pricing_source", "pricing_as_of"):
                value = self.model_configuration.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"measured frontier artifact requires {field}")
        if self.backend == "hf-adapter" and self.training_run is None:
            raise ValueError("adapter evaluation artifacts require verified training-run provenance")
        if self.backend != "hf-adapter" and self.training_run is not None:
            raise ValueError("training-run provenance is only valid for adapter evaluations")
        if self.evaluated_count > self.full_split_count:
            raise ValueError("evaluated_count cannot exceed full_split_count")
        if self.partial_run != (self.evaluated_count != self.full_split_count):
            raise ValueError("partial_run does not match evaluated and full split counts")
        if self.evaluated_count != len(self.evaluation.items):
            raise ValueError("evaluated_count does not match evaluation items")
        if self.evaluation.aggregate.total_cases != self.evaluated_count:
            raise ValueError("evaluation aggregate does not cover evaluated_count")
        if self.case_ids_sha256 != case_ids_sha256(self.evaluation):
            raise ValueError("case_ids_sha256 does not match evaluation items")
        if self.failures.evaluated_cases != self.evaluated_count:
            raise ValueError("failure taxonomy does not cover evaluated_count")
        if self.operations.attempted_cases != self.evaluated_count:
            raise ValueError("operational accounting does not cover evaluated_count")
        if (self.split == "test") != (self.test_lock_sha256 is not None):
            raise ValueError("test_lock_sha256 is required only for locked-test artifacts")
        return self


class EvaluationExecution:
    """In-memory details used to assemble one immutable artifact."""

    def __init__(self) -> None:
        self.records: list[EvaluationRecord] = []
        self.latencies_ms: list[float] = []
        self.model_versions: set[str] = set()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.missing_token_usage_cases = 0
        self.total_cost_usd = 0.0
        self.cost_accounted_cases = 0
        self.missing_cost_accounting_cases = 0


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _load_examples(dataset_root: Path, split: EvaluationSplit) -> list[LabeledExample]:
    if split == "validation":
        return load_validation_examples(dataset_root)
    if split == "test":
        return load_test_examples(dataset_root)
    if split == "adversarial":
        return load_adversarial_examples(dataset_root)
    raise ValueError(f"unsupported evaluation split: {split}")


def _case_slice(example: LabeledExample) -> str:
    robustness = getattr(example.robustness_category, "value", None)
    return str(robustness or example.split.value)


async def execute_model(
    model: TriageModel,
    examples: Sequence[LabeledExample],
    *,
    timeout_s: float,
    require_token_usage_for_cost: bool = False,
) -> EvaluationExecution:
    """Evaluate sequentially; native model runtimes are not assumed thread-safe."""

    execution = EvaluationExecution()
    for example in examples:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                prediction = await model.triage(example.ticket, timeout_s=timeout_s)
        except ModelOutputError as exc:
            latency_ms = (time.perf_counter() - started) * 1_000
            execution.latencies_ms.append(latency_ms)
            execution.total_cost_usd += exc.estimated_cost_usd
            if exc.model_version is not None:
                execution.model_versions.add(exc.model_version)
            if exc.input_tokens is None or exc.output_tokens is None:
                execution.missing_token_usage_cases += 1
                if require_token_usage_for_cost:
                    execution.missing_cost_accounting_cases += 1
                else:
                    execution.cost_accounted_cases += 1
            else:
                execution.total_input_tokens += exc.input_tokens
                execution.total_output_tokens += exc.output_tokens
                execution.cost_accounted_cases += 1
            if exc.raw_output is None:
                execution.records.append(
                    EvaluationRecord(
                        case_id=example.example_id,
                        expected=example.expected.model_dump(mode="json"),
                        error=PredictionError(
                            kind=exc.code,
                            message="model output failed validation",
                            retriable=False,
                        ),
                        slice=_case_slice(example),
                        latency_ms=latency_ms,
                        cost_usd=exc.estimated_cost_usd,
                    )
                )
            else:
                execution.records.append(
                    EvaluationRecord(
                        case_id=example.example_id,
                        expected=example.expected.model_dump(mode="json"),
                        raw_prediction=exc.raw_output,
                        slice=_case_slice(example),
                        input_text=f"{example.ticket.subject}\n{example.ticket.body}",
                        latency_ms=latency_ms,
                        cost_usd=exc.estimated_cost_usd,
                        metadata={
                            "robustness_category": getattr(
                                example.robustness_category,
                                "value",
                                None,
                            )
                        },
                    )
                )
            continue
        except ModelAdapterError as exc:
            latency_ms = (time.perf_counter() - started) * 1_000
            execution.latencies_ms.append(latency_ms)
            execution.missing_token_usage_cases += 1
            execution.missing_cost_accounting_cases += 1
            execution.records.append(
                EvaluationRecord(
                    case_id=example.example_id,
                    expected=example.expected.model_dump(mode="json"),
                    error=PredictionError(
                        kind=exc.code,
                        message="model adapter failed",
                        retriable=exc.retryable,
                    ),
                    slice=_case_slice(example),
                    latency_ms=latency_ms,
                )
            )
            continue
        except TimeoutError:
            latency_ms = (time.perf_counter() - started) * 1_000
            execution.latencies_ms.append(latency_ms)
            execution.missing_token_usage_cases += 1
            execution.missing_cost_accounting_cases += 1
            execution.records.append(
                EvaluationRecord(
                    case_id=example.example_id,
                    expected=example.expected.model_dump(mode="json"),
                    error=PredictionError(
                        kind="runner_timeout",
                        message="evaluation deadline exceeded",
                        retriable=True,
                    ),
                    slice=_case_slice(example),
                    latency_ms=latency_ms,
                )
            )
            continue

        latency_ms = (time.perf_counter() - started) * 1_000
        record = evaluation_record_from_domain(example=example, prediction=prediction)
        execution.records.append(
            record.model_copy(update={"latency_ms": latency_ms})
        )
        execution.latencies_ms.append(latency_ms)
        execution.model_versions.add(prediction.model_version)
        execution.total_cost_usd += prediction.estimated_cost_usd
        if prediction.input_tokens is None or prediction.output_tokens is None:
            execution.missing_token_usage_cases += 1
            if require_token_usage_for_cost:
                execution.missing_cost_accounting_cases += 1
            else:
                execution.cost_accounted_cases += 1
        else:
            execution.total_input_tokens += prediction.input_tokens
            execution.total_output_tokens += prediction.output_tokens
            execution.cost_accounted_cases += 1
    return execution


def _model_artifact_configuration(model: TriageModel, backend: BackendName) -> dict[str, Any]:
    if backend in {"hf-base", "hf-adapter"}:
        config = model.config
        values = config.model_dump(mode="json")
        for field in ("model_name_or_path", "adapter_name_or_path"):
            reference = values.get(field)
            if isinstance(reference, str) and (
                Path(reference).is_absolute() or Path(reference).exists()
            ):
                values[field] = {
                    "kind": "local_artifact",
                    "basename": Path(reference).name,
                    "path_reference_sha256": canonical_sha256(reference),
                    "content_addressed_here": False,
                }
        return values
    if backend == "frontier":
        provider = model.provider
        config = provider.config
        return {
            "provider_id": config.provider_id,
            "endpoint_configured": True,
            "endpoint_recorded": False,
            "model": config.model,
            "timeout_seconds": config.timeout_seconds,
            "input_usd_per_million": config.input_usd_per_million,
            "output_usd_per_million": config.output_usd_per_million,
            "pricing_source": config.pricing_source,
            "pricing_as_of": (
                config.pricing_as_of.isoformat()
                if config.pricing_as_of is not None
                else None
            ),
            "json_mode": config.json_mode,
            "credential_source": "environment",
        }
    return {"harness_fixture": True, "network": False}


def _require_pinned_huggingface_revision(model: TriageModel) -> HuggingFaceModelConfig:
    config = getattr(model, "config", None)
    if not isinstance(config, HuggingFaceModelConfig):
        raise ValueError("Hugging Face evaluation requires a HuggingFaceModelConfig")
    if not _COMMIT_SHA.fullmatch(config.revision):
        raise ValueError(
            "measured Hugging Face evaluation requires a 40-character immutable commit revision"
        )
    return config


def _verified_training_run_provenance(
    model: TriageModel,
    training_manifest_path: Path,
) -> TrainingRunProvenance:
    """Bind a local adapter evaluation to the exact training manifest and bytes."""

    config = _require_pinned_huggingface_revision(model)
    adapter_reference = config.adapter_name_or_path
    if not adapter_reference:
        raise ValueError("adapter evaluation requires a local adapter_name_or_path")
    manifest_path = Path(training_manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("training manifest must be a regular file")
    try:
        manifest = ExperimentManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"training manifest is invalid: {manifest_path}") from exc
    if config.revision != manifest.resolved_model_revision:
        raise ValueError(
            "adapter evaluation revision does not match the training manifest's resolved model revision"
        )

    expected_adapter = (manifest_path.parent / "adapter").resolve()
    adapter_path = Path(adapter_reference)
    if adapter_path.is_symlink() or not adapter_path.is_dir():
        raise ValueError("adapter evaluation requires a regular local adapter directory")
    if adapter_path.resolve() != expected_adapter:
        raise ValueError("adapter path must be the adapter directory beside its training manifest")
    actual_hashes = artifact_hashes(adapter_path)
    expected_hashes = {
        name.removeprefix("adapter/"): digest
        for name, digest in manifest.artifacts_sha256.items()
        if name.startswith("adapter/")
    }
    if not expected_hashes:
        raise ValueError("training manifest does not contain adapter artifact hashes")
    if actual_hashes != expected_hashes:
        raise ValueError("adapter files do not match the hashes recorded by the training manifest")
    return TrainingRunProvenance(
        training_manifest_sha256=sha256_file(manifest_path),
        experiment_name=manifest.experiment_name,
        resolved_model_revision=manifest.resolved_model_revision,
        resolved_tokenizer_revision=manifest.resolved_tokenizer_revision,
        adapter_artifacts_sha256=actual_hashes,
    )


def _split_manifest(dataset_root: Path, split: EvaluationSplit) -> tuple[Any, Any]:
    manifest = load_manifest(dataset_root)
    try:
        split_manifest = next(item for item in manifest.splits if item.split.value == split)
    except StopIteration as exc:
        raise ValueError(f"dataset manifest does not declare split {split!r}") from exc
    return manifest, split_manifest


def _privacy_safe_report(report: EvaluationReport) -> EvaluationReport:
    """Drop invalid payloads and validation strings that may echo model input."""

    safe_items = tuple(
        item.model_copy(
            update={
                "prediction": None,
                "schema_error": "prediction failed strict schema validation",
            }
        )
        if item.status == EvaluationStatus.SCHEMA_INVALID
        else item
        for item in report.items
    )
    return report.model_copy(update={"items": safe_items})


async def run_model_evaluation(
    *,
    model: TriageModel,
    backend: BackendName,
    dataset_root: Path,
    split: EvaluationSplit,
    output_path: Path,
    experiment_id: str,
    timeout_s: float = 120.0,
    limit: int | None = None,
    confirm_locked_test: bool = False,
    training_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> ModelEvaluationArtifact:
    require_available_outputs((output_path,), overwrite=overwrite)
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if split == "test" and not confirm_locked_test:
        raise ValueError(
            "locked-test evaluation requires explicit confirmation; "
            "use --confirm-locked-test only for a deliberate final measurement"
        )
    if backend in {"hf-base", "hf-adapter"}:
        _require_pinned_huggingface_revision(model)
    if backend == "hf-adapter":
        if training_manifest_path is None:
            raise ValueError("adapter evaluation requires --training-manifest")
        training_run = _verified_training_run_provenance(model, training_manifest_path)
    else:
        if training_manifest_path is not None:
            raise ValueError("--training-manifest is only valid for hf-adapter evaluation")
        training_run = None
    if backend == "frontier":
        provider_config = model.provider.config
        if provider_config.pricing_source is None or provider_config.pricing_as_of is None:
            raise ValueError(
                "measured frontier evaluation requires pricing_source and pricing_as_of provenance"
            )

    manifest, split_manifest = _split_manifest(dataset_root, split)
    full_examples = _load_examples(dataset_root, split)
    examples = full_examples if limit is None else full_examples[:limit]
    execution = await execute_model(
        model,
        examples,
        timeout_s=timeout_s,
        require_token_usage_for_cost=backend == "frontier",
    )
    config = EvaluationConfig(issue_labels=tuple(item.value for item in IssueType))
    report = _privacy_safe_report(
        evaluate_records(
            execution.records,
            schema_model=TriageResult,
            config=config,
        )
    )
    failures = build_failure_taxonomy(report, config=config)
    prompt = getattr(model, "prompt", None)
    schema_invalid = report.aggregate.schema_invalid_count
    errors = report.aggregate.error_count
    valid = report.aggregate.schema_valid_count
    latencies = execution.latencies_ms
    average_cost = (
        execution.total_cost_usd / execution.cost_accounted_cases
        if execution.cost_accounted_cases
        else None
    )
    artifact = ModelEvaluationArtifact(
        result_status="fixture" if backend == "heuristic" else "measured",
        experiment_id=experiment_id,
        completed_at=datetime.now(UTC),
        backend=backend,
        model_id=model.model_id,
        observed_model_versions=tuple(sorted(execution.model_versions)),
        model_configuration=_model_artifact_configuration(model, backend),
        training_run=training_run,
        prompt_version=getattr(prompt, "version", None),
        prompt_sha256=getattr(prompt, "sha256", None),
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        split=split,
        split_file_sha256=split_manifest.sha256,
        test_lock_sha256=(
            sha256_file(dataset_root / manifest.test_lock_file) if split == "test" else None
        ),
        full_split_count=len(full_examples),
        evaluated_count=len(examples),
        partial_run=len(examples) != len(full_examples),
        case_ids_sha256=case_ids_sha256(report),
        result_schema_sha256=canonical_sha256(TriageResult.model_json_schema()),
        evaluation=report,
        failures=failures,
        confidence_diagnostic=calibration_from_evaluation(report),
        operations=OperationalSummary(
            attempted_cases=len(examples),
            valid_predictions=valid,
            schema_invalid_outputs=schema_invalid,
            operational_errors=errors,
            average_latency_ms=(sum(latencies) / len(latencies) if latencies else None),
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            total_input_tokens=execution.total_input_tokens,
            total_output_tokens=execution.total_output_tokens,
            missing_token_usage_cases=execution.missing_token_usage_cases,
            total_estimated_cost_usd=execution.total_cost_usd,
            average_estimated_cost_usd_per_accounted_case=average_cost,
            cost_accounted_cases=execution.cost_accounted_cases,
            missing_cost_accounting_cases=execution.missing_cost_accounting_cases,
            cost_estimate_is_lower_bound=execution.missing_cost_accounting_cases > 0,
            peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_platform_note=(
                "ru_maxrss is bytes on macOS and KiB on Linux; it is process peak RSS, not model-only memory"
            ),
        ),
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pydantic": importlib.metadata.version("pydantic"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
        },
        privacy={
            "classification": "sensitive_evaluation_artifact",
            "contains_raw_ticket_text": False,
            "contains_model_raw_output": False,
            "contains_gold_and_predicted_evidence_snippets": True,
            "schema_validation_details_redacted": True,
        },
        limitations=(
            "The dataset is template-generated and remains pending human review.",
            "Latency is sequential, single-process, warm-and-cold combined; it is not a load test.",
            "The confidence diagnostic uses model self-report, not a calibrated probability.",
            "Average cost uses only cases with explicit accounting and is a lower bound when accounting is missing.",
            "Gold and predicted evidence snippets remain sensitive and require access-controlled storage.",
            "Memory is process peak RSS rather than model-only allocation.",
        ),
    )
    write_json_artifact(output_path, artifact)
    return artifact


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _build_model(args: argparse.Namespace) -> tuple[TriageModel, BackendName]:
    backend: BackendName = args.backend
    if backend == "heuristic":
        return HeuristicTriageModel(), backend
    if backend == "frontier":
        provider = OpenAICompatibleFrontierProvider()
        return FrontierTriageModel(
            provider,
            model_id=args.serving_id or "frontier-iam-triage-v1",
            default_timeout_s=args.timeout,
        ), backend
    if not args.model_name_or_path or not args.revision:
        raise ValueError("Hugging Face evaluation requires --model-name-or-path and --revision")
    adapter = args.adapter_name_or_path if backend == "hf-adapter" else None
    if backend == "hf-adapter" and not adapter:
        raise ValueError("hf-adapter evaluation requires --adapter-name-or-path")
    config = HuggingFaceModelConfig(
        model_name_or_path=args.model_name_or_path,
        revision=args.revision,
        adapter_name_or_path=adapter,
        adapter_revision=args.adapter_revision,
        serving_id=args.serving_id or backend,
        device=args.device,
        precision=args.precision,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )
    return HuggingFaceTriageModel(config, default_timeout_s=args.timeout), backend


async def _close_model(model: TriageModel) -> None:
    backend = getattr(model, "backend", None)
    close = getattr(backend, "close", None)
    if callable(close):
        close()
    provider = getattr(model, "provider", None)
    aclose = getattr(provider, "aclose", None)
    if callable(aclose):
        await aclose()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


async def _async_main(args: argparse.Namespace) -> ModelEvaluationArtifact:
    model, backend = _build_model(args)
    try:
        return await run_model_evaluation(
            model=model,
            backend=backend,
            dataset_root=args.dataset_root,
            split=args.split,
            output_path=args.output,
            experiment_id=args.experiment_id,
            timeout_s=args.timeout,
            limit=args.limit,
            confirm_locked_test=args.confirm_locked_test,
            training_manifest_path=args.training_manifest,
            overwrite=args.overwrite,
        )
    finally:
        await _close_model(model)


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("heuristic", "hf-base", "hf-adapter", "frontier"),
        required=True,
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--split", choices=("validation", "test", "adversarial"), default="validation")
    parser.add_argument("--dataset-root", type=Path, default=root / "data" / "iam_triage_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--revision")
    parser.add_argument("--adapter-name-or-path")
    parser.add_argument("--adapter-revision")
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help="Required for hf-adapter: manifest from the exact local training run.",
    )
    parser.add_argument("--serving-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=2_048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Confirm this is a deliberate final measurement on the locked test split.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(_async_main(args))
    metrics = result.evaluation.aggregate
    print(
        f"{result.experiment_id}: {result.evaluated_count} cases; "
        f"schema={metrics.schema_validity_rate}; issue_macro_f1={metrics.issue.macro_f1}; "
        f"errors={metrics.error_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
