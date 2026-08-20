"""Local bounded evaluation executor used by the modular-monolith API."""

from __future__ import annotations

import time
from pathlib import Path

from modelforge.api.contracts import EvaluationRunRequest
from modelforge.api.registry import ModelRegistry
from modelforge.datasets import (
    load_adversarial_examples,
    load_test_examples,
    load_validation_examples,
)
from modelforge.evaluation.metrics import evaluate_records, evaluation_record_from_domain
from modelforge.evaluation.models import EvaluationConfig, EvaluationRecord, PredictionError
from modelforge.models.base import ModelAdapterError, ModelOutputError
from modelforge.schemas import IssueType, TriageResult


class LocalEvaluationExecutor:
    """Evaluate one allowlisted model sequentially on an immutable split.

    Sequential case execution is deliberate for v1: local Transformer runtimes
    are not assumed thread-safe, and latency measurements need a declared load
    profile. Multiple evaluation jobs are bounded separately by the API.
    """

    def __init__(self, *, registry: ModelRegistry, dataset_root: Path) -> None:
        self.registry = registry
        self.dataset_root = dataset_root

    async def run(self, request: EvaluationRunRequest) -> dict:
        model = self.registry.require(request.model_id)
        examples = self._load(request.split)
        if request.limit is not None:
            examples = examples[: request.limit]
        records = []
        for example in examples:
            started = time.perf_counter()
            try:
                prediction = await model.triage(example.ticket)
                records.append(
                    evaluation_record_from_domain(example=example, prediction=prediction)
                )
            except ModelOutputError as exc:
                if exc.raw_output is None:
                    records.append(
                        evaluation_record_from_domain(
                            example=example,
                            error=PredictionError(
                                kind=exc.code,
                                message="model output failed validation",
                                retriable=False,
                            ),
                        )
                    )
                else:
                    records.append(
                        EvaluationRecord(
                            case_id=example.example_id,
                            expected=example.expected.model_dump(mode="json"),
                            raw_prediction=exc.raw_output,
                            slice=(
                                example.robustness_category.value
                                if example.robustness_category is not None
                                else example.split.value
                            ),
                            input_text=f"{example.ticket.subject}\n{example.ticket.body}",
                            latency_ms=(time.perf_counter() - started) * 1_000,
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
            except ModelAdapterError as exc:
                records.append(
                    evaluation_record_from_domain(
                        example=example,
                        error=PredictionError(
                            kind=exc.code,
                            message="model adapter failed",
                            retriable=exc.retryable,
                        ),
                    )
                )
        report = evaluate_records(
            records,
            schema_model=TriageResult,
            config=EvaluationConfig(issue_labels=tuple(item.value for item in IssueType)),
        )
        return report.model_dump(mode="json")

    def _load(self, split: str):
        if split == "validation":
            return load_validation_examples(self.dataset_root)
        if split == "test":
            return load_test_examples(self.dataset_root)
        if split == "adversarial":
            return load_adversarial_examples(self.dataset_root)
        raise ValueError(f"unsupported evaluation split: {split}")
