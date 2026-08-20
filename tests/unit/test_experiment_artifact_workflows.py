from __future__ import annotations

import asyncio
import hashlib
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelforge.evaluation import MetricFloor, RegressionPolicy, write_json_artifact
from modelforge.experiments.run_evaluation import (
    ModelEvaluationArtifact,
    run_model_evaluation,
)
from modelforge.experiments.run_regression_gate import run_regression_gate
from modelforge.experiments.run_routing_sweep import run_routing_sweep
from modelforge.models import HuggingFaceModelConfig, ModelPrediction
from modelforge.schemas import IssueType, TriageResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "iam_triage_v1"


class _MeasuredFixtureModel:
    """Deterministic unit fixture; it is not production model evidence."""

    def __init__(
        self,
        *,
        model_id: str,
        frontier: bool,
        cost_usd: float,
    ) -> None:
        from modelforge.datasets import load_validation_examples

        examples = load_validation_examples(DATASET_ROOT)
        self.model_id = model_id
        self.role = "frontier" if frontier else "small"
        self._frontier = frontier
        self._cost_usd = cost_usd
        self._index = {
            example.ticket.ticket_id: (index, example.expected)
            for index, example in enumerate(examples)
        }
        if frontier:
            self.provider = SimpleNamespace(
                config=SimpleNamespace(
                    provider_id="unit-fixture-provider",
                    base_url="https://fixture.invalid/v1",
                    model="unit-fixture-frontier",
                    timeout_seconds=1.0,
                    input_usd_per_million=1.0,
                    output_usd_per_million=2.0,
                    pricing_source="https://fixture.invalid/pricing",
                    pricing_as_of=date(2026, 1, 1),
                    json_mode=True,
                )
            )
        else:
            self.config = HuggingFaceModelConfig(
                model_name_or_path="unit/fixture-student",
                revision="1" * 40,
                serving_id=model_id,
            )

    async def triage(self, ticket, *, timeout_s=None):
        del timeout_s
        index, expected = self._index[ticket.ticket_id]
        result = TriageResult(
            **expected.model_dump(),
            confidence=0.62 + (index % 7) * 0.05,
        )
        if not self._frontier and index == 0:
            result = result.model_copy(update={"issue_type": IssueType.OTHER_IAM})
        return ModelPrediction(
            result=result,
            model_id=self.model_id,
            model_version=f"{self.model_id}-version",
            role=self.role,
            latency_ms=1.0 if not self._frontier else 4.0,
            input_tokens=20,
            output_tokens=10,
            estimated_cost_usd=self._cost_usd,
            metadata={"fixture": True},
        )


@pytest.fixture(scope="module")
def measured_fixture_sources(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, ModelEvaluationArtifact, ModelEvaluationArtifact]:
    directory = tmp_path_factory.mktemp("measured-fixture-sources")
    student_path = directory / "unit-fixture-student.json"
    frontier_path = directory / "unit-fixture-frontier.json"
    student = asyncio.run(
        run_model_evaluation(
            model=_MeasuredFixtureModel(
                model_id="unit-fixture-student",
                frontier=False,
                cost_usd=0.001,
            ),
            backend="hf-base",
            dataset_root=DATASET_ROOT,
            split="validation",
            output_path=student_path,
            experiment_id="unit-fixture-student-measurement",
        )
    )
    frontier = asyncio.run(
        run_model_evaluation(
            model=_MeasuredFixtureModel(
                model_id="unit-fixture-frontier",
                frontier=True,
                cost_usd=0.02,
            ),
            backend="frontier",
            dataset_root=DATASET_ROOT,
            split="validation",
            output_path=frontier_path,
            experiment_id="unit-fixture-frontier-measurement",
        )
    )
    return student_path, frontier_path, student, frontier


def test_routing_sweep_uses_comparable_cached_measurements_and_safe_svg(
    tmp_path: Path,
    measured_fixture_sources: tuple[
        Path,
        Path,
        ModelEvaluationArtifact,
        ModelEvaluationArtifact,
    ],
) -> None:
    student_path, frontier_path, _, _ = measured_fixture_sources
    output = tmp_path / "unit-fixture-routing.json"
    graph = tmp_path / "unit-fixture-routing.svg"

    result = run_routing_sweep(
        student_artifact_path=student_path,
        frontier_artifact_path=frontier_path,
        dataset_root=DATASET_ROOT,
        output_path=output,
        graph_path=graph,
        experiment_id="unit-fixture-routing-sweep",
    )

    assert result.result_status == "derived_from_measured_sources"
    assert result.thresholds == (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
    assert len(result.confidence_assessor.successful_student_cases) == 100
    assert result.confidence_assessor.is_probability is False
    assert all(
        case.is_probability is False
        for case in result.confidence_assessor.successful_student_cases
    )
    assert result.comparability.exact_case_population_verified is True
    assert len(result.quality_cost_points) == 6

    svg = graph.read_text(encoding="utf-8")
    assert "<script" not in svg.casefold()
    assert "javascript:" not in svg.casefold()
    assert "href=" not in svg.casefold()
    assert hashlib.sha256(svg.encode("utf-8")).hexdigest() == result.graph_sha256

    from modelforge.datasets import load_validation_examples

    first = load_validation_examples(DATASET_ROOT)[0]
    derived_json = output.read_text(encoding="utf-8")
    assert first.ticket.subject not in derived_json
    assert first.ticket.body not in derived_json

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_routing_sweep(
            student_artifact_path=student_path,
            frontier_artifact_path=frontier_path,
            dataset_root=DATASET_ROOT,
            output_path=output,
            graph_path=graph,
            experiment_id="unit-fixture-routing-sweep",
        )


def test_routing_sweep_rejects_incomparable_source_identity(
    tmp_path: Path,
    measured_fixture_sources: tuple[
        Path,
        Path,
        ModelEvaluationArtifact,
        ModelEvaluationArtifact,
    ],
) -> None:
    student_path, _, _, frontier = measured_fixture_sources
    incompatible_path = tmp_path / "unit-fixture-incompatible-frontier.json"
    write_json_artifact(
        incompatible_path,
        frontier.model_copy(update={"dataset_version": "9.9.9"}),
    )

    with pytest.raises(ValueError, match="incomparable"):
        run_routing_sweep(
            student_artifact_path=student_path,
            frontier_artifact_path=incompatible_path,
            dataset_root=DATASET_ROOT,
            output_path=tmp_path / "unused.json",
            graph_path=tmp_path / "unused.svg",
            experiment_id="unit-fixture-incomparable",
        )


def test_regression_gate_cli_artifact_is_derived_and_immutable(
    tmp_path: Path,
    measured_fixture_sources: tuple[
        Path,
        Path,
        ModelEvaluationArtifact,
        ModelEvaluationArtifact,
    ],
) -> None:
    student_path, frontier_path, _, _ = measured_fixture_sources
    policy_path = tmp_path / "unit-fixture-policy.json"
    output = tmp_path / "unit-fixture-regression.json"
    write_json_artifact(
        policy_path,
        RegressionPolicy(
            policy_name="unit-fixture-policy",
            policy_version="1.0.0",
            overall=MetricFloor(
                max_error_rate=0.0,
                min_schema_validity_rate=1.0,
                min_exact_match_rate=0.0,
                min_issue_macro_f1=0.0,
            ),
            max_exact_match_drop=1.0,
            max_issue_macro_f1_drop=1.0,
            max_per_class_f1_drop=1.0,
        ),
    )

    result = run_regression_gate(
        baseline_path=student_path,
        candidate_path=frontier_path,
        policy_path=policy_path,
        output_path=output,
    )

    assert result.result_status == "derived_from_measured_sources"
    assert result.gate.passed is True
    assert result.gate.exit_code == 0
    assert result.privacy["contains_raw_ticket_text"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_regression_gate(
            baseline_path=student_path,
            candidate_path=frontier_path,
            policy_path=policy_path,
            output_path=output,
        )
