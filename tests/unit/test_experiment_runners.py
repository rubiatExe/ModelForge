from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from modelforge.experiments.run_data_ablation import corrupted_labels
from modelforge.experiments.run_evaluation import run_model_evaluation
from modelforge.models import (
    HeuristicTriageModel,
    HuggingFaceModelConfig,
    ModelOutputError,
)
from modelforge.schemas import IssueType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "iam_triage_v1"


def test_corruption_plan_is_exact_deterministic_and_changes_labels() -> None:
    from modelforge.datasets import load_training_examples

    examples = load_training_examples(DATASET_ROOT)
    left_labels, left_plan = corrupted_labels(examples, fraction=0.1, seed=7)
    right_labels, right_plan = corrupted_labels(examples, fraction=0.1, seed=7)

    assert left_labels == right_labels
    assert left_plan == right_plan
    assert len(left_plan) == 50
    assert all(original != replacement for _, original, replacement in left_plan)
    assert all(isinstance(label, IssueType) for label in left_labels)


def test_evaluation_runner_labels_heuristic_as_partial_fixture(
    tmp_path: Path,
) -> None:
    output = tmp_path / "heuristic.json"
    result = asyncio.run(
        run_model_evaluation(
            model=HeuristicTriageModel(),
            backend="heuristic",
            dataset_root=DATASET_ROOT,
            split="validation",
            output_path=output,
            experiment_id="heuristic-smoke",
            limit=3,
        )
    )

    assert output.is_file()
    assert result.result_status == "fixture"
    assert result.partial_run is True
    assert result.evaluated_count == 3
    assert result.operations.attempted_cases == 3


def test_evaluation_runner_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        asyncio.run(
            run_model_evaluation(
                model=HeuristicTriageModel(),
                backend="heuristic",
                dataset_root=DATASET_ROOT,
                split="validation",
                output_path=output,
                experiment_id="no-overwrite",
            )
        )


def test_evaluation_runner_requires_explicit_locked_test_confirmation(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="explicit confirmation"):
        asyncio.run(
            run_model_evaluation(
                model=HeuristicTriageModel(),
                backend="heuristic",
                dataset_root=DATASET_ROOT,
                split="test",
                output_path=tmp_path / "must-not-exist.json",
                experiment_id="unit-fixture-accidental-test",
                limit=1,
            )
        )


class _InvalidOutputFixtureModel:
    model_id = "unit-fixture-invalid-output"
    role = "small"
    config = HuggingFaceModelConfig(
        model_name_or_path="unit-fixture-model",
        revision="0" * 40,
    )

    async def triage(self, ticket, *, timeout_s=None):
        del ticket, timeout_s
        raise ModelOutputError(
            "fixture invalid output",
            raw_output='{"private":"must-not-survive"}',
            input_tokens=11,
            output_tokens=4,
            estimated_cost_usd=0.003,
            model_version="unit-fixture-version",
        )


def test_evaluation_runner_accounts_for_and_redacts_invalid_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "invalid-output-fixture.json"
    result = asyncio.run(
        run_model_evaluation(
            model=_InvalidOutputFixtureModel(),
            backend="hf-base",
            dataset_root=DATASET_ROOT,
            split="validation",
            output_path=output,
            experiment_id="unit-fixture-invalid-output",
            limit=1,
        )
    )

    assert result.result_status == "measured"
    assert result.operations.schema_invalid_outputs == 1
    assert result.operations.total_input_tokens == 11
    assert result.operations.total_output_tokens == 4
    assert result.operations.total_estimated_cost_usd == pytest.approx(0.003)
    assert result.operations.cost_accounted_cases == 1
    assert result.operations.missing_cost_accounting_cases == 0
    assert result.observed_model_versions == ("unit-fixture-version",)
    assert result.evaluation.items[0].prediction is None
    assert result.evaluation.items[0].schema_error == "prediction failed strict schema validation"
    assert "must-not-survive" not in output.read_text(encoding="utf-8")
