from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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
from modelforge.training.manifest import (
    ExperimentManifest,
    GitMetadata,
    HardwareMetadata,
    LossHistory,
    OverfittingSignal,
    ParameterCounts,
    sha256_file,
    write_manifest_once,
)

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


class _AdapterFixtureModel(HeuristicTriageModel):
    role = "small"
    model_id = "adapter-fixture-v1"

    def __init__(self, adapter_path: Path) -> None:
        self.config = HuggingFaceModelConfig(
            model_name_or_path="unit/fixture-model",
            revision="a" * 40,
            adapter_name_or_path=str(adapter_path),
            serving_id=self.model_id,
        )

    async def triage(self, ticket, *, timeout_s=None):
        prediction = await super().triage(ticket, timeout_s=timeout_s)
        return prediction.model_copy(
            update={
                "model_id": self.model_id,
                "model_version": "adapter-fixture-version",
                "role": self.role,
            }
        )


def _write_adapter_training_manifest(run_directory: Path) -> tuple[Path, Path]:
    adapter_directory = run_directory / "adapter"
    adapter_directory.mkdir(parents=True)
    adapter_file = adapter_directory / "adapter_config.json"
    adapter_file.write_text('{"fixture":true}\n', encoding="utf-8")
    manifest = ExperimentManifest(
        experiment_name="adapter-fixture-run",
        completed_at=datetime(2026, 9, 6, tzinfo=UTC),
        config={"fixture": True},
        config_sha256="b" * 64,
        dataset_sha256={"train": "c" * 64, "validation": "d" * 64},
        prompt_version="fixture-prompt",
        prompt_sha256="e" * 64,
        requested_model_revision="a" * 40,
        resolved_model_revision="a" * 40,
        requested_tokenizer_revision="a" * 40,
        resolved_tokenizer_revision="a" * 40,
        seed=17,
        parameter_counts=ParameterCounts(trainable=10, total=100, trainable_fraction=0.1),
        hardware=HardwareMetadata(
            platform="fixture",
            python_version="3.13",
            device="cpu",
            precision="fp32",
        ),
        git=GitMetadata(commit_sha="f" * 40, dirty=False),
        packages={"torch": "fixture"},
        losses=LossHistory(),
        overfitting=OverfittingSignal(detected=False, reason="fixture"),
        trainer_metrics={},
        artifacts_sha256={"adapter/adapter_config.json": sha256_file(adapter_file)},
        truncated_training_examples=0,
        truncated_validation_examples=0,
    )
    manifest_path = run_directory / "manifest.json"
    write_manifest_once(manifest_path, manifest)
    return manifest_path, adapter_directory


def test_adapter_evaluation_binds_to_the_verified_training_manifest(tmp_path: Path) -> None:
    manifest_path, adapter_directory = _write_adapter_training_manifest(tmp_path / "run")
    output = tmp_path / "adapter-evaluation.json"

    result = asyncio.run(
        run_model_evaluation(
            model=_AdapterFixtureModel(adapter_directory),
            backend="hf-adapter",
            dataset_root=DATASET_ROOT,
            split="validation",
            output_path=output,
            experiment_id="adapter-fixture-evaluation",
            training_manifest_path=manifest_path,
            limit=1,
        )
    )

    assert result.training_run is not None
    assert result.training_run.experiment_name == "adapter-fixture-run"
    assert result.training_run.adapter_artifacts_sha256 == {
        "adapter_config.json": sha256_file(adapter_directory / "adapter_config.json")
    }


def test_adapter_evaluation_rejects_a_tampered_adapter(tmp_path: Path) -> None:
    manifest_path, adapter_directory = _write_adapter_training_manifest(tmp_path / "run")
    (adapter_directory / "adapter_config.json").write_text('{"fixture":false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="adapter files do not match"):
        asyncio.run(
            run_model_evaluation(
                model=_AdapterFixtureModel(adapter_directory),
                backend="hf-adapter",
                dataset_root=DATASET_ROOT,
                split="validation",
                output_path=tmp_path / "must-not-exist.json",
                experiment_id="tampered-adapter-fixture",
                training_manifest_path=manifest_path,
                limit=1,
            )
        )


def test_measured_huggingface_evaluation_rejects_mutable_revision(tmp_path: Path) -> None:
    class MutableRevisionFixture:
        model_id = "mutable-revision-fixture"
        role = "small"
        config = HuggingFaceModelConfig(
            model_name_or_path="unit/fixture-model",
            revision="main",
        )

        async def triage(self, ticket, *, timeout_s=None):
            del ticket, timeout_s
            raise AssertionError("mutable model must be rejected before inference")

    with pytest.raises(ValueError, match="immutable commit revision"):
        asyncio.run(
            run_model_evaluation(
                model=MutableRevisionFixture(),
                backend="hf-base",
                dataset_root=DATASET_ROOT,
                split="validation",
                output_path=tmp_path / "must-not-exist.json",
                experiment_id="mutable-revision-fixture",
                limit=1,
            )
        )
