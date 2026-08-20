from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from modelforge.models.base import ModelConfigurationError
from modelforge.models.prompting import load_iam_prompt
from modelforge.schemas import (
    AffectedScope,
    AnnotationStatus,
    DatasetSource,
    DatasetSplit,
    ExampleProvenance,
    IssueType,
    LabeledExample,
    RecommendedAction,
    ReviewStatus,
    RoutingTeam,
    Severity,
    TicketInput,
    TriageLabel,
)
from modelforge.training.config import load_training_config
from modelforge.training.manifest import (
    ExperimentManifest,
    GitMetadata,
    HardwareMetadata,
    LossHistory,
    MetricPoint,
    OverfittingSignal,
    ParameterCounts,
    detect_overfitting,
    sha256_file,
    write_manifest_once,
)
from modelforge.training.runtime import choose_precision
from modelforge.training.tokenization import (
    IGNORE_INDEX,
    TrainingTokenizationError,
    pad_response_only_batch,
    tokenize_response_only,
)
from modelforge.training.train_lora import supervised_messages


class _CharacterTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.tokenization_calls = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.tokenization_calls += 1
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["truncation"] is False
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_single_sequence_response_masking_calls_tokenizer_once() -> None:
    tokenizer = _CharacterTokenizer()
    encoded = tokenize_response_only(
        tokenizer,
        [
            {"role": "system", "content": "classify"},
            {"role": "user", "content": "untrusted ticket"},
            {"role": "assistant", "content": '{"issue_type":"MFA_FAILURE"}'},
        ],
        max_length=1_000,
        minimum_response_tokens=5,
    )
    assert tokenizer.tokenization_calls == 1
    first_supervised = next(index for index, value in enumerate(encoded.labels) if value != IGNORE_INDEX)
    assert all(value == IGNORE_INDEX for value in encoded.labels[:first_supervised])
    assert encoded.labels[first_supervised] == encoded.input_ids[first_supervised]
    assert encoded.supervised_tokens > 5


def test_truncation_guard_rejects_all_masked_example() -> None:
    tokenizer = _CharacterTokenizer()
    messages = [
        {"role": "system", "content": "x" * 100},
        {"role": "user", "content": "y" * 100},
        {"role": "assistant", "content": "response"},
    ]
    with pytest.raises(TrainingTokenizationError, match="too few supervised"):
        tokenize_response_only(tokenizer, messages, max_length=40)


def test_response_only_collation_uses_safe_padding() -> None:
    padded = pad_response_only_batch(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1], "labels": [-100, 4, 5]},
        ],
        pad_token_id=9,
        pad_to_multiple_of=4,
    )
    assert padded["input_ids"] == [[1, 2, 9, 9], [3, 4, 5, 9]]
    assert padded["attention_mask"] == [[1, 1, 0, 0], [1, 1, 1, 0]]
    assert padded["labels"] == [[-100, 2, -100, -100], [-100, 4, 5, -100]]


def test_yaml_config_is_strict_resolved_and_content_addressed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """config_version: '1.0'
experiment_name: fixture_run
model_name_or_path: fixture/model
model_revision: deadbeef
train_data: data/train.jsonl
validation_data: data/validation.jsonl
prompt_path: prompts/iam_triage_v1.txt
output_root: experiments/runs
""",
        encoding="utf-8",
    )
    config = load_training_config(config_path, project_root=tmp_path)
    assert config.train_data == tmp_path / "data/train.jsonl"
    assert config.validation_data == tmp_path / "data/validation.jsonl"
    assert len(config.content_hash) == 64
    assert config == load_training_config(config_path, project_root=tmp_path)


def test_checked_in_qwen_config_loads_without_ml_dependencies() -> None:
    config = load_training_config(
        Path("experiments/configs/qwen_lora_r16_v1.yaml"),
        project_root=Path.cwd(),
    )
    assert config.model_name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.lora.rank == 16
    assert config.optimizer == "adamw_torch"
    assert config.require_resolved_revision


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def is_bf16_supported() -> bool:
        return False


class _FakeMps:
    @staticmethod
    def is_available() -> bool:
        return False


def _fake_torch() -> Any:
    return SimpleNamespace(
        cuda=_FakeCuda(),
        backends=SimpleNamespace(mps=_FakeMps()),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )


def test_precision_selection_is_safe_on_cpu() -> None:
    plan = choose_precision(
        _fake_torch(),
        requested_device="auto",
        requested_precision="auto",
    )
    assert (plan.device, plan.precision, plan.fp16, plan.bf16) == ("cpu", "fp32", False, False)
    with pytest.raises(ModelConfigurationError, match="requires a compatible CUDA"):
        choose_precision(
            _fake_torch(),
            requested_device="cpu",
            requested_precision="bf16",
        )


def test_overfitting_signal_requires_train_down_and_validation_up() -> None:
    losses = LossHistory(
        training=(
            MetricPoint(step=1, value=2.0),
            MetricPoint(step=2, value=1.0),
        ),
        validation=(
            MetricPoint(step=1, value=0.8),
            MetricPoint(step=2, value=1.0),
        ),
    )
    signal = detect_overfitting(losses)
    assert signal.detected
    assert signal.best_validation_loss == 0.8
    assert signal.final_validation_loss == 1.0


def _example() -> LabeledExample:
    return LabeledExample(
        example_id="train_fixture_1",
        split=DatasetSplit.TRAIN,
        ticket=TicketInput(
            ticket_id="TKT-2001",
            subject="MFA push rejected",
            body="The MFA push is rejected for my account every time.",
            employee_department="Finance",
            submitted_at=datetime(2026, 8, 20, tzinfo=UTC),
        ),
        expected=TriageLabel(
            issue_type=IssueType.MFA_FAILURE,
            severity=Severity.P3,
            routing_team=RoutingTeam.IDENTITY_PLATFORM,
            affected_scope=AffectedScope.SINGLE_USER,
            recommended_action=RecommendedAction.INVESTIGATE_MFA_CONFIGURATION,
            evidence=["MFA push", "my account"],
        ),
        provenance=ExampleProvenance(
            source=DatasetSource.SYNTHETIC_TEMPLATE_GENERATOR,
            generator="unit-test",
            generator_version="1.0.0",
            template_id="mfa_fixture",
            seed=1,
        ),
        annotation_status=AnnotationStatus.SYNTHETICALLY_LABELED,
        review_status=ReviewStatus.PENDING_HUMAN_REVIEW,
    )


def test_training_target_uses_gold_label_and_neutral_non_gold_confidence() -> None:
    messages = supervised_messages(_example(), prompt=load_iam_prompt(), confidence=0.5)
    assistant = json.loads(messages[-1]["content"])
    assert assistant["issue_type"] == "MFA_FAILURE"
    assert assistant["evidence"] == ["MFA push", "my account"]
    assert assistant["confidence"] == 0.5


def test_manifest_is_hashable_and_write_once(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("adapter fixture", encoding="utf-8")
    digest = sha256_file(artifact)
    manifest = ExperimentManifest(
        experiment_name="fixture_run",
        completed_at=datetime(2026, 8, 20, tzinfo=UTC),
        config={"seed": 17},
        config_sha256="a" * 64,
        dataset_sha256={"train": "b" * 64, "validation": "c" * 64},
        prompt_version="iam_triage_v1",
        prompt_sha256="d" * 64,
        requested_model_revision="main",
        resolved_model_revision="deadbeef",
        requested_tokenizer_revision="main",
        resolved_tokenizer_revision="deadbeef",
        seed=17,
        parameter_counts=ParameterCounts(
            trainable=10,
            total=100,
            trainable_fraction=0.1,
        ),
        hardware=HardwareMetadata(
            platform="fixture",
            python_version="3.13",
            device="cpu",
            precision="fp32",
        ),
        git=GitMetadata(commit_sha="e" * 40, dirty=False),
        packages={"torch": "not-installed"},
        losses=LossHistory(),
        overfitting=OverfittingSignal(
            detected=False,
            reason="fixture",
        ),
        trainer_metrics={"train_loss": 1.0},
        artifacts_sha256={"artifact.txt": digest},
        truncated_training_examples=0,
        truncated_validation_examples=0,
    )
    destination = tmp_path / "manifest.json"
    write_manifest_once(destination, manifest)
    assert ExperimentManifest.model_validate_json(destination.read_text()) == manifest
    with pytest.raises(FileExistsError):
        write_manifest_once(destination, manifest)
