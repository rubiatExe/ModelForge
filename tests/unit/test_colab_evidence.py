from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from modelforge.datasets import load_training_examples, load_validation_examples
from modelforge.models import HeuristicTriageModel, HuggingFaceModelConfig
from modelforge.models.prompting import load_iam_prompt
from modelforge.training.config import load_training_config
from modelforge.training.manifest import (
    ExperimentManifest,
    GitMetadata,
    HardwareMetadata,
    LossHistory,
    MetricPoint,
    OverfittingSignal,
    ParameterCounts,
    canonical_sha256,
    sha256_file,
    write_manifest_once,
)
from scripts.colab_workflow import (
    QWEN_MODEL_ID,
    QWEN_REVISION,
    SMOKE_VALIDATION_CASE_ID,
    AdapterApiSmokeArtifact,
    ColabWorkflowError,
    ConfidenceRoutingSmokeArtifact,
    ResponseOnlyMaskAuditArtifact,
    _resolved_tokenizer_snapshot,
    audit_response_only_masks,
    build_artifact_bundle,
    run_adapter_api_smoke,
    run_confidence_routing_smoke,
    validate_completed_training_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/iam_triage_v1"
CONFIG_PATH = PROJECT_ROOT / "experiments/configs/qwen_lora_r16_colab_v1.yaml"
RUN_ID = "qwen-colab-evidence-001"


class _OffsetTokenizer:
    is_fast = True
    init_kwargs = {"_commit_hash": QWEN_REVISION}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        rendered = "".join(f"<{message['role']}>\n{message['content']}\n" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return rendered

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        pieces = list(re.finditer(r"\w+|[^\w\s]", text))
        return {
            "input_ids": list(range(1, len(pieces) + 1)),
            "offset_mapping": [(match.start(), match.end()) for match in pieces],
        }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_project(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    shutil.copytree(DATASET_ROOT, root / "data/iam_triage_v1")
    for relative in (
        "src/modelforge/routing/confidence.py",
        "src/modelforge/routing/policy.py",
        "src/modelforge/routing/router.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, destination)
    _git(root, "init")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=ModelForge Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "-m",
        "evidence fixture",
    )
    return root, _git(root, "rev-parse", "HEAD")


def _training_run(tmp_path: Path, *, source_sha: str, project_root: Path) -> Path:
    run_directory = tmp_path / "runs" / RUN_ID
    adapter_config = run_directory / "adapter" / "adapter_config.json"
    adapter_weights = run_directory / "adapter" / "adapter_model.safetensors"
    adapter_config.parent.mkdir(parents=True)
    adapter_config.write_text('{"adapter":true}\n', encoding="utf-8")
    adapter_weights.write_bytes(b"fixture-safetensors")
    config = {
        "experiment_name": RUN_ID,
        "model_name_or_path": QWEN_MODEL_ID,
        "model_revision": QWEN_REVISION,
        "tokenizer_name_or_path": QWEN_MODEL_ID,
        "tokenizer_revision": QWEN_REVISION,
        "trust_remote_code": False,
        "require_resolved_revision": True,
        "device": "cuda",
        "epochs": 1.0,
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "output_root": str(run_directory.parent),
        "train_data": str(project_root / "data/iam_triage_v1/train.jsonl"),
        "validation_data": str(project_root / "data/iam_triage_v1/validation.jsonl"),
    }
    manifest = ExperimentManifest(
        experiment_name=RUN_ID,
        completed_at=datetime(2026, 9, 6, tzinfo=UTC),
        config=config,
        config_sha256=canonical_sha256(config),
        dataset_sha256={
            "train": sha256_file(project_root / "data/iam_triage_v1/train.jsonl"),
            "validation": sha256_file(project_root / "data/iam_triage_v1/validation.jsonl"),
        },
        prompt_version="iam_triage_v1",
        prompt_sha256="e" * 64,
        requested_model_revision=QWEN_REVISION,
        resolved_model_revision=QWEN_REVISION,
        requested_tokenizer_revision=QWEN_REVISION,
        resolved_tokenizer_revision=QWEN_REVISION,
        seed=17,
        parameter_counts=ParameterCounts(
            trainable=10,
            total=100,
            trainable_fraction=0.1,
        ),
        hardware=HardwareMetadata(
            platform="Linux",
            python_version="3.12",
            device="cuda",
            precision="fp16",
            accelerator_name="Tesla T4",
        ),
        git=GitMetadata(commit_sha=source_sha, dirty=False),
        packages={
            "torch": "fixture",
            "transformers": "fixture",
            "peft": "fixture",
            "accelerate": "fixture",
        },
        losses=LossHistory(
            training=(MetricPoint(step=1, epoch=1.0, value=1.0),),
            validation=(MetricPoint(step=1, epoch=1.0, value=0.9),),
        ),
        overfitting=OverfittingSignal(detected=False, reason="fixture"),
        trainer_metrics={"train_loss": 1.0, "eval_loss": 0.9},
        artifacts_sha256={
            "adapter/adapter_config.json": sha256_file(adapter_config),
            "adapter/adapter_model.safetensors": sha256_file(adapter_weights),
        },
        truncated_training_examples=0,
        truncated_validation_examples=0,
    )
    manifest_path = run_directory / "manifest.json"
    write_manifest_once(manifest_path, manifest)
    return manifest_path


class _FastApiAdapterFixture:
    role = "small"

    def __init__(self, config: HuggingFaceModelConfig, **_: object) -> None:
        assert config.adapter_name_or_path is not None
        assert config.local_files_only is True
        assert config.trust_remote_code is False
        assert config.max_new_tokens == 256
        self.config = config
        self.model_id = config.serving_id
        self.closed = False

    async def triage(self, ticket, *, timeout_s=None):
        prediction = await HeuristicTriageModel().triage(ticket, timeout_s=timeout_s)
        return prediction.model_copy(
            update={
                "model_id": self.model_id,
                "model_version": "adapter-api-fixture-v1",
                "role": self.role,
            }
        )

    def close(self) -> None:
        self.closed = True


def test_response_mask_audit_covers_every_training_row_without_content() -> None:
    config = load_training_config(CONFIG_PATH, project_root=PROJECT_ROOT)
    examples = load_training_examples(DATASET_ROOT)
    prompt = load_iam_prompt(config.prompt_path)
    audit = audit_response_only_masks(
        _OffsetTokenizer(),
        examples,
        config=config,
        prompt=prompt,
    )
    repeated = audit_response_only_masks(
        _OffsetTokenizer(),
        examples,
        config=config,
        prompt=prompt,
    )
    assert audit["audited_train_examples"] == len(examples) == 500
    assert audit["truncated_examples"] == 0
    assert audit["masked_prefix_tokens"].minimum > 0
    assert audit["supervised_response_tokens"].minimum >= config.minimum_response_tokens
    assert audit["row_audit_sha256"] == repeated["row_audit_sha256"]
    serialized = json.dumps(
        audit,
        default=lambda value: value.model_dump(mode="json"),
    )
    assert not any(example.ticket.body in serialized for example in examples)

    truncated_config = config.model_copy(update={"max_length": 800})
    with pytest.raises(ColabWorkflowError, match="truncated training rows"):
        audit_response_only_masks(
            _OffsetTokenizer(),
            examples,
            config=truncated_config,
            prompt=prompt,
        )


def test_tokenizer_revision_can_be_proven_from_hugging_face_snapshot_paths() -> None:
    class SnapshotTokenizer:
        init_kwargs = {
            "vocab_file": f"/cache/models--Qwen/snapshots/{QWEN_REVISION}/vocab.json",
            "merges_file": f"/cache/models--Qwen/snapshots/{QWEN_REVISION}/merges.txt",
        }

    assert _resolved_tokenizer_snapshot(SnapshotTokenizer()) == (
        QWEN_REVISION,
        "cached_snapshot_paths",
    )


def test_routing_smoke_persists_uncalibrated_fail_closed_evidence(
    tmp_path: Path,
) -> None:
    project, source_sha = _clean_project(tmp_path)
    output = (tmp_path / "evidence" / "confidence-routing-smoke.json").resolve()
    assert (
        run_confidence_routing_smoke(
            output,
            project_root=project,
            source_commit=source_sha,
            run_id=RUN_ID,
        )
        == output
    )
    artifact = ConfidenceRoutingSmokeArtifact.model_validate_json(
        output.read_text(encoding="utf-8"), strict=True
    )
    assert artifact.result_status == "fixture"
    assert artifact.policy_status == "uncalibrated"
    assert artifact.confidence_is_probability is False
    assert artifact.claims_calibration is False
    by_name = {case.scenario: case for case in artifact.cases}
    assert by_name["high_confidence_local"].route_reason == "small_confident"
    assert by_name["high_confidence_local"].used_frontier is False
    assert by_name["low_confidence_fail_closed"].safe_error_type == "FrontierUnavailableError"
    assert by_name["small_model_error_fail_closed"].safe_error_type == "FrontierUnavailableError"
    serialized = output.read_text(encoding="utf-8")
    assert "Okta SSO redirect loop" not in serialized
    assert "Ignore previous instructions" not in serialized


def test_adapter_api_smoke_uses_testclient_and_redacts_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, source_sha = _clean_project(tmp_path)
    manifest_path = _training_run(tmp_path, source_sha=source_sha, project_root=project)
    import modelforge.models.huggingface as huggingface

    monkeypatch.setattr(huggingface, "HuggingFaceTriageModel", _FastApiAdapterFixture)
    output = (tmp_path / "evidence" / "adapter-api-smoke.json").resolve()
    assert (
        run_adapter_api_smoke(
            manifest_path,
            output,
            dataset_root=project / "data/iam_triage_v1",
            project_root=project,
            source_commit=source_sha,
            run_id=RUN_ID,
        )
        == output
    )
    artifact = AdapterApiSmokeArtifact.model_validate_json(
        output.read_text(encoding="utf-8"), strict=True
    )
    assert artifact.response_status_code == 200
    assert artifact.x_modelforge_model == "small-iam-triage-v1"
    assert artifact.response_schema_name == "TriageResult"
    assert artifact.response_schema_validated is True
    assert artifact.transport == "fastapi.testclient.TestClient"
    assert artifact.max_new_tokens == 256
    assert artifact.validation_case_predeclared is True
    assert artifact.validation_case_count == 100
    assert artifact.request_body_persisted is False
    assert artifact.response_body_persisted is False
    serialized = output.read_text(encoding="utf-8")
    examples = load_validation_examples(project / "data/iam_triage_v1")
    selected = next(
        example for example in examples if str(example.example_id) == SMOKE_VALIDATION_CASE_ID
    )
    assert artifact.validation_case_id == str(selected.example_id)
    expected_case_hash = hashlib.sha256(
        "".join(
            f"{example_id}\n" for example_id in sorted(str(e.example_id) for e in examples)
        ).encode("utf-8")
    ).hexdigest()
    assert artifact.validation_case_ids_sha256 == expected_case_hash
    dataset_root = project / "data/iam_triage_v1"
    assert artifact.dataset_manifest_sha256 == sha256_file(dataset_root / "manifest.json")
    assert artifact.validation_split_file_sha256 == sha256_file(dataset_root / "validation.jsonl")
    prediction = asyncio.run(HeuristicTriageModel().triage(selected.ticket)).result
    private_fragments = (
        selected.ticket.subject,
        selected.ticket.body,
        *prediction.evidence,
    )
    assert all(fragment not in serialized for fragment in private_fragments)


def test_bundle_places_named_evidence_only_under_audits(tmp_path: Path) -> None:
    project, source_sha = _clean_project(tmp_path)
    manifest_path = _training_run(tmp_path, source_sha=source_sha, project_root=project)
    training = validate_completed_training_run(
        manifest_path,
        expected_source_sha=source_sha,
        expected_run_id=RUN_ID,
    )
    audit = tmp_path / "routing-smoke.json"
    audit.write_text('{"safe":true}\n', encoding="utf-8")
    bundle = build_artifact_bundle(
        training,
        (tmp_path / "bundle").resolve(),
        environment_files={},
        evaluation_files={},
        audit_files={"routing-smoke.json": audit},
    )
    assert (bundle.root / "audits/routing-smoke.json").read_bytes() == audit.read_bytes()
    assert "audits/routing-smoke.json" in bundle.files_sha256


def test_training_evidence_rejects_non_lora_or_unmeasured_manifest(tmp_path: Path) -> None:
    project, source_sha = _clean_project(tmp_path)
    manifest_path = _training_run(tmp_path, source_sha=source_sha, project_root=project)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["parameter_counts"]["trainable"] = 0
    raw["parameter_counts"]["trainable_fraction"] = 0.0
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ColabWorkflowError, match="parameter-efficient training"):
        validate_completed_training_run(
            manifest_path,
            expected_source_sha=source_sha,
            expected_run_id=RUN_ID,
        )


def test_mask_audit_artifact_rejects_incomplete_coverage() -> None:
    values = {
        "completed_at": datetime(2026, 9, 6, tzinfo=UTC),
        "source_commit": "a" * 40,
        "run_id": RUN_ID,
        "training_config_sha256": "b" * 64,
        "dataset_name": "fixture",
        "dataset_version": "1.0.0",
        "train_file_sha256": "c" * 64,
        "case_ids_sha256": "d" * 64,
        "prompt_version": "iam_triage_v1",
        "prompt_sha256": "e" * 64,
        "tokenizer_name_or_path": QWEN_MODEL_ID,
        "requested_tokenizer_revision": QWEN_REVISION,
        "resolved_tokenizer_revision": QWEN_REVISION,
        "resolved_revision_evidence": "tokenizer_init_commit_hash",
        "tokenizer_is_fast": True,
        "tokenizer_local_files_only": True,
        "trust_remote_code": False,
        "max_length": 1024,
        "minimum_response_tokens": 16,
        "expected_train_examples": 500,
        "audited_train_examples": 499,
        "truncated_examples": 0,
        "sequence_tokens": {"minimum": 100, "maximum": 110, "total": 1000, "mean": 105.0},
        "masked_prefix_tokens": {"minimum": 50, "maximum": 60, "total": 500, "mean": 55.0},
        "supervised_response_tokens": {
            "minimum": 40,
            "maximum": 50,
            "total": 450,
            "mean": 45.0,
        },
        "row_audit_sha256": "f" * 64,
        "assistant_boundary_matches_offset_audit": True,
        "every_pre_response_label_is_ignored": True,
        "every_response_label_matches_input_id": True,
        "every_row_meets_minimum_response_tokens": True,
        "contains_ticket_text": False,
        "contains_token_ids": False,
        "passed": True,
        "limitations": ("fixture",),
    }
    with pytest.raises(ValueError, match="every manifest-declared training row"):
        ResponseOnlyMaskAuditArtifact.model_validate(values)
