from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from modelforge.experiments.run_evaluation import run_model_evaluation
from modelforge.models import HeuristicTriageModel, HuggingFaceModelConfig
from modelforge.training.manifest import (
    ExperimentManifest,
    GitMetadata,
    HardwareMetadata,
    LossHistory,
    OverfittingSignal,
    ParameterCounts,
    canonical_sha256,
    sha256_file,
    write_manifest_once,
)
from scripts.colab_workflow import (
    EVIDENCE_INDEX_NAME,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    ColabWorkflowError,
    build_artifact_bundle,
    derive_cloud_config,
    derive_cloud_training_yaml,
    prepare_evidence_bundle,
    validate_commit_sha,
    validate_completed_training_run,
    validate_hf_repo_id,
    validate_inputs,
    validate_paired_evaluations,
    validate_paired_validation_evaluations,
    validate_run_id,
    verify_downloaded_bundle,
    verify_evidence_index,
    verify_git_checkout,
    verify_training_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "iam_triage_v1"


def _run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""config_version: '1.0'
experiment_name: qwen_lora_source_v1
model_name_or_path: {QWEN_MODEL_ID}
model_revision: {QWEN_REVISION}
tokenizer_name_or_path: {QWEN_MODEL_ID}
tokenizer_revision: {QWEN_REVISION}
trust_remote_code: false
local_files_only: false
require_resolved_revision: true
train_data: data/train.jsonl
validation_data: data/validation.jsonl
prompt_path: prompts/prompt.txt
output_root: experiments/runs
device: auto
precision: auto
""",
        encoding="utf-8",
    )


def _git_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    root = tmp_path / "source"
    root.mkdir()
    config = root / "experiments" / "config.yaml"
    _write_config(config)
    _run_git(root, "init")
    origin = "https://github.com/example/ModelForge.git"
    _run_git(root, "remote", "add", "origin", origin)
    _run_git(root, "add", ".")
    _run_git(
        root,
        "-c",
        "user.name=ModelForge Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return root, config, origin, _run_git(root, "rev-parse", "HEAD")


def _write_training_run(
    tmp_path: Path,
    *,
    run_id: str = "qwen-colab-run-001",
    source_sha: str = "a" * 40,
) -> Path:
    run_directory = tmp_path / "runs" / run_id
    adapter = run_directory / "adapter" / "adapter_config.json"
    checkpoint = run_directory / "checkpoints" / "trainer_state.json"
    adapter.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    adapter.write_text('{"adapter":true}\n', encoding="utf-8")
    checkpoint.write_text('{"step":10}\n', encoding="utf-8")
    artifacts = {
        "adapter/adapter_config.json": sha256_file(adapter),
        "checkpoints/trainer_state.json": sha256_file(checkpoint),
    }
    config = {
        "experiment_name": run_id,
        "model_name_or_path": QWEN_MODEL_ID,
        "model_revision": QWEN_REVISION,
        "tokenizer_name_or_path": QWEN_MODEL_ID,
        "tokenizer_revision": QWEN_REVISION,
        "trust_remote_code": False,
        "require_resolved_revision": True,
        "device": "cuda",
        "output_root": str(run_directory.parent),
    }
    manifest = ExperimentManifest(
        experiment_name=run_id,
        completed_at=datetime(2026, 9, 6, tzinfo=UTC),
        config=config,
        config_sha256=canonical_sha256(config),
        dataset_sha256={"train": "c" * 64, "validation": "d" * 64},
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
        packages={"torch": "fixture"},
        losses=LossHistory(),
        overfitting=OverfittingSignal(detected=False, reason="fixture"),
        trainer_metrics={"train_loss": 1.0},
        artifacts_sha256=artifacts,
        truncated_training_examples=0,
        truncated_validation_examples=0,
    )
    manifest_path = run_directory / "manifest.json"
    write_manifest_once(manifest_path, manifest)
    return manifest_path


def test_exact_identifier_validation() -> None:
    assert validate_commit_sha("a" * 40) == "a" * 40
    assert validate_hf_repo_id("rubiat/model-forge_artifacts") == "rubiat/model-forge_artifacts"
    assert validate_run_id("qwen-colab-001") == "qwen-colab-001"
    assert validate_inputs(
        source_commit="a" * 40,
        artifact_repo_id="rubiat/model-forge_artifacts",
        run_id="qwen-colab-001",
    ) == {
        "source_commit": "a" * 40,
        "artifact_repo_id": "rubiat/model-forge_artifacts",
        "run_id": "qwen-colab-001",
    }
    for invalid in ("main", "A" * 40, "a" * 39, " a" * 20):
        with pytest.raises(ColabWorkflowError):
            validate_commit_sha(invalid)
    for invalid in (
        "repo-only",
        "a/b/c",
        "a/../b",
        "a/-repo",
        "a/repo--copy",
        "YOUR_HF_USERNAME/artifacts",
    ):
        with pytest.raises(ColabWorkflowError):
            validate_hf_repo_id(invalid)
    for invalid in ("ab", "../run", "Run-1", "run--1", "run."):
        with pytest.raises(ColabWorkflowError):
            validate_run_id(invalid)


def test_checkout_and_cloud_yaml_are_pinned_and_unique(tmp_path: Path) -> None:
    root, config, origin, head = _git_fixture(tmp_path)
    checkout = verify_git_checkout(
        root,
        expected_origin=origin,
        expected_head=head,
        required_files=(config,),
    )
    cloud = derive_cloud_training_yaml(
        config,
        checkout=checkout,
        output_root=(tmp_path / "colab-output").resolve(),
        run_id="qwen-colab-001",
    )
    values = yaml.safe_load(cloud.yaml_text)
    assert values["experiment_name"] == "qwen-colab-001"
    assert values["output_root"] == str((tmp_path / "colab-output").resolve())
    assert values["device"] == "cuda"
    assert values["precision"] == "auto"
    assert values["model_revision"] == QWEN_REVISION
    assert values["tokenizer_revision"] == QWEN_REVISION
    assert values["trust_remote_code"] is False
    assert cloud.run_directory == tmp_path / "colab-output" / "qwen-colab-001"

    destination = (tmp_path / "derived" / "cloud.yaml").resolve()
    assert (
        derive_cloud_config(
            config,
            destination,
            checkout=checkout,
            output_root=(tmp_path / "second-output").resolve(),
            run_id="qwen-colab-002",
        )
        == destination
    )
    assert destination.stat().st_mode & 0o777 == 0o600
    assert yaml.safe_load(destination.read_text(encoding="utf-8"))["device"] == "cuda"

    cloud.run_directory.mkdir(parents=True)
    with pytest.raises(ColabWorkflowError, match="already exists"):
        derive_cloud_training_yaml(
            config,
            checkout=checkout,
            output_root=cloud.output_root,
            run_id=cloud.run_id,
        )


def test_checkout_rejects_dirty_or_wrong_identity(tmp_path: Path) -> None:
    root, config, origin, head = _git_fixture(tmp_path)
    with pytest.raises(ColabWorkflowError, match="origin"):
        verify_git_checkout(
            root, expected_origin="https://example.invalid/wrong.git", expected_head=head
        )
    config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ColabWorkflowError, match="clean"):
        verify_git_checkout(root, expected_origin=origin, expected_head=head)


def test_completed_training_run_rehashes_every_artifact(tmp_path: Path) -> None:
    source_sha = "a" * 40
    manifest_path = _write_training_run(tmp_path, source_sha=source_sha)
    training = validate_completed_training_run(
        manifest_path,
        expected_source_sha=source_sha,
        expected_run_id="qwen-colab-run-001",
    )
    assert training.manifest_sha256 == sha256_file(manifest_path)
    assert set(training.artifact_hashes) == {
        "adapter/adapter_config.json",
        "checkpoints/trainer_state.json",
    }
    summary = verify_training_manifest(
        manifest_path,
        source_commit=source_sha,
        run_id="qwen-colab-run-001",
    )
    assert summary["manifest_sha256"] == training.manifest_sha256
    assert summary["device"] == "cuda"

    (manifest_path.parent / "adapter" / "adapter_config.json").write_text(
        '{"adapter":false}\n',
        encoding="utf-8",
    )
    with pytest.raises(ColabWorkflowError, match="SHA-256"):
        validate_completed_training_run(
            manifest_path,
            expected_source_sha=source_sha,
            expected_run_id="qwen-colab-run-001",
        )


def test_bundle_contains_only_allowlisted_files_and_deterministic_index(tmp_path: Path) -> None:
    manifest_path = _write_training_run(tmp_path)
    training = validate_completed_training_run(
        manifest_path,
        expected_source_sha="a" * 40,
        expected_run_id="qwen-colab-run-001",
    )
    environment = tmp_path / "pip-freeze.txt"
    evaluation = tmp_path / "base-validation.json"
    environment.write_text("torch==2.0\n", encoding="utf-8")
    evaluation.write_text('{"fixture":true}\n', encoding="utf-8")
    first = build_artifact_bundle(
        training,
        (tmp_path / "bundle-one").resolve(),
        environment_files={"pip-freeze.txt": environment},
        evaluation_files={"base-validation.json": evaluation},
    )
    second = build_artifact_bundle(
        training,
        (tmp_path / "bundle-two").resolve(),
        environment_files={"pip-freeze.txt": environment},
        evaluation_files={"base-validation.json": evaluation},
    )
    expected = {
        EVIDENCE_INDEX_NAME,
        "environment/pip-freeze.txt",
        "evaluations/base-validation.json",
        "training/qwen-colab-run-001/adapter/adapter_config.json",
        "training/qwen-colab-run-001/checkpoints/trainer_state.json",
        "training/qwen-colab-run-001/manifest.json",
    }
    actual = {
        path.relative_to(first.root).as_posix() for path in first.root.rglob("*") if path.is_file()
    }
    assert actual == expected
    assert first.index_path.read_bytes() == second.index_path.read_bytes()
    assert verify_evidence_index(first.root) == first.files_sha256
    assert verify_downloaded_bundle(first.root) == first.files_sha256

    wrapper = prepare_evidence_bundle(
        manifest_path,
        (tmp_path / "bundle-wrapper").resolve(),
        source_commit="a" * 40,
        run_id="qwen-colab-run-001",
        environment_files={"pip-freeze.txt": environment},
        evaluation_files={"base-validation.json": evaluation},
    )
    assert wrapper.is_dir()


def test_bundle_rejects_tokens_traversal_and_symlinks(tmp_path: Path) -> None:
    manifest_path = _write_training_run(tmp_path)
    training = validate_completed_training_run(
        manifest_path,
        expected_source_sha="a" * 40,
        expected_run_id="qwen-colab-run-001",
    )
    secret = tmp_path / "environment.txt"
    secret.write_text(f"HF_TOKEN=hf_{'a' * 32}\n", encoding="utf-8")
    with pytest.raises(ColabWorkflowError, match="token-like"):
        build_artifact_bundle(
            training,
            (tmp_path / "token-bundle").resolve(),
            environment_files={"environment.txt": secret},
            evaluation_files={},
        )
    with pytest.raises(ColabWorkflowError, match="traversal"):
        build_artifact_bundle(
            training,
            (tmp_path / "traversal-bundle").resolve(),
            environment_files={"../environment.txt": secret},
            evaluation_files={},
        )

    credentials = tmp_path / "harmless.txt"
    credentials.write_text("no secret content\n", encoding="utf-8")
    with pytest.raises(ColabWorkflowError, match="secret-like"):
        build_artifact_bundle(
            training,
            (tmp_path / "filename-bundle").resolve(),
            environment_files={"credentials.json": credentials},
            evaluation_files={},
        )

    safe = tmp_path / "safe.txt"
    safe.write_text("safe\n", encoding="utf-8")
    link = tmp_path / "linked.txt"
    link.symlink_to(safe)
    with pytest.raises(ColabWorkflowError, match="non-symlink"):
        build_artifact_bundle(
            training,
            (tmp_path / "symlink-bundle").resolve(),
            environment_files={"linked.txt": link},
            evaluation_files={},
        )


class _MeasuredFixtureModel(HeuristicTriageModel):
    role = "small"

    def __init__(self, *, adapter_path: Path | None = None) -> None:
        self.model_id = "small-iam-triage-v1"
        self.config = HuggingFaceModelConfig(
            model_name_or_path=QWEN_MODEL_ID,
            revision=QWEN_REVISION,
            adapter_name_or_path=str(adapter_path) if adapter_path is not None else None,
            serving_id=self.model_id,
            device="cuda",
        )

    async def triage(self, ticket, *, timeout_s=None):
        prediction = await super().triage(ticket, timeout_s=timeout_s)
        return prediction.model_copy(
            update={
                "model_id": self.model_id,
                "model_version": "measured-fixture-version",
                "role": self.role,
            }
        )


def test_paired_validation_requires_full_matched_provenance(tmp_path: Path) -> None:
    manifest_path = _write_training_run(tmp_path)
    training = validate_completed_training_run(
        manifest_path,
        expected_source_sha="a" * 40,
        expected_run_id="qwen-colab-run-001",
    )
    base_path = tmp_path / "base.json"
    adapter_path = tmp_path / "adapter.json"
    asyncio.run(
        run_model_evaluation(
            model=_MeasuredFixtureModel(),
            backend="hf-base",
            dataset_root=DATASET_ROOT,
            split="validation",
            output_path=base_path,
            experiment_id="base-validation-fixture",
        )
    )
    asyncio.run(
        run_model_evaluation(
            model=_MeasuredFixtureModel(adapter_path=manifest_path.parent / "adapter"),
            backend="hf-adapter",
            dataset_root=DATASET_ROOT,
            split="validation",
            output_path=adapter_path,
            experiment_id="adapter-validation-fixture",
            training_manifest_path=manifest_path,
        )
    )
    pair = validate_paired_validation_evaluations(
        base_path,
        adapter_path,
        training=training,
    )
    assert pair.base.evaluated_count == 100
    assert pair.adapter.training_run is not None
    assert pair.adapter.training_run.training_manifest_sha256 == training.manifest_sha256
    summary = validate_paired_evaluations(
        base_path,
        adapter_path,
        training_manifest_path=manifest_path,
        source_commit="a" * 40,
        run_id="qwen-colab-run-001",
    )
    assert summary["validation_case_count"] == 100
    assert summary["training_manifest_sha256"] == training.manifest_sha256

    bad_base_path = tmp_path / "base-operational-error.json"
    bad_base = json.loads(base_path.read_text(encoding="utf-8"))
    bad_base["operations"]["valid_predictions"] -= 1
    bad_base["operations"]["operational_errors"] = 1
    bad_base_path.write_text(json.dumps(bad_base), encoding="utf-8")
    with pytest.raises(ColabWorkflowError, match="operational"):
        validate_paired_evaluations(
            bad_base_path,
            adapter_path,
            training_manifest_path=manifest_path,
            source_commit="a" * 40,
            run_id="qwen-colab-run-001",
        )

    raw = json.loads(adapter_path.read_text(encoding="utf-8"))
    raw["training_run"]["training_manifest_sha256"] = "f" * 64
    adapter_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ColabWorkflowError, match="not linked"):
        validate_paired_validation_evaluations(
            base_path,
            adapter_path,
            training=training,
        )
