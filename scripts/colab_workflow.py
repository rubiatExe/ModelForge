"""Pure safety and provenance helpers for the hosted Colab workflow.

This module intentionally has no Google Colab or Hugging Face Hub imports.  A
notebook can use these helpers around its external API calls, while the
validation and bundle-building behavior remains locally testable.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from modelforge.experiments.run_evaluation import ModelEvaluationArtifact
from modelforge.training.config import LoraTrainingConfig
from modelforge.training.manifest import ExperimentManifest, canonical_sha256, sha256_file

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_VALIDATION_CASES = 100
EVIDENCE_INDEX_NAME = "evidence-index.sha256"
MAX_TEXT_SCAN_BYTES = 16 * 1024 * 1024

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,99}$")
_REPO_SEGMENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_HF_TOKEN = re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}")
_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_TEXT_NAMES = {"license", "notice", "readme", "sha256sums"}
_SECRET_FILENAMES = {
    ".netrc",
    "auth.json",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secret",
    "secrets",
    "secrets.json",
    "token",
    "token.json",
}
_SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}


class ColabWorkflowError(ValueError):
    """A fail-closed workflow validation error safe to show in a notebook."""


@dataclass(frozen=True, slots=True)
class VerifiedCheckout:
    root: Path
    origin: str
    head: str
    tracked_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CloudTrainingConfig:
    yaml_text: str
    run_id: str
    output_root: Path
    run_directory: Path


@dataclass(frozen=True, slots=True)
class ValidatedTrainingRun:
    manifest: ExperimentManifest
    manifest_path: Path
    manifest_sha256: str
    run_directory: Path
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    root: Path
    index_path: Path
    files_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ValidatedEvaluationPair:
    base: ModelEvaluationArtifact
    adapter: ModelEvaluationArtifact


def validate_commit_sha(value: str) -> str:
    """Require a full lowercase Git commit SHA, never a branch or short hash."""

    if not isinstance(value, str) or not _SHA40.fullmatch(value):
        raise ColabWorkflowError(
            "source commit must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def validate_hf_repo_id(value: str) -> str:
    """Validate an explicit ``namespace/name`` Hugging Face repository ID."""

    if not isinstance(value, str) or value != value.strip() or len(value) > 96:
        raise ColabWorkflowError("artifact repository must be a valid namespace/name identifier")
    if value.startswith("YOUR_HF_USERNAME/"):
        raise ColabWorkflowError("replace the artifact repository placeholder")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ColabWorkflowError("artifact repository must include exactly one namespace and name")
    for part in parts:
        if (
            not _REPO_SEGMENT.fullmatch(part)
            or part[0] in ".-"
            or part[-1] in ".-"
            or ".." in part
            or "--" in part
        ):
            raise ColabWorkflowError("artifact repository contains an unsafe identifier segment")
    return value


def validate_run_id(value: str) -> str:
    """Require one safe, portable experiment and remote-prefix identifier."""

    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ColabWorkflowError(
            "run id must be 3-100 lowercase letters, digits, dots, underscores, or hyphens"
        )
    if value.endswith((".", "-")) or ".." in value or "--" in value:
        raise ColabWorkflowError("run id contains an unsafe or ambiguous path component")
    return value


def _git(project_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ColabWorkflowError("unable to verify the Git checkout") from exc
    return result.stdout.strip()


def _relative_to_root(path: Path, root: Path, *, label: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ColabWorkflowError(
            f"{label} must be a regular file inside the verified checkout"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise ColabWorkflowError(f"{label} must be a regular non-symlink file")
    return relative.as_posix()


def verify_git_checkout(
    project_root: Path,
    *,
    expected_origin: str,
    expected_head: str,
    required_files: Sequence[Path] = (),
) -> VerifiedCheckout:
    """Verify repository identity, detached content identity, and cleanliness."""

    head = validate_commit_sha(expected_head)
    if not expected_origin or expected_origin != expected_origin.strip():
        raise ColabWorkflowError("expected Git origin must be a non-empty exact URL")
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise ColabWorkflowError("project root must be a directory")
    reported_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if reported_root != root:
        raise ColabWorkflowError("project root is not the verified Git checkout root")
    if _git(root, "remote", "get-url", "origin") != expected_origin:
        raise ColabWorkflowError("Git origin does not match the expected repository")
    if _git(root, "rev-parse", "HEAD") != head:
        raise ColabWorkflowError("Git HEAD does not match the pinned source commit")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ColabWorkflowError("Git checkout must be clean, including untracked files")

    tracked: list[str] = []
    for required in required_files:
        candidate = Path(required)
        if not candidate.is_absolute():
            candidate = root / candidate
        relative = _relative_to_root(candidate, root, label="required source file")
        _git(root, "ls-files", "--error-unmatch", "--", relative)
        _git(root, "cat-file", "-e", f"{head}:{relative}")
        tracked.append(relative)
    return VerifiedCheckout(
        root=root,
        origin=expected_origin,
        head=head,
        tracked_files=tuple(sorted(tracked)),
    )


def derive_cloud_training_yaml(
    source_config: Path,
    *,
    checkout: VerifiedCheckout,
    output_root: Path,
    run_id: str,
) -> CloudTrainingConfig:
    """Derive a CUDA config from one verified, committed Qwen configuration."""

    safe_run_id = validate_run_id(run_id)
    source_path = Path(source_config)
    if not source_path.is_absolute():
        source_path = checkout.root / source_path
    relative = _relative_to_root(source_path, checkout.root, label="training config")
    source_path = checkout.root / relative
    if relative not in checkout.tracked_files:
        raise ColabWorkflowError("training config was not included in checkout verification")
    if _git(checkout.root, "rev-parse", "HEAD") != checkout.head:
        raise ColabWorkflowError("Git HEAD changed after checkout verification")
    if _git(checkout.root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ColabWorkflowError("Git checkout changed after checkout verification")

    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ColabWorkflowError("unable to read the committed training YAML") from exc
    if not isinstance(raw, dict):
        raise ColabWorkflowError("committed training YAML must contain an object")
    try:
        source = LoraTrainingConfig.model_validate(raw)
    except Exception as exc:
        raise ColabWorkflowError("committed training YAML failed schema validation") from exc
    expected_identity = {
        "model_name_or_path": QWEN_MODEL_ID,
        "model_revision": QWEN_REVISION,
        "tokenizer_name_or_path": QWEN_MODEL_ID,
        "tokenizer_revision": QWEN_REVISION,
        "trust_remote_code": False,
        "require_resolved_revision": True,
    }
    for field, expected in expected_identity.items():
        if getattr(source, field) != expected:
            raise ColabWorkflowError(f"committed training config has an unexpected {field}")

    requested_output = Path(output_root)
    if not requested_output.is_absolute():
        raise ColabWorkflowError("Colab output root must be an absolute path")
    resolved_output = requested_output.resolve(strict=False)
    try:
        resolved_output.relative_to(checkout.root)
    except ValueError:
        pass
    else:
        raise ColabWorkflowError("Colab output root must be outside the Git checkout")
    run_directory = resolved_output / safe_run_id
    if run_directory.exists() or run_directory.is_symlink():
        raise ColabWorkflowError("run directory already exists; choose a new run id")

    cloud_raw = dict(raw)
    cloud_raw.update(
        {
            "experiment_name": safe_run_id,
            "output_root": str(resolved_output),
            "device": "cuda",
            "precision": "auto",
            "local_files_only": False,
        }
    )
    try:
        cloud = LoraTrainingConfig.model_validate(cloud_raw)
    except Exception as exc:
        raise ColabWorkflowError("derived Colab training config failed schema validation") from exc
    for field, expected in expected_identity.items():
        if getattr(cloud, field) != expected:
            raise ColabWorkflowError(f"derived training config changed protected field {field}")
    text = yaml.safe_dump(cloud_raw, allow_unicode=True, sort_keys=False)
    if yaml.safe_load(text) != cloud_raw:
        raise ColabWorkflowError("derived training YAML did not round-trip exactly")
    return CloudTrainingConfig(
        yaml_text=text,
        run_id=safe_run_id,
        output_root=resolved_output,
        run_directory=run_directory,
    )


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ColabWorkflowError(f"{label} must be a non-empty relative POSIX path")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ColabWorkflowError(f"{label} contains unsafe characters")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ColabWorkflowError(f"{label} contains path traversal or ambiguous components")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ColabWorkflowError(f"{label} must be relative")
    return path


def _walk_regular_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ColabWorkflowError("unable to inspect an artifact path") from exc
        if stat.S_ISLNK(mode):
            raise ColabWorkflowError("symlinks are forbidden in evidence artifacts")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ColabWorkflowError("only regular files are allowed in evidence artifacts")
        relative = path.relative_to(root).as_posix()
        _safe_relative_path(relative, label="artifact path")
        files[relative] = path
    return files


def _declared_artifact_hashes(
    run_directory: Path,
    manifest_path: Path,
    declared: Mapping[str, str],
) -> dict[str, str]:
    actual_files = _walk_regular_files(run_directory)
    manifest_relative = manifest_path.relative_to(run_directory).as_posix()
    actual_files.pop(manifest_relative, None)
    normalized: dict[str, str] = {}
    for name, digest in declared.items():
        relative = _safe_relative_path(name, label="manifest artifact path").as_posix()
        if relative in normalized:
            raise ColabWorkflowError("training manifest contains duplicate artifact paths")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ColabWorkflowError("training manifest contains an invalid artifact SHA-256")
        normalized[relative] = digest
    if not normalized:
        raise ColabWorkflowError("training manifest declares no artifacts")
    if set(actual_files) != set(normalized):
        raise ColabWorkflowError(
            "run directory does not contain exactly the manifest-declared files"
        )
    actual = {name: sha256_file(actual_files[name]) for name in sorted(actual_files)}
    if actual != dict(sorted(normalized.items())):
        raise ColabWorkflowError("training artifacts do not match the manifest SHA-256 values")
    return actual


def validate_completed_training_run(
    manifest_path: Path,
    *,
    expected_source_sha: str,
    expected_run_id: str,
    expected_revision: str = QWEN_REVISION,
) -> ValidatedTrainingRun:
    """Validate a completed CUDA run and re-hash every declared artifact."""

    source_sha = validate_commit_sha(expected_source_sha)
    run_id = validate_run_id(expected_run_id)
    revision = validate_commit_sha(expected_revision)
    path = Path(manifest_path)
    if path.is_symlink() or not path.is_file() or path.name != "manifest.json":
        raise ColabWorkflowError("training manifest must be a regular manifest.json file")
    path = path.resolve(strict=True)
    run_directory = path.parent.resolve(strict=True)
    if run_directory.name != run_id:
        raise ColabWorkflowError("training manifest is not inside the expected run directory")
    try:
        manifest = ExperimentManifest.model_validate_json(
            path.read_text(encoding="utf-8"),
            strict=True,
        )
    except Exception as exc:
        raise ColabWorkflowError("training manifest failed strict schema validation") from exc
    if manifest.experiment_name != run_id:
        raise ColabWorkflowError("training manifest experiment name does not match the run id")
    if manifest.git.commit_sha != source_sha or manifest.git.dirty is not False:
        raise ColabWorkflowError(
            "training manifest does not prove the expected clean source commit"
        )
    if manifest.resolved_model_revision != revision:
        raise ColabWorkflowError("training manifest resolved an unexpected model revision")
    if manifest.resolved_tokenizer_revision != revision:
        raise ColabWorkflowError("training manifest resolved an unexpected tokenizer revision")
    if manifest.requested_model_revision != revision:
        raise ColabWorkflowError("training manifest requested an unexpected model revision")
    if manifest.requested_tokenizer_revision != revision:
        raise ColabWorkflowError("training manifest requested an unexpected tokenizer revision")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", manifest.hardware.device):
        raise ColabWorkflowError("training manifest does not record a CUDA device")
    if manifest.config_sha256 != canonical_sha256(manifest.config):
        raise ColabWorkflowError("training manifest config hash does not match its config")

    config = manifest.config
    protected_config = {
        "experiment_name": run_id,
        "model_name_or_path": QWEN_MODEL_ID,
        "model_revision": revision,
        "tokenizer_name_or_path": QWEN_MODEL_ID,
        "tokenizer_revision": revision,
        "trust_remote_code": False,
        "require_resolved_revision": True,
        "device": "cuda",
    }
    for field, expected in protected_config.items():
        if config.get(field) != expected:
            raise ColabWorkflowError(f"training manifest config has an unexpected {field}")
    configured_output = config.get("output_root")
    if not isinstance(configured_output, str):
        raise ColabWorkflowError("training manifest config lacks an absolute output root")
    output_root = Path(configured_output)
    if not output_root.is_absolute() or output_root.resolve(strict=True) != run_directory.parent:
        raise ColabWorkflowError("training manifest output root does not match the run location")

    hashes = _declared_artifact_hashes(run_directory, path, manifest.artifacts_sha256)
    return ValidatedTrainingRun(
        manifest=manifest,
        manifest_path=path.resolve(strict=True),
        manifest_sha256=sha256_file(path),
        run_directory=run_directory,
        artifact_hashes=hashes,
    )


def _looks_secret_filename(relative: str) -> bool:
    name = PurePosixPath(relative).name.casefold()
    return (
        name == ".env"
        or name.startswith(".env.")
        or name in _SECRET_FILENAMES
        or any(name.endswith(suffix) for suffix in _SECRET_SUFFIXES)
    )


def _is_text_artifact(path: Path) -> bool:
    return path.suffix.casefold() in _TEXT_SUFFIXES or path.name.casefold() in _TEXT_NAMES


def _validate_upload_source(path: Path, *, relative_name: str, require_text: bool) -> None:
    if path.is_symlink() or not path.is_file():
        raise ColabWorkflowError("bundle inputs must be regular non-symlink files")
    if _looks_secret_filename(relative_name):
        raise ColabWorkflowError("secret-like filenames are forbidden in the artifact bundle")
    is_text = _is_text_artifact(path)
    if require_text and not is_text:
        raise ColabWorkflowError("named environment and evaluation files must be text artifacts")
    if is_text:
        size = path.stat().st_size
        if size > MAX_TEXT_SCAN_BYTES:
            raise ColabWorkflowError("text artifact is too large for the bounded secret scan")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ColabWorkflowError("unable to scan a bundle input") from exc
        if _HF_TOKEN.search(payload):
            raise ColabWorkflowError("artifact text contains a Hugging Face token-like value")


def _copy_bundle_file(
    source: Path, destination: Path, *, relative_name: str, require_text: bool
) -> None:
    _safe_relative_path(relative_name, label="bundle destination")
    _validate_upload_source(source, relative_name=relative_name, require_text=require_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _validate_upload_source(destination, relative_name=relative_name, require_text=require_text)


def _hash_tree(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        name: sha256_file(path)
        for name, path in sorted(_walk_regular_files(root).items())
        if name not in excluded
    }


def write_evidence_index(root: Path) -> tuple[Path, dict[str, str]]:
    """Write deterministic standard SHA-256 lines for every other bundle file."""

    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise ColabWorkflowError("evidence bundle root may not be a symlink")
    bundle_root = supplied_root.resolve(strict=True)
    index_path = bundle_root / EVIDENCE_INDEX_NAME
    if index_path.exists() or index_path.is_symlink():
        raise ColabWorkflowError("evidence index already exists")
    hashes = _hash_tree(bundle_root, exclude=(EVIDENCE_INDEX_NAME,))
    lines = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    index_path.write_text(lines, encoding="utf-8", newline="\n")
    return index_path, hashes


def verify_evidence_index(root: Path) -> Mapping[str, str]:
    """Re-hash a local or downloaded bundle against its evidence index."""

    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise ColabWorkflowError("evidence bundle root may not be a symlink")
    bundle_root = supplied_root.resolve(strict=True)
    index_path = bundle_root / EVIDENCE_INDEX_NAME
    if index_path.is_symlink() or not index_path.is_file():
        raise ColabWorkflowError("evidence index is missing or is not a regular file")
    declared: dict[str, str] = {}
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ColabWorkflowError("unable to read the evidence index") from exc
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ColabWorkflowError("evidence index contains an invalid line")
        digest, name = line[:64], line[66:]
        relative = _safe_relative_path(name, label="evidence index path").as_posix()
        if not _SHA256.fullmatch(digest) or relative in declared:
            raise ColabWorkflowError("evidence index contains an invalid or duplicate entry")
        declared[relative] = digest
    actual = _hash_tree(bundle_root, exclude=(EVIDENCE_INDEX_NAME,))
    if actual != dict(sorted(declared.items())):
        raise ColabWorkflowError("evidence bundle does not match its SHA-256 index")
    return actual


def build_artifact_bundle(
    training: ValidatedTrainingRun,
    destination: Path,
    *,
    environment_files: Mapping[str, Path],
    evaluation_files: Mapping[str, Path],
) -> ArtifactBundle:
    """Atomically build an upload bundle from explicit, privacy-checked inputs."""

    run_id = validate_run_id(training.manifest.experiment_name)
    target = Path(destination)
    if not target.is_absolute():
        raise ColabWorkflowError("artifact bundle destination must be absolute")
    target = target.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise ColabWorkflowError("artifact bundle destination already exists")
    if target == training.run_directory or target.is_relative_to(training.run_directory):
        raise ColabWorkflowError("artifact bundle must be outside the immutable training run")
    target.parent.mkdir(parents=True, exist_ok=True)

    if sha256_file(training.manifest_path) != training.manifest_sha256:
        raise ColabWorkflowError("training manifest changed after validation")

    current_hashes = _declared_artifact_hashes(
        training.run_directory,
        training.manifest_path,
        training.manifest.artifacts_sha256,
    )
    if current_hashes != dict(training.artifact_hashes):
        raise ColabWorkflowError("training run changed after manifest validation")

    named: list[tuple[Path, str, bool]] = []
    training_prefix = f"training/{run_id}"
    named.append((training.manifest_path, f"{training_prefix}/manifest.json", True))
    for relative in sorted(current_hashes):
        named.append(
            (
                training.run_directory / relative,
                f"{training_prefix}/{relative}",
                False,
            )
        )
    for name, source in sorted(environment_files.items()):
        safe_name = _safe_relative_path(name, label="environment artifact name").as_posix()
        named.append((Path(source), f"environment/{safe_name}", True))
    for name, source in sorted(evaluation_files.items()):
        safe_name = _safe_relative_path(name, label="evaluation artifact name").as_posix()
        named.append((Path(source), f"evaluations/{safe_name}", True))

    casefold_names: set[str] = set()
    for _, relative, _ in named:
        folded = relative.casefold()
        if folded in casefold_names:
            raise ColabWorkflowError("bundle inputs contain a case-insensitive path collision")
        casefold_names.add(folded)

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for source, relative, require_text in named:
            _copy_bundle_file(
                source,
                staging / PurePosixPath(relative),
                relative_name=relative,
                require_text=require_text,
            )
        copied_manifest = staging / training_prefix / "manifest.json"
        if sha256_file(copied_manifest) != training.manifest_sha256:
            raise ColabWorkflowError(
                "copied training manifest does not match the validated manifest"
            )
        for relative, digest in current_hashes.items():
            copied = staging / training_prefix / PurePosixPath(relative)
            if sha256_file(copied) != digest:
                raise ColabWorkflowError("copied training artifact does not match its manifest")
        index_path, hashes = write_evidence_index(staging)
        expected_files = {relative for _, relative, _ in named} | {EVIDENCE_INDEX_NAME}
        if set(_walk_regular_files(staging)) != expected_files:
            raise ColabWorkflowError(
                "artifact bundle contains a file outside the explicit allowlist"
            )
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    final_index = target / index_path.relative_to(staging)
    verify_evidence_index(target)
    return ArtifactBundle(root=target, index_path=final_index, files_sha256=hashes)


def _read_evaluation(path: Path) -> ModelEvaluationArtifact:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ColabWorkflowError("evaluation artifact must be a regular non-symlink file")
    try:
        return ModelEvaluationArtifact.model_validate_json(
            candidate.read_text(encoding="utf-8"),
            strict=True,
        )
    except Exception as exc:
        raise ColabWorkflowError("evaluation artifact failed strict schema validation") from exc


def validate_paired_validation_evaluations(
    base_path: Path,
    adapter_path: Path,
    *,
    training: ValidatedTrainingRun,
    expected_case_count: int = EXPECTED_VALIDATION_CASES,
) -> ValidatedEvaluationPair:
    """Validate paired, complete base/adapter validation evidence and provenance."""

    if expected_case_count < 1:
        raise ColabWorkflowError("expected validation case count must be positive")
    if sha256_file(training.manifest_path) != training.manifest_sha256:
        raise ColabWorkflowError("training manifest changed after validation")
    base = _read_evaluation(base_path)
    adapter = _read_evaluation(adapter_path)
    for artifact, backend in ((base, "hf-base"), (adapter, "hf-adapter")):
        if artifact.backend != backend or artifact.result_status != "measured":
            raise ColabWorkflowError(f"expected a measured {backend} evaluation artifact")
        if artifact.split != "validation" or artifact.test_lock_sha256 is not None:
            raise ColabWorkflowError(
                "paired evidence must use validation, never the locked test split"
            )
        if (
            artifact.partial_run
            or artifact.full_split_count != expected_case_count
            or artifact.evaluated_count != expected_case_count
        ):
            raise ColabWorkflowError("evaluation artifact does not cover the full validation split")
        if artifact.operations.operational_errors != 0:
            raise ColabWorkflowError("evaluation artifact contains operational model errors")
        if not artifact.observed_model_versions:
            raise ColabWorkflowError("evaluation artifact records no observed model version")
        if artifact.model_configuration.get("model_name_or_path") != QWEN_MODEL_ID:
            raise ColabWorkflowError("evaluation artifact used an unexpected base model")
        if artifact.model_configuration.get("revision") != QWEN_REVISION:
            raise ColabWorkflowError("evaluation artifact used an unexpected model revision")
        if artifact.model_configuration.get("trust_remote_code") is not False:
            raise ColabWorkflowError("evaluation artifact did not disable remote model code")
        device = artifact.model_configuration.get("device")
        if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device):
            raise ColabWorkflowError("evaluation artifact does not record a CUDA device")

    comparable_fields = (
        "dataset_name",
        "dataset_version",
        "split_file_sha256",
        "case_ids_sha256",
        "result_schema_sha256",
        "prompt_version",
        "prompt_sha256",
    )
    for field in comparable_fields:
        if getattr(base, field) != getattr(adapter, field):
            raise ColabWorkflowError(f"base and adapter evaluations differ in {field}")
    if base.model_id != adapter.model_id:
        raise ColabWorkflowError("base and adapter evaluations use different serving identities")

    provenance = adapter.training_run
    if provenance is None:
        raise ColabWorkflowError("adapter evaluation schema lacks training-run provenance")
    expected_adapter_hashes = {
        name.removeprefix("adapter/"): digest
        for name, digest in training.artifact_hashes.items()
        if name.startswith("adapter/")
    }
    if not expected_adapter_hashes:
        raise ColabWorkflowError("training manifest contains no adapter artifacts")
    if (
        provenance.training_manifest_sha256 != training.manifest_sha256
        or provenance.experiment_name != training.manifest.experiment_name
        or provenance.resolved_model_revision != training.manifest.resolved_model_revision
        or provenance.resolved_tokenizer_revision != training.manifest.resolved_tokenizer_revision
        or provenance.adapter_artifacts_sha256 != expected_adapter_hashes
    ):
        raise ColabWorkflowError("adapter evaluation is not linked to the validated training run")
    return ValidatedEvaluationPair(base=base, adapter=adapter)


def validate_inputs(*, source_commit: str, artifact_repo_id: str, run_id: str) -> dict[str, str]:
    """Validate the three user-controlled notebook identifiers at once."""

    return {
        "source_commit": validate_commit_sha(source_commit),
        "artifact_repo_id": validate_hf_repo_id(artifact_repo_id),
        "run_id": validate_run_id(run_id),
    }


def derive_cloud_config(
    source_config: Path,
    destination: Path,
    *,
    checkout: VerifiedCheckout,
    output_root: Path,
    run_id: str,
) -> Path:
    """Derive and exclusively create the notebook's concrete cloud YAML file."""

    derived = derive_cloud_training_yaml(
        source_config,
        checkout=checkout,
        output_root=output_root,
        run_id=run_id,
    )
    path = Path(destination)
    if not path.is_absolute() or path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ColabWorkflowError("derived config destination must be an absolute YAML path")
    path = path.resolve(strict=False)
    if path.exists() or path.is_symlink():
        raise ColabWorkflowError("derived config destination already exists")
    if path.is_relative_to(checkout.root):
        raise ColabWorkflowError("derived config must be written outside the Git checkout")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as sink:
            sink.write(derived.yaml_text)
            sink.flush()
            os.fsync(sink.fileno())
    except OSError as exc:
        raise ColabWorkflowError("unable to create the derived cloud config") from exc
    return path


def verify_training_manifest(
    manifest_path: Path,
    *,
    source_commit: str,
    run_id: str,
) -> dict[str, object]:
    """Notebook-facing training verification with a serialization-friendly summary."""

    training = validate_completed_training_run(
        manifest_path,
        expected_source_sha=source_commit,
        expected_run_id=run_id,
    )
    return {
        "manifest_path": training.manifest_path,
        "manifest_sha256": training.manifest_sha256,
        "run_directory": training.run_directory,
        "artifact_hashes": dict(training.artifact_hashes),
        "experiment_name": training.manifest.experiment_name,
        "source_commit": training.manifest.git.commit_sha,
        "model_revision": training.manifest.resolved_model_revision,
        "device": training.manifest.hardware.device,
    }


def prepare_evidence_bundle(
    manifest_path: Path,
    destination: Path,
    *,
    source_commit: str,
    run_id: str,
    environment_files: Mapping[str, Path],
    evaluation_files: Mapping[str, Path],
) -> Path:
    """Notebook-facing atomic bundle builder that revalidates training first."""

    training = validate_completed_training_run(
        manifest_path,
        expected_source_sha=source_commit,
        expected_run_id=run_id,
    )
    return build_artifact_bundle(
        training,
        destination,
        environment_files=environment_files,
        evaluation_files=evaluation_files,
    ).root


def validate_paired_evaluations(
    base_path: Path,
    adapter_path: Path,
    *,
    training_manifest_path: Path,
    source_commit: str,
    run_id: str,
    expected_case_count: int = EXPECTED_VALIDATION_CASES,
) -> dict[str, object]:
    """Notebook-facing paired-evidence verification and concise result summary."""

    training = validate_completed_training_run(
        training_manifest_path,
        expected_source_sha=source_commit,
        expected_run_id=run_id,
    )
    pair = validate_paired_validation_evaluations(
        base_path,
        adapter_path,
        training=training,
        expected_case_count=expected_case_count,
    )
    return {
        "base_path": Path(base_path).resolve(strict=True),
        "adapter_path": Path(adapter_path).resolve(strict=True),
        "validation_case_count": pair.base.evaluated_count,
        "split_file_sha256": pair.base.split_file_sha256,
        "case_ids_sha256": pair.base.case_ids_sha256,
        "training_manifest_sha256": training.manifest_sha256,
        "base_metrics": pair.base.evaluation.aggregate.model_dump(mode="json"),
        "adapter_metrics": pair.adapter.evaluation.aggregate.model_dump(mode="json"),
    }


def verify_downloaded_bundle(root: Path) -> dict[str, str]:
    """Notebook-facing exact verification for a commit-pinned downloaded bundle."""

    return dict(verify_evidence_index(root))
