"""Immutable experiment identity, environment capture, and loss diagnostics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from modelforge.schemas._base import StrictBaseModel


class MetricPoint(StrictBaseModel):
    step: int = Field(ge=0)
    epoch: float | None = Field(default=None, ge=0.0)
    value: float


class LossHistory(StrictBaseModel):
    training: tuple[MetricPoint, ...] = ()
    validation: tuple[MetricPoint, ...] = ()


class OverfittingSignal(StrictBaseModel):
    detected: bool
    reason: str
    best_validation_loss: float | None = None
    final_validation_loss: float | None = None


class GitMetadata(StrictBaseModel):
    commit_sha: str | None = None
    dirty: bool | None = None


class HardwareMetadata(StrictBaseModel):
    platform: str
    python_version: str
    device: str
    precision: str
    accelerator_name: str | None = None


class ParameterCounts(StrictBaseModel):
    trainable: int = Field(ge=0)
    total: int = Field(ge=0)
    trainable_fraction: float = Field(ge=0.0, le=1.0)


class ExperimentManifest(StrictBaseModel):
    manifest_version: Literal["1.0"] = "1.0"
    status: Literal["complete"] = "complete"
    experiment_name: str
    completed_at: datetime
    config: dict[str, Any]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: dict[str, str]
    prompt_version: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_model_revision: str
    resolved_model_revision: str
    requested_tokenizer_revision: str
    resolved_tokenizer_revision: str
    seed: int = Field(ge=0)
    parameter_counts: ParameterCounts
    hardware: HardwareMetadata
    git: GitMetadata
    packages: dict[str, str]
    losses: LossHistory
    overfitting: OverfittingSignal
    trainer_metrics: dict[str, float | int | str | bool | None]
    artifacts_sha256: dict[str, str]
    truncated_training_examples: int = Field(ge=0)
    truncated_validation_examples: int = Field(ge=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "pydantic",
        "PyYAML",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def collect_git_metadata(project_root: Path) -> GitMetadata:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return GitMetadata(commit_sha=commit, dirty=dirty)
    except (OSError, subprocess.SubprocessError):
        return GitMetadata()


def collect_hardware_metadata(torch: Any, *, device: str, precision: str) -> HardwareMetadata:
    accelerator_name: str | None = None
    try:
        if device.startswith("cuda"):
            accelerator_name = str(torch.cuda.get_device_name(device))
        elif device == "mps":
            accelerator_name = "Apple Metal Performance Shaders"
    except Exception:
        accelerator_name = None
    return HardwareMetadata(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        device=device,
        precision=precision,
        accelerator_name=accelerator_name,
    )


def extract_loss_history(log_history: Sequence[Mapping[str, Any]]) -> LossHistory:
    training: list[MetricPoint] = []
    validation: list[MetricPoint] = []
    for entry in log_history:
        step = int(entry.get("step", 0))
        epoch_raw = entry.get("epoch")
        epoch = float(epoch_raw) if epoch_raw is not None else None
        if "loss" in entry:
            training.append(MetricPoint(step=step, epoch=epoch, value=float(entry["loss"])))
        if "eval_loss" in entry:
            validation.append(
                MetricPoint(step=step, epoch=epoch, value=float(entry["eval_loss"]))
            )
    return LossHistory(training=tuple(training), validation=tuple(validation))


def detect_overfitting(
    losses: LossHistory,
    *,
    relative_validation_increase: float = 0.05,
) -> OverfittingSignal:
    validation = [point.value for point in losses.validation]
    training = [point.value for point in losses.training]
    if len(validation) < 2 or len(training) < 2:
        return OverfittingSignal(
            detected=False,
            reason="insufficient loss history for an overfitting signal",
            best_validation_loss=min(validation) if validation else None,
            final_validation_loss=validation[-1] if validation else None,
        )
    best = min(validation)
    final = validation[-1]
    train_improved = training[-1] < training[0]
    validation_worsened = final > best * (1.0 + relative_validation_increase)
    detected = train_improved and validation_worsened
    return OverfittingSignal(
        detected=detected,
        reason=(
            "training loss fell while validation loss moved materially above its best value"
            if detected
            else "loss curves do not meet the configured overfitting heuristic"
        ),
        best_validation_loss=best,
        final_validation_loss=final,
    )


def artifact_hashes(root: Path, *, exclude: Sequence[Path] = ()) -> dict[str, str]:
    root = root.resolve()
    excluded = {item.resolve() for item in exclude}
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def write_manifest_once(path: Path, manifest: ExperimentManifest) -> Path:
    """Atomically create a manifest and refuse to rewrite experiment history."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"experiment manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(manifest.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # The run directory is new and checked by the trainer, so replace is
        # atomic without authorizing replacement of a prior manifest.
        if path.exists():
            raise FileExistsError(f"experiment manifest already exists: {path}")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def utc_now() -> datetime:
    return datetime.now(UTC)

