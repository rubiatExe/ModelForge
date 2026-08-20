"""Purpose-aware dataset loading with manifest and test-lock enforcement."""

from __future__ import annotations

from pathlib import Path

from modelforge.datasets.io import read_json_model, read_jsonl, verify_sha256
from modelforge.datasets.validation import (
    DatasetValidationError,
    examples_fingerprint_digest,
    label_distribution,
    robustness_distribution,
    validate_no_conflicting_labels,
    validate_no_duplicates,
)
from modelforge.schemas import (
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    LabeledExample,
    TestSetLock,
)


class PurposeViolationError(DatasetValidationError):
    pass


class ManifestMismatchError(DatasetValidationError):
    pass


_PURPOSE_SPLITS: dict[DatasetPurpose, frozenset[DatasetSplit]] = {
    DatasetPurpose.TRAINING: frozenset({DatasetSplit.TRAIN}),
    DatasetPurpose.VALIDATION: frozenset({DatasetSplit.VALIDATION}),
    DatasetPurpose.TEST_EVALUATION: frozenset({DatasetSplit.TEST}),
    DatasetPurpose.ADVERSARIAL_EVALUATION: frozenset({DatasetSplit.ADVERSARIAL}),
    DatasetPurpose.ANALYSIS: frozenset(DatasetSplit),
}


def _safe_artifact(dataset_directory: Path, relative_name: str) -> Path:
    root = dataset_directory.resolve()
    unresolved = root / relative_name
    if unresolved.is_symlink():
        raise ManifestMismatchError(f"artifact may not be a symlink: {unresolved}")
    candidate = unresolved.resolve()
    if candidate.parent != root:
        raise ManifestMismatchError(f"artifact must be directly inside {root}: {relative_name}")
    return candidate


def load_manifest(dataset_directory: str | Path) -> DatasetManifest:
    root = Path(dataset_directory)
    return read_json_model(_safe_artifact(root, "manifest.json"), DatasetManifest)


def load_test_lock(dataset_directory: str | Path, manifest: DatasetManifest) -> TestSetLock:
    root = Path(dataset_directory)
    lock = read_json_model(_safe_artifact(root, manifest.test_lock_file), TestSetLock)
    if (
        lock.dataset_name != manifest.dataset_name
        or lock.dataset_version != manifest.dataset_version
    ):
        raise ManifestMismatchError("test lock dataset identity does not match manifest")
    return lock


def verify_test_lock(dataset_directory: str | Path) -> TestSetLock:
    root = Path(dataset_directory)
    manifest = load_manifest(root)
    lock = load_test_lock(root, manifest)
    test_manifest = next(item for item in manifest.splits if item.split is DatasetSplit.TEST)
    if (
        lock.file != test_manifest.file
        or lock.count != test_manifest.count
        or lock.sha256 != test_manifest.sha256
        or lock.fingerprint_sha256 != test_manifest.fingerprint_sha256
    ):
        raise ManifestMismatchError("test lock does not match the test split manifest")
    verify_sha256(_safe_artifact(root, lock.file), lock.sha256)
    return lock


def load_dataset_split(
    dataset_directory: str | Path,
    split: DatasetSplit,
    *,
    purpose: DatasetPurpose,
) -> list[LabeledExample]:
    """Load one declared split; purpose must explicitly authorize that split."""

    if split not in _PURPOSE_SPLITS[purpose]:
        raise PurposeViolationError(
            f"purpose {purpose.value!r} may not load the {split.value!r} split"
        )

    root = Path(dataset_directory)
    manifest = load_manifest(root)
    split_manifest = next(item for item in manifest.splits if item.split is split)
    artifact = _safe_artifact(root, split_manifest.file)
    verify_sha256(artifact, split_manifest.sha256)
    if split is DatasetSplit.TEST:
        verify_test_lock(root)

    examples = read_jsonl(artifact, LabeledExample)
    if any(example.split is not split for example in examples):
        raise ManifestMismatchError(f"{artifact} contains a record from another split")
    validate_no_conflicting_labels(examples)
    validate_no_duplicates(examples)

    if len(examples) != split_manifest.count:
        raise ManifestMismatchError(
            f"{split.value} count mismatch: expected {split_manifest.count}, got {len(examples)}"
        )
    if examples_fingerprint_digest(examples) != split_manifest.fingerprint_sha256:
        raise ManifestMismatchError(f"{split.value} fingerprint digest mismatch")
    if label_distribution(examples) != split_manifest.label_distribution:
        raise ManifestMismatchError(f"{split.value} label distribution mismatch")
    if split is DatasetSplit.ADVERSARIAL:
        if robustness_distribution(examples) != split_manifest.robustness_distribution:
            raise ManifestMismatchError("adversarial robustness distribution mismatch")
    return examples


def load_training_examples(dataset_directory: str | Path) -> list[LabeledExample]:
    return load_dataset_split(
        dataset_directory,
        DatasetSplit.TRAIN,
        purpose=DatasetPurpose.TRAINING,
    )


def load_validation_examples(dataset_directory: str | Path) -> list[LabeledExample]:
    return load_dataset_split(
        dataset_directory,
        DatasetSplit.VALIDATION,
        purpose=DatasetPurpose.VALIDATION,
    )


def load_test_examples(dataset_directory: str | Path) -> list[LabeledExample]:
    return load_dataset_split(
        dataset_directory,
        DatasetSplit.TEST,
        purpose=DatasetPurpose.TEST_EVALUATION,
    )


def load_adversarial_examples(dataset_directory: str | Path) -> list[LabeledExample]:
    return load_dataset_split(
        dataset_directory,
        DatasetSplit.ADVERSARIAL,
        purpose=DatasetPurpose.ADVERSARIAL_EVALUATION,
    )
