"""Apply a strict regression policy to two comparable measured evaluation artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from modelforge.datasets import sha256_file
from modelforge.evaluation import (
    RegressionGateResult,
    RegressionPolicy,
    RunIdentity,
    gate_regression,
    read_json_artifact,
    write_json_artifact,
)
from modelforge.evaluation.regression import case_ids_sha256
from modelforge.experiments._artifacts import (
    canonical_sha256,
    require_available_outputs,
)
from modelforge.experiments.run_evaluation import ModelEvaluationArtifact
from modelforge.schemas._base import StrictBaseModel

ArtifactT = TypeVar("ArtifactT", bound=BaseModel)


class RegressionSource(StrictBaseModel):
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity: RunIdentity


class RegressionGateArtifact(StrictBaseModel):
    artifact_version: Literal["1.0"] = "1.0"
    result_status: Literal["derived_from_measured_sources"] = (
        "derived_from_measured_sources"
    )
    completed_at: datetime
    baseline: RegressionSource
    candidate: RegressionSource
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy: RegressionPolicy
    gate: RegressionGateResult
    privacy: dict[str, str | bool]


def _read_bounded(path: Path, model_type: type[ArtifactT]) -> ArtifactT:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError(f"artifact exceeds 256 MiB: {path}")
    return read_json_artifact(path, model_type)


def _identity(artifact: ModelEvaluationArtifact) -> RunIdentity:
    if artifact.result_status != "measured":
        raise ValueError(
            f"regression sources must be measured, not {artifact.result_status!r}"
        )
    if artifact.evaluated_count != len(artifact.evaluation.items):
        raise ValueError("evaluation artifact count does not match report items")
    report_digest = case_ids_sha256(artifact.evaluation)
    if artifact.case_ids_sha256 != report_digest:
        raise ValueError("evaluation artifact case digest does not match report items")
    prompt_sha = artifact.prompt_sha256 or canonical_sha256({"prompt": None})
    return RunIdentity(
        run_id=artifact.experiment_id,
        dataset_name=artifact.dataset_name,
        dataset_version=artifact.dataset_version,
        dataset_sha256=artifact.split_file_sha256,
        split=artifact.split,
        case_ids_sha256=report_digest,
        case_count=artifact.evaluated_count,
        evaluator_version=artifact.evaluation.evaluator_version,
        schema_version=artifact.result_schema_sha256,
        judge_identity=None,
        model_identity=(
            f"{artifact.model_id}@"
            f"{canonical_sha256(list(artifact.observed_model_versions))[:16]}"
        ),
        prompt_sha256=prompt_sha,
        generation_config_sha256=canonical_sha256(artifact.model_configuration),
        completed=(
            not artifact.partial_run
            and artifact.evaluated_count == artifact.full_split_count
        ),
    )


def run_regression_gate(
    *,
    baseline_path: Path,
    candidate_path: Path,
    policy_path: Path,
    output_path: Path,
) -> RegressionGateArtifact:
    """Run a gate without mutating or re-evaluating either cached source run."""

    require_available_outputs((output_path,), overwrite=False)
    if baseline_path.resolve() == candidate_path.resolve():
        raise ValueError("baseline and candidate paths must be distinct")
    baseline = _read_bounded(baseline_path, ModelEvaluationArtifact)
    candidate = _read_bounded(candidate_path, ModelEvaluationArtifact)
    policy = _read_bounded(policy_path, RegressionPolicy)
    baseline_identity = _identity(baseline)
    candidate_identity = _identity(candidate)
    result = gate_regression(
        baseline.evaluation,
        candidate.evaluation,
        baseline_identity=baseline_identity,
        candidate_identity=candidate_identity,
        policy=policy,
    )
    artifact = RegressionGateArtifact(
        completed_at=datetime.now(UTC),
        baseline=RegressionSource(
            artifact_sha256=sha256_file(baseline_path),
            identity=baseline_identity,
        ),
        candidate=RegressionSource(
            artifact_sha256=sha256_file(candidate_path),
            identity=candidate_identity,
        ),
        policy_sha256=sha256_file(policy_path),
        policy=policy,
        gate=result,
        privacy={
            "classification": "derived_evaluation_artifact",
            "contains_raw_ticket_text": False,
            "contains_predictions_or_gold_labels": False,
            "contains_case_ids": False,
        },
    )
    write_json_artifact(output_path, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_regression_gate(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        policy_path=args.policy,
        output_path=args.output,
    )
    print(
        f"regression gate {artifact.gate.policy_name}@{artifact.gate.policy_version}: "
        f"{artifact.gate.status.value} ({len(artifact.gate.violations)} violations)"
    )
    return artifact.gate.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
