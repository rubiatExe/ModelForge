"""Run comparability checks and policy-based regression gates."""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import Field, field_validator

from modelforge.evaluation.models import EvaluationReport, EvaluationSliceMetrics, FrozenModel


def case_ids_sha256(report: EvaluationReport) -> str:
    payload = "".join(f"{case_id}\n" for case_id in sorted(item.case_id for item in report.items))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunIdentity(FrozenModel):
    """Immutable facts needed to decide whether two reports are comparable."""

    run_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str = Field(min_length=1)
    case_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=0)
    evaluator_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    judge_identity: str | None = None
    model_identity: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed: bool = True


class ComparabilityResult(FrozenModel):
    comparable: bool
    reasons: tuple[str, ...]


DEFAULT_COMPARABILITY_FIELDS = (
    "dataset_name",
    "dataset_version",
    "dataset_sha256",
    "split",
    "case_ids_sha256",
    "case_count",
    "evaluator_version",
    "schema_version",
    "judge_identity",
)


def compare_run_identities(
    baseline: RunIdentity,
    candidate: RunIdentity,
    *,
    required_equal_fields: tuple[str, ...] = DEFAULT_COMPARABILITY_FIELDS,
) -> ComparabilityResult:
    """Prove comparability before looking at metric deltas.

    Model, prompt, and generation identities are recorded but intentionally not
    required to match: those are normal experimental variables.  Dataset item
    population, evaluator, schema, and judge/rubric identity are controls.
    """

    unknown = [field for field in required_equal_fields if field not in RunIdentity.model_fields]
    if unknown:
        raise ValueError(f"unknown RunIdentity comparability fields: {unknown}")
    reasons: list[str] = []
    if not baseline.completed:
        reasons.append("baseline run is incomplete")
    if not candidate.completed:
        reasons.append("candidate run is incomplete")
    for field in required_equal_fields:
        left = getattr(baseline, field)
        right = getattr(candidate, field)
        if left != right:
            reasons.append(f"{field} differs: baseline={left!r}, candidate={right!r}")
    return ComparabilityResult(comparable=not reasons, reasons=tuple(reasons))


class MetricFloor(FrozenModel):
    max_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_schema_validity_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_exact_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_issue_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    min_issue_macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    field_accuracy_floors: dict[str, float] = Field(default_factory=dict)
    per_class_f1_floors: dict[str, float] = Field(default_factory=dict)

    @field_validator("field_accuracy_floors", "per_class_f1_floors")
    @classmethod
    def _bounded_floors(cls, values: dict[str, float]) -> dict[str, float]:
        invalid = {name: value for name, value in values.items() if not 0.0 <= value <= 1.0}
        if invalid:
            raise ValueError(f"metric floors must be in [0, 1]: {invalid}")
        return values


class RegressionPolicy(FrozenModel):
    policy_name: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    overall: MetricFloor
    slice_floors: dict[str, MetricFloor] = Field(default_factory=dict)
    max_exact_match_drop: float | None = Field(default=None, ge=0.0, le=1.0)
    max_issue_macro_f1_drop: float | None = Field(default=None, ge=0.0, le=1.0)
    max_per_class_f1_drop: float | None = Field(default=None, ge=0.0, le=1.0)
    required_comparability_fields: tuple[str, ...] = DEFAULT_COMPARABILITY_FIELDS
    fail_on_missing_metric: bool = True


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCOMPARABLE = "incomparable"


class GateViolation(FrozenModel):
    code: str
    metric: str
    detail: str
    actual: float | None = None
    threshold: float | None = None
    slice: str | None = None


class RegressionGateResult(FrozenModel):
    policy_name: str
    policy_version: str
    status: GateStatus
    comparable: bool
    violations: tuple[GateViolation, ...]

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


def _minimum(
    violations: list[GateViolation],
    *,
    metric: str,
    actual: float | None,
    threshold: float | None,
    slice_name: str | None,
    missing_fails: bool,
) -> None:
    if threshold is None:
        return
    if actual is None:
        if missing_fails:
            violations.append(
                GateViolation(
                    code="metric_missing",
                    metric=metric,
                    detail=f"{metric} is unavailable",
                    threshold=threshold,
                    slice=slice_name,
                )
            )
    elif actual < threshold:
        violations.append(
            GateViolation(
                code="floor_failed",
                metric=metric,
                detail=f"{metric}={actual:.6f} is below {threshold:.6f}",
                actual=actual,
                threshold=threshold,
                slice=slice_name,
            )
        )


def _maximum(
    violations: list[GateViolation],
    *,
    metric: str,
    actual: float | None,
    threshold: float | None,
    slice_name: str | None,
    missing_fails: bool,
) -> None:
    if threshold is None:
        return
    if actual is None:
        if missing_fails:
            violations.append(
                GateViolation(
                    code="metric_missing",
                    metric=metric,
                    detail=f"{metric} is unavailable",
                    threshold=threshold,
                    slice=slice_name,
                )
            )
    elif actual > threshold:
        violations.append(
            GateViolation(
                code="ceiling_failed",
                metric=metric,
                detail=f"{metric}={actual:.6f} exceeds {threshold:.6f}",
                actual=actual,
                threshold=threshold,
                slice=slice_name,
            )
        )


def _apply_floor(
    metrics: EvaluationSliceMetrics,
    floor: MetricFloor,
    *,
    slice_name: str | None,
    missing_fails: bool,
) -> list[GateViolation]:
    violations: list[GateViolation] = []
    _maximum(
        violations,
        metric="error_rate",
        actual=metrics.error_rate,
        threshold=floor.max_error_rate,
        slice_name=slice_name,
        missing_fails=missing_fails,
    )
    for metric, actual, threshold in (
        ("schema_validity_rate", metrics.schema_validity_rate, floor.min_schema_validity_rate),
        ("exact_match_rate", metrics.exact_match_rate, floor.min_exact_match_rate),
        ("issue_accuracy", metrics.issue.accuracy, floor.min_issue_accuracy),
        ("issue_macro_f1", metrics.issue.macro_f1, floor.min_issue_macro_f1),
    ):
        _minimum(
            violations,
            metric=metric,
            actual=actual,
            threshold=threshold,
            slice_name=slice_name,
            missing_fails=missing_fails,
        )
    for field, threshold in sorted(floor.field_accuracy_floors.items()):
        _minimum(
            violations,
            metric=f"field_accuracy.{field}",
            actual=metrics.field_accuracies.get(field),
            threshold=threshold,
            slice_name=slice_name,
            missing_fails=missing_fails,
        )
    for label, threshold in sorted(floor.per_class_f1_floors.items()):
        per_class = metrics.issue.per_class.get(label)
        _minimum(
            violations,
            metric=f"issue_f1.{label}",
            actual=per_class.f1 if per_class is not None else None,
            threshold=threshold,
            slice_name=slice_name,
            missing_fails=missing_fails,
        )
    return violations


def gate_regression(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    baseline_identity: RunIdentity,
    candidate_identity: RunIdentity,
    policy: RegressionPolicy,
) -> RegressionGateResult:
    """Check comparability first, then absolute and relative quality policy."""

    comparability = compare_run_identities(
        baseline_identity,
        candidate_identity,
        required_equal_fields=policy.required_comparability_fields,
    )
    identity_reasons = list(comparability.reasons)
    for label, identity, report in (
        ("baseline", baseline_identity, baseline),
        ("candidate", candidate_identity, candidate),
    ):
        if identity.case_count != len(report.items):
            identity_reasons.append(
                f"{label} identity case_count={identity.case_count} but report has {len(report.items)} items"
            )
        digest = case_ids_sha256(report)
        if identity.case_ids_sha256 != digest:
            identity_reasons.append(f"{label} case_ids_sha256 does not match report items")
        if identity.evaluator_version != report.evaluator_version:
            identity_reasons.append(f"{label} evaluator identity does not match report")
    if baseline.schema_name != candidate.schema_name:
        identity_reasons.append(
            f"report schema differs: baseline={baseline.schema_name!r}, candidate={candidate.schema_name!r}"
        )
    if identity_reasons:
        violations = tuple(
            GateViolation(code="incomparable", metric="comparability", detail=reason)
            for reason in identity_reasons
        )
        return RegressionGateResult(
            policy_name=policy.policy_name,
            policy_version=policy.policy_version,
            status=GateStatus.INCOMPARABLE,
            comparable=False,
            violations=violations,
        )

    violations = _apply_floor(
        candidate.aggregate,
        policy.overall,
        slice_name=None,
        missing_fails=policy.fail_on_missing_metric,
    )
    for slice_name, floor in sorted(policy.slice_floors.items()):
        metrics = candidate.slices.get(slice_name)
        if metrics is None:
            violations.append(
                GateViolation(
                    code="slice_missing",
                    metric="slice",
                    detail=f"required slice {slice_name!r} is absent",
                    slice=slice_name,
                )
            )
        else:
            violations.extend(
                _apply_floor(
                    metrics,
                    floor,
                    slice_name=slice_name,
                    missing_fails=policy.fail_on_missing_metric,
                )
            )

    relative_metrics = (
        (
            "exact_match_rate",
            baseline.aggregate.exact_match_rate,
            candidate.aggregate.exact_match_rate,
            policy.max_exact_match_drop,
        ),
        (
            "issue_macro_f1",
            baseline.aggregate.issue.macro_f1,
            candidate.aggregate.issue.macro_f1,
            policy.max_issue_macro_f1_drop,
        ),
    )
    for metric, before, after, allowed_drop in relative_metrics:
        if allowed_drop is None:
            continue
        if before is None or after is None:
            if policy.fail_on_missing_metric:
                violations.append(
                    GateViolation(
                        code="metric_missing",
                        metric=metric,
                        detail=f"cannot calculate regression delta for {metric}",
                    )
                )
            continue
        drop = before - after
        if drop > allowed_drop:
            violations.append(
                GateViolation(
                    code="regression",
                    metric=metric,
                    detail=f"{metric} dropped {drop:.6f}; allowed {allowed_drop:.6f}",
                    actual=drop,
                    threshold=allowed_drop,
                )
            )

    if policy.max_per_class_f1_drop is not None:
        labels = sorted(
            set(baseline.aggregate.issue.per_class) | set(candidate.aggregate.issue.per_class)
        )
        for label in labels:
            before = baseline.aggregate.issue.per_class.get(label)
            after = candidate.aggregate.issue.per_class.get(label)
            if before is None or after is None:
                if policy.fail_on_missing_metric:
                    violations.append(
                        GateViolation(
                            code="metric_missing",
                            metric=f"issue_f1.{label}",
                            detail=f"cannot calculate per-class F1 delta for {label}",
                        )
                    )
                continue
            drop = before.f1 - after.f1
            if drop > policy.max_per_class_f1_drop:
                violations.append(
                    GateViolation(
                        code="regression",
                        metric=f"issue_f1.{label}",
                        detail=(
                            f"issue_f1.{label} dropped {drop:.6f}; "
                            f"allowed {policy.max_per_class_f1_drop:.6f}"
                        ),
                        actual=drop,
                        threshold=policy.max_per_class_f1_drop,
                    )
                )

    return RegressionGateResult(
        policy_name=policy.policy_name,
        policy_version=policy.policy_version,
        status=GateStatus.FAIL if violations else GateStatus.PASS,
        comparable=True,
        violations=tuple(violations),
    )
