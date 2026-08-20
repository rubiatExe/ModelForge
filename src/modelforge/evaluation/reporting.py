"""Privacy-conscious Markdown summaries for evaluation artifacts."""

from __future__ import annotations

from modelforge.evaluation.calibration import CalibrationReport
from modelforge.evaluation.failures import FailureTaxonomyReport
from modelforge.evaluation.judge import EvidenceComparisonReport, JudgeValidationReport
from modelforge.evaluation.models import EvaluationReport, EvaluationSliceMetrics
from modelforge.evaluation.regression import RegressionGateResult
from modelforge.evaluation.routing import RoutingSweepReport


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _evaluation_row(name: str, metrics: EvaluationSliceMetrics) -> str:
    return (
        f"| {_cell(name)} | {metrics.total_cases} | {metrics.error_count} | "
        f"{_pct(metrics.schema_validity_rate)} | {_pct(metrics.exact_match_rate)} | "
        f"{_number(metrics.issue.macro_f1)} | {_pct(metrics.routing_accuracy)} | "
        f"{_pct(metrics.severity_accuracy)} | {_pct(metrics.scope_accuracy)} | "
        f"{_pct(metrics.action_accuracy)} |"
    )


def evaluation_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Evaluator: `{_cell(report.evaluator_version)}`  ",
        f"Schema: `{_cell(report.schema_name)}`",
        "",
        "| population | cases | errors | schema valid | exact match | issue macro F1 | routing | severity | scope | action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _evaluation_row("overall", report.aggregate),
    ]
    lines.extend(_evaluation_row(name, metrics) for name, metrics in sorted(report.slices.items()))
    lines.extend(
        [
            "",
            "Errors are operational failures and are excluded from wrong-prediction denominators.",
        ]
    )
    return "\n".join(lines) + "\n"


def calibration_markdown(report: CalibrationReport) -> str:
    lines = [
        "# Calibration and risk/coverage",
        "",
        f"Scored predictions: {report.scored_count}  ",
        f"Missing scores: {report.missing_score_count}  ",
        f"ECE diagnostic: {_number(report.ece)}",
        "",
        "| accept score >= | coverage | selective risk | selective accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for point in report.risk_coverage:
        lines.append(
            f"| {point.threshold:.3f} | {_pct(point.coverage)} | {_pct(point.risk)} | "
            f"{_pct(point.selective_accuracy)} |"
        )
    lines.extend(["", "The routing score is a score, not an asserted probability."])
    return "\n".join(lines) + "\n"


def failure_taxonomy_markdown(report: FailureTaxonomyReport) -> str:
    lines = [
        "# Failure taxonomy",
        "",
        f"Failed cases: {report.failed_cases}/{report.evaluated_cases}",
        "",
        "| category | occurrences | representative case IDs |",
        "|---|---:|---|",
    ]
    for bucket in report.buckets:
        examples = ", ".join(_cell(example.case_id) for example in bucket.examples) or "—"
        lines.append(f"| {_cell(bucket.category.value)} | {bucket.count} | {examples} |")
    return "\n".join(lines) + "\n"


def judge_validation_markdown(report: JudgeValidationReport) -> str:
    lines = [
        "# Judge validation",
        "",
        f"Matched reviews: {report.matched_count}/{report.human_review_count}  ",
        f"Missing judge results: {report.missing_judge_count}  ",
        f"Judge errors: {report.judge_error_count}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| joint agreement | {_pct(report.joint_agreement)} |",
        f"| relevance Cohen kappa | {_number(report.relevance_kappa)} |",
        f"| unsupported-claim Cohen kappa | {_number(report.unsupported_kappa)} |",
        f"| relevance rank correlation | {_number(report.rank_correlation)} |",
        f"| false acceptance rate | {_pct(report.false_accept_rate)} |",
        f"| false rejection rate | {_pct(report.false_reject_rate)} |",
        "",
        "Disagreement case IDs: "
        + (", ".join(_cell(item.case_id) for item in report.disagreements) or "none"),
    ]
    return "\n".join(lines) + "\n"


def evidence_comparison_markdown(report: EvidenceComparisonReport) -> str:
    """Render aligned gold/model/human/judge decisions without ticket text."""

    lines = [
        "# Evidence comparison",
        "",
        f"Judge: `{_cell(report.judge_identity or 'not available')}`",
        "",
        "| case | model status | model exact | literal evidence | human relevant | human unsupported | judge relevant | judge unsupported |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            f"| {_cell(row.case_id)} | {_cell(row.model_status)} | "
            f"{_cell(row.model_exact_match)} | {_cell(row.model_evidence_all_literal)} | "
            f"{_cell(row.human_evidence_relevant)} | {_cell(row.human_unsupported_claim)} | "
            f"{_cell(row.judge_evidence_relevant)} | {_cell(row.judge_unsupported_claim)} |"
        )
    return "\n".join(lines) + "\n"


def regression_gate_markdown(result: RegressionGateResult) -> str:
    lines = [
        "# Regression gate",
        "",
        f"Policy: `{_cell(result.policy_name)}@{_cell(result.policy_version)}`  ",
        f"Status: **{result.status.value.upper()}**",
    ]
    if result.violations:
        lines.extend(["", "| code | metric | slice | detail |", "|---|---|---|---|"])
        for violation in result.violations:
            lines.append(
                f"| {_cell(violation.code)} | {_cell(violation.metric)} | "
                f"{_cell(violation.slice or 'overall')} | {_cell(violation.detail)} |"
            )
    return "\n".join(lines) + "\n"


def routing_sweep_markdown(report: RoutingSweepReport) -> str:
    lines = [
        "# Routing threshold sweep",
        "",
        f"Calibration split: `{_cell(report.split)}`  ",
        f"Rule: {_cell(report.route_rule)}",
        "",
        "| threshold | issue macro F1 | frontier share | calls / 100 | avg latency ms | p95 latency ms | avg cost USD | overconfident errors / 100 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in report.results:
        metrics = point.aggregate
        lines.append(
            f"| {point.threshold:.2f} | {_number(metrics.issue.macro_f1)} | "
            f"{_pct(metrics.frontier_rate)} | {_number(metrics.frontier_calls_per_100, 1)} | "
            f"{_number(metrics.average_latency_ms, 1)} | {_number(metrics.p95_latency_ms, 1)} | "
            f"{_number(metrics.average_cost_usd, 6)} | "
            f"{_number(metrics.overconfident_errors_per_100, 1)} |"
        )
    return "\n".join(lines) + "\n"
