from __future__ import annotations

import json

import pytest

from modelforge.evaluation.models import PredictionError
from modelforge.evaluation.routing import (
    CachedPrediction,
    CachedRoutingCase,
    evaluate_routing_threshold,
    sweep_routing_thresholds,
    weighted_percentile,
)
from modelforge.schemas import TriageResult


def label(issue_type: str) -> dict:
    return {
        "issue_type": issue_type,
        "severity": "P2",
        "routing_team": "IDENTITY_PLATFORM",
        "affected_scope": "MULTIPLE_USERS",
        "recommended_action": "INVESTIGATE_IDP_OR_SSO_CONFIGURATION",
        "evidence": ["authentication failed"],
    }


def cached(issue_type: str, *, latency_ms: float, cost_usd: float) -> CachedPrediction:
    return CachedPrediction(
        prediction={**label(issue_type), "confidence": 0.9},
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def routing_cases() -> tuple[CachedRoutingCase, ...]:
    sso = "SSO_AUTHENTICATION_FAILURE"
    mfa = "MFA_FAILURE"
    return (
        CachedRoutingCase(
            case_id="high-volume-correct",
            split="validation",
            slice="common",
            expected=label(sso),
            input_text="authentication failed",
            student=cached(sso, latency_ms=100, cost_usd=0.01),
            frontier=cached(sso, latency_ms=500, cost_usd=0.10),
            student_score=0.95,
            population_weight=7.0,
        ),
        CachedRoutingCase(
            case_id="low-score-wrong",
            split="validation",
            slice="common",
            expected=label(mfa),
            input_text="authentication failed",
            student=cached(sso, latency_ms=100, cost_usd=0.01),
            frontier=cached(mfa, latency_ms=500, cost_usd=0.10),
            student_score=0.80,
        ),
        CachedRoutingCase(
            case_id="overconfident-wrong",
            split="validation",
            slice="adversarial",
            expected=label(mfa),
            input_text="authentication failed",
            student=cached(sso, latency_ms=100, cost_usd=0.01),
            frontier=cached(mfa, latency_ms=500, cost_usd=0.10),
            student_score=0.90,
        ),
        CachedRoutingCase(
            case_id="invalid-then-frontier-error",
            split="validation",
            slice="adversarial",
            expected=label(sso),
            input_text="authentication failed",
            student=CachedPrediction(
                raw_prediction=json.dumps({**label(sso), "confidence": 0.99, "extra": "bad"}),
                latency_ms=100,
                cost_usd=0.01,
            ),
            frontier=CachedPrediction(
                error=PredictionError(kind="provider_timeout", retriable=True),
                latency_ms=500,
                cost_usd=0.10,
            ),
            student_score=0.99,
        ),
    )


def test_threshold_metrics_are_population_weighted_and_include_sequential_fallback() -> None:
    point = evaluate_routing_threshold(routing_cases(), 0.85, schema_model=TriageResult)
    metrics = point.aggregate
    assert metrics.population_weight == 10.0
    assert metrics.frontier_rate == pytest.approx(0.20)
    assert metrics.frontier_calls_per_100 == pytest.approx(20.0)
    assert metrics.selected_error_rate == pytest.approx(0.10)
    assert metrics.schema_validity_rate == 1.0
    assert metrics.exact_match_rate == pytest.approx(8 / 9)
    assert metrics.average_latency_ms == pytest.approx(200.0)
    assert metrics.p95_latency_ms == 600.0
    assert metrics.average_cost_usd == pytest.approx(0.03)
    assert metrics.overconfident_errors_per_100 == pytest.approx(10.0)
    assert metrics.issue.macro_f1 == pytest.approx(0.8)
    assert metrics.routing_accuracy == 1.0
    assert metrics.severity_accuracy == 1.0
    assert metrics.scope_accuracy == 1.0
    assert metrics.action_accuracy == 1.0

    adversarial = point.slices["adversarial"]
    assert adversarial.frontier_rate == 0.5
    assert adversarial.selected_error_rate == 0.5
    assert adversarial.overconfident_errors_per_100 == 50.0


def test_invalid_student_is_always_escalated_even_at_high_score() -> None:
    point = evaluate_routing_threshold(routing_cases(), 0.0, schema_model=TriageResult)
    assert point.aggregate.frontier_calls_per_100 == pytest.approx(10.0)
    assert point.aggregate.selected_error_rate == pytest.approx(0.10)


def test_sweep_reuses_pairs_and_increases_frontier_share() -> None:
    sweep = sweep_routing_thresholds(
        routing_cases(),
        thresholds=(0.85, 0.95),
        schema_model=TriageResult,
    )
    assert sweep.thresholds == (0.85, 0.95)
    assert [item.aggregate.frontier_calls_per_100 for item in sweep.results] == pytest.approx(
        [20.0, 30.0]
    )
    assert sweep.results[1].aggregate.overconfident_errors_per_100 == 0.0
    assert sweep.results[1].aggregate.p95_latency_ms == 600.0


def test_threshold_sweep_refuses_final_test_data_by_default() -> None:
    cases = tuple(case.model_copy(update={"split": "test"}) for case in routing_cases())
    with pytest.raises(ValueError, match="requires only split 'validation'"):
        sweep_routing_thresholds(cases, thresholds=(0.8,), schema_model=TriageResult)


def test_weighted_percentile_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        weighted_percentile([(1.0, -1.0)], 0.95)
    with pytest.raises(ValueError, match="finite"):
        weighted_percentile([(1.0, float("nan"))], 0.95)
