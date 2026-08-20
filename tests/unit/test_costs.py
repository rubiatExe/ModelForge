from datetime import date

import pytest

from modelforge.economics import (
    InferenceUsage,
    TokenPricing,
    estimate_token_cost,
    hybrid_average_cost,
    monthly_variable_cost,
)


def test_token_and_hybrid_costs_include_sequential_small_call() -> None:
    pricing = TokenPricing(
        provider="fixture-provider",
        model_revision="fixture-v1",
        effective_date=date(2026, 8, 20),
        input_usd_per_million=2,
        output_usd_per_million=8,
    )
    frontier = estimate_token_cost(InferenceUsage(input_tokens=1_000, output_tokens=100), pricing)
    assert frontier == 0.0028
    average = hybrid_average_cost(
        small_cost_per_request=0.0002,
        frontier_cost_per_request=frontier,
        escalation_rate=0.25,
    )
    assert average == pytest.approx(0.0009)
    assert monthly_variable_cost(average, 100_000) == 90.0


def test_hybrid_cost_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="escalation_rate"):
        hybrid_average_cost(
            small_cost_per_request=0,
            frontier_cost_per_request=1,
            escalation_rate=1.1,
        )
