"""Explicit, date-stamped variable-cost calculations.

Self-hosted fixed infrastructure costs belong in an experiment's workload
assumptions; this module does not pretend every local token has a universal
price.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class TokenPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=80)
    model_revision: str = Field(min_length=1, max_length=200)
    effective_date: date
    source_url: HttpUrl | None = None
    input_usd_per_million: float = Field(ge=0.0)
    output_usd_per_million: float = Field(ge=0.0)


class InferenceUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


def estimate_token_cost(usage: InferenceUsage, pricing: TokenPricing) -> float:
    cost = (
        usage.input_tokens * pricing.input_usd_per_million
        + usage.output_tokens * pricing.output_usd_per_million
    ) / 1_000_000
    return round(cost, 10)


def hybrid_average_cost(
    *,
    small_cost_per_request: float,
    frontier_cost_per_request: float,
    escalation_rate: float,
) -> float:
    """Sequential fallback pays the small-model cost on every request."""

    if small_cost_per_request < 0 or frontier_cost_per_request < 0:
        raise ValueError("costs cannot be negative")
    if not 0.0 <= escalation_rate <= 1.0:
        raise ValueError("escalation_rate must be within [0, 1]")
    return small_cost_per_request + escalation_rate * frontier_cost_per_request


def monthly_variable_cost(average_cost_per_request: float, monthly_requests: int) -> float:
    if average_cost_per_request < 0 or monthly_requests < 0:
        raise ValueError("cost and requests cannot be negative")
    return round(average_cost_per_request * monthly_requests, 2)
