"""Production request path: small model, confidence assessment, explicit fallback."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import ConfigDict, Field

from modelforge.errors import FrontierUnavailableError
from modelforge.models.base import ModelAdapterError, ModelPrediction, TriageModel
from modelforge.routing.confidence import ConfidenceAssessment, ConfidenceAssessor
from modelforge.routing.policy import RoutingPolicy
from modelforge.schemas import TicketInput
from modelforge.schemas._base import StrictBaseModel

RouteReason = Literal["small_confident", "low_confidence", "small_model_error"]


class RoutedPrediction(StrictBaseModel):
    """Selected answer and full sequential-route accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction: ModelPrediction
    small_prediction: ModelPrediction | None = Field(default=None, exclude=True, repr=False)
    confidence_assessment: ConfidenceAssessment | None
    used_frontier: bool
    reason: RouteReason
    policy_id: str
    policy_hash: str
    total_latency_ms: float = Field(ge=0.0)
    total_estimated_cost_usd: float = Field(ge=0.0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)


def _sum_optional(first: int | None, second: int | None = None) -> int | None:
    values = [value for value in (first, second) if value is not None]
    return sum(values) if values else None


class TriageRouter:
    def __init__(
        self,
        *,
        small_model: TriageModel,
        frontier_model: TriageModel | None,
        policy: RoutingPolicy,
        assessor: ConfidenceAssessor | None = None,
        per_model_timeout_s: float = 30.0,
    ) -> None:
        if per_model_timeout_s <= 0:
            raise ValueError("per_model_timeout_s must be positive")
        self.small_model = small_model
        self.frontier_model = frontier_model
        self.policy = policy
        self.assessor = assessor or ConfidenceAssessor()
        self.per_model_timeout_s = per_model_timeout_s

    async def triage(self, ticket: TicketInput) -> RoutedPrediction:
        started = time.perf_counter()
        try:
            small = await self.small_model.triage(ticket, timeout_s=self.per_model_timeout_s)
        except (ModelAdapterError, TimeoutError) as exc:
            return await self._fallback_after_error(ticket, started, exc)

        assessment = self.assessor.assess(
            ticket,
            small.result,
            label_logprob=small.label_logprob,
            sample_agreement=small.sample_agreement,
        )
        if assessment.score >= self.policy.threshold:
            return RoutedPrediction(
                prediction=small,
                small_prediction=small,
                confidence_assessment=assessment,
                used_frontier=False,
                reason="small_confident",
                policy_id=self.policy.policy_id,
                policy_hash=self.policy.content_hash,
                total_latency_ms=(time.perf_counter() - started) * 1_000,
                total_estimated_cost_usd=small.estimated_cost_usd,
                total_input_tokens=small.input_tokens,
                total_output_tokens=small.output_tokens,
            )
        frontier = self._require_frontier("ticket requires confidence escalation")
        try:
            selected = await frontier.triage(ticket, timeout_s=self.per_model_timeout_s)
        except (ModelAdapterError, TimeoutError) as exc:
            raise FrontierUnavailableError(
                "ticket requires escalation but the frontier model is unavailable"
            ) from exc
        return RoutedPrediction(
            prediction=selected,
            small_prediction=small,
            confidence_assessment=assessment,
            used_frontier=True,
            reason="low_confidence",
            policy_id=self.policy.policy_id,
            policy_hash=self.policy.content_hash,
            total_latency_ms=(time.perf_counter() - started) * 1_000,
            total_estimated_cost_usd=(
                small.estimated_cost_usd + selected.estimated_cost_usd
            ),
            total_input_tokens=_sum_optional(small.input_tokens, selected.input_tokens),
            total_output_tokens=_sum_optional(small.output_tokens, selected.output_tokens),
        )

    async def _fallback_after_error(
        self,
        ticket: TicketInput,
        started: float,
        small_error: BaseException,
    ) -> RoutedPrediction:
        frontier = self._require_frontier("small model failed and no frontier is configured")
        try:
            selected = await frontier.triage(ticket, timeout_s=self.per_model_timeout_s)
        except (ModelAdapterError, TimeoutError) as exc:
            raise FrontierUnavailableError("both small and frontier model paths failed") from exc
        return RoutedPrediction(
            prediction=selected,
            small_prediction=None,
            confidence_assessment=None,
            used_frontier=True,
            reason="small_model_error",
            policy_id=self.policy.policy_id,
            policy_hash=self.policy.content_hash,
            total_latency_ms=(time.perf_counter() - started) * 1_000,
            total_estimated_cost_usd=selected.estimated_cost_usd,
            total_input_tokens=selected.input_tokens,
            total_output_tokens=selected.output_tokens,
        )

    def _require_frontier(self, message: str) -> TriageModel:
        if self.frontier_model is None:
            raise FrontierUnavailableError(message)
        return self.frontier_model
