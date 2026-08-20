"""Inspectable uncertainty signals for selective model routing.

The resulting score is a routing score, not a calibrated probability. Each
component is preserved so experiments can ablate and recalibrate it.
"""

from __future__ import annotations

import math
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

from modelforge.schemas import AffectedScope, IssueType, TicketInput, TriageResult

_SPACE = re.compile(r"\s+")
_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all instructions",
    "system prompt",
    "developer message",
    "classify this as",
    "return p1",
)
_AMBIGUOUS_MARKERS = (
    "not sure",
    "something is wrong",
    "doesn't work",
    "does not work",
    "please help",
    "no other details",
    "unknown error",
)
_CATEGORY_HINTS = {
    "sso": ("sso", "okta", "single sign-on", "redirect loop"),
    "mfa": ("mfa", "authenticator", "verification code", "push"),
    "locked": ("locked", "disabled account", "too many attempts"),
    "access": ("access denied", "permission", "forbidden", "entitlement"),
    "provisioning": ("provision", "sync", "new hire", "group membership"),
}


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return _SPACE.sub(" ", value).strip()


class ConfidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    is_probability: bool = False
    hard_failure: bool = False
    components: dict[str, float | None]
    reasons: tuple[str, ...] = ()


class ConfidenceAssessor:
    """Combine grounding, ambiguity, and optional model signals conservatively."""

    _weights = {
        "structural": 0.10,
        "evidence_grounding": 0.25,
        "ambiguity": 0.25,
        "self_reported": 0.15,
        "label_logprob": 0.15,
        "sample_agreement": 0.10,
    }

    def assess(
        self,
        ticket: TicketInput,
        result: TriageResult,
        *,
        structural_valid: bool = True,
        label_logprob: float | None = None,
        sample_agreement: float | None = None,
    ) -> ConfidenceAssessment:
        reasons: list[str] = []
        corpus = _normalize(
            " ".join((ticket.subject, ticket.body, ticket.employee_department))
        )
        supported = [
            bool(_normalize(item)) and _normalize(item) in corpus for item in result.evidence
        ]
        evidence_grounding = sum(supported) / len(supported) if supported else 0.0
        ambiguity = self._ambiguity_score(ticket, result, corpus, reasons)

        components: dict[str, float | None] = {
            "structural": 1.0 if structural_valid else 0.0,
            "evidence_grounding": round(evidence_grounding, 4),
            "ambiguity": round(ambiguity, 4),
            "self_reported": float(result.confidence),
            "label_logprob": self._optional_unit(label_logprob, "label_logprob"),
            "sample_agreement": self._optional_unit(sample_agreement, "sample_agreement"),
        }

        if not structural_valid:
            reasons.append("schema_invalid")
        if evidence_grounding < 1.0:
            reasons.append("unsupported_evidence")
        hard_failure = not structural_valid or evidence_grounding < 1.0
        if hard_failure:
            return ConfidenceAssessment(
                score=0.0,
                hard_failure=True,
                components=components,
                reasons=tuple(dict.fromkeys(reasons)),
            )

        available = {
            name: value for name, value in components.items() if value is not None
        }
        weight_total = sum(self._weights[name] for name in available)
        # A weighted geometric mean prevents a high self-report from completely
        # hiding one weak available signal. Ambiguity is also a hard upper cap.
        log_score = sum(
            self._weights[name] * math.log(max(float(value), 1e-6))
            for name, value in available.items()
        ) / weight_total
        score = min(math.exp(log_score), ambiguity)
        return ConfidenceAssessment(
            score=round(max(0.0, min(1.0, score)), 4),
            components=components,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _optional_unit(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
        return float(value)

    @staticmethod
    def _ambiguity_score(
        ticket: TicketInput,
        result: TriageResult,
        corpus: str,
        reasons: list[str],
    ) -> float:
        score = 1.0
        if len(_normalize(ticket.subject + " " + ticket.body)) < 45:
            score -= 0.30
            reasons.append("insufficient_detail")
        if any(marker in corpus for marker in _AMBIGUOUS_MARKERS):
            score -= 0.25
            reasons.append("ambiguous_language")
        if any(marker in corpus for marker in _INJECTION_MARKERS):
            score -= 0.45
            reasons.append("prompt_injection_marker")
        categories_present = sum(
            any(hint in corpus for hint in hints) for hints in _CATEGORY_HINTS.values()
        )
        if categories_present >= 3:
            score -= 0.20
            reasons.append("conflicting_issue_signals")
        if result.issue_type == IssueType.OTHER_IAM:
            score -= 0.15
            reasons.append("other_iam")
        if result.affected_scope == AffectedScope.UNKNOWN:
            score -= 0.15
            reasons.append("unknown_scope")
        return max(0.0, score)
