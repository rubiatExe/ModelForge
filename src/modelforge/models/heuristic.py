"""A deterministic, local-only demo model.

This adapter is intentionally modest: it proves API/routing plumbing without
network calls, credentials, model downloads, or persistence of ticket text.
It is a harness fixture, never model-quality evidence.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from modelforge.models.base import ModelPrediction, ModelRole
from modelforge.schemas import (
    AffectedScope,
    IssueType,
    RecommendedAction,
    RoutingTeam,
    Severity,
    TicketInput,
    TriageResult,
)

_ISSUE_HINTS: tuple[tuple[IssueType, tuple[str, ...]], ...] = (
    (
        IssueType.MFA_FAILURE,
        ("mfa", "multi-factor", "multifactor", "authenticator", "verification code", "push denied"),
    ),
    (
        IssueType.ACCOUNT_LOCKED_OR_DISABLED,
        ("account locked", "locked out", "account disabled", "too many attempts"),
    ),
    (
        IssueType.ACCESS_DENIED_AFTER_LOGIN,
        ("access denied", "forbidden", "permission denied", "missing entitlement"),
    ),
    (
        IssueType.PROVISIONING_OR_SYNC_FAILURE,
        ("provision", "scim", "directory sync", "group sync", "new hire"),
    ),
    (
        IssueType.SSO_AUTHENTICATION_FAILURE,
        ("sso", "single sign-on", "saml", "identity provider", "redirect loop", "okta"),
    ),
)

_ACTION_BY_ISSUE = {
    IssueType.MFA_FAILURE: RecommendedAction.INVESTIGATE_MFA_CONFIGURATION,
    IssueType.ACCOUNT_LOCKED_OR_DISABLED: RecommendedAction.RESET_OR_UNLOCK_ACCOUNT,
    IssueType.ACCESS_DENIED_AFTER_LOGIN: RecommendedAction.REVIEW_ACCESS_ENTITLEMENTS,
    IssueType.PROVISIONING_OR_SYNC_FAILURE: RecommendedAction.INVESTIGATE_PROVISIONING_SYNC,
    IssueType.SSO_AUTHENTICATION_FAILURE: RecommendedAction.INVESTIGATE_IDP_OR_SSO_CONFIGURATION,
    IssueType.OTHER_IAM: RecommendedAction.COLLECT_MORE_INFORMATION,
}

_TEAM_BY_ISSUE = {
    IssueType.MFA_FAILURE: RoutingTeam.IDENTITY_PLATFORM,
    IssueType.ACCOUNT_LOCKED_OR_DISABLED: RoutingTeam.SERVICE_DESK,
    IssueType.ACCESS_DENIED_AFTER_LOGIN: RoutingTeam.ACCESS_MANAGEMENT,
    IssueType.PROVISIONING_OR_SYNC_FAILURE: RoutingTeam.IDENTITY_PLATFORM,
    IssueType.SSO_AUTHENTICATION_FAILURE: RoutingTeam.IDENTITY_PLATFORM,
    IssueType.OTHER_IAM: RoutingTeam.SERVICE_DESK,
}

_SECURITY_HINTS = ("compromised", "phishing", "stolen token", "unauthorized login")
_ORGANIZATION_HINTS = ("organization-wide", "company-wide", "all employees", "all users", "everyone")
_DEPARTMENT_HINTS = ("our department", "entire department", "department-wide")
_MULTIPLE_HINTS = ("multiple users", "several users", "our team", "teammates", "many users")
_SINGLE_HINTS = ("my account", "i cannot", "i can't", "for me", "one user", "single user")


def _matches(text: str, hints: Iterable[str]) -> list[str]:
    """Return minimal literal evidence snippets, preserving input spelling."""

    folded = text.casefold()
    matches: list[str] = []
    for hint in hints:
        start = folded.find(hint.casefold())
        if start >= 0:
            snippet = text[start : start + len(hint)]
            if snippet.casefold() not in {item.casefold() for item in matches}:
                matches.append(snippet)
    return matches


def routing_for_issue(issue_type: IssueType) -> RoutingTeam:
    """Return the deterministic demo routing default for an issue label."""

    return _TEAM_BY_ISSUE[issue_type]


def action_for_issue(issue_type: IssueType) -> RecommendedAction:
    """Return the deterministic demo action default for an issue label."""

    return _ACTION_BY_ISSUE[issue_type]


class HeuristicTriageModel:
    """Privacy-safe local demo adapter implementing :class:`TriageModel`."""

    model_id = "heuristic-demo-v1"
    role: ModelRole = "demo"

    async def triage(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        started = time.perf_counter()
        text = f"{ticket.subject}\n{ticket.body}"

        issue_type = IssueType.OTHER_IAM
        issue_evidence: list[str] = []
        for candidate, hints in _ISSUE_HINTS:
            found = _matches(text, hints)
            if found:
                issue_type = candidate
                issue_evidence = found
                break

        scope, scope_evidence = self._scope(text)
        action = action_for_issue(issue_type)
        security_evidence = _matches(text, _SECURITY_HINTS)
        if security_evidence:
            action = RecommendedAction.ESCALATE_SECURITY_REVIEW

        evidence = list(dict.fromkeys(issue_evidence + scope_evidence + security_evidence))[:5]
        if not evidence:
            # TriageResult requires grounded evidence. The subject is returned to
            # the same caller but is never placed in metadata or logs.
            evidence = [ticket.subject]

        confidence = 0.82 if issue_evidence else 0.52
        if scope is AffectedScope.UNKNOWN:
            confidence -= 0.12
        if security_evidence:
            confidence = min(confidence, 0.68)
        confidence = round(max(0.0, confidence), 2)
        result = TriageResult(
            issue_type=issue_type,
            severity=self._severity(scope, bool(security_evidence)),
            routing_team=routing_for_issue(issue_type),
            affected_scope=scope,
            recommended_action=action,
            evidence=evidence,
            confidence=confidence,
        )
        return ModelPrediction(
            result=result,
            model_id=self.model_id,
            model_version="1",
            role=self.role,
            prompt_version=None,
            latency_ms=(time.perf_counter() - started) * 1_000,
            estimated_cost_usd=0.0,
            metadata={
                "local_only": True,
                "privacy_safe": True,
                "harness_fixture": True,
            },
        )

    async def predict(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction:
        """Alias retained for experiment code that uses ``predict`` terminology."""

        return await self.triage(ticket, timeout_s=timeout_s)

    @staticmethod
    def _scope(text: str) -> tuple[AffectedScope, list[str]]:
        for scope, hints in (
            (AffectedScope.ORGANIZATION, _ORGANIZATION_HINTS),
            (AffectedScope.DEPARTMENT, _DEPARTMENT_HINTS),
            (AffectedScope.MULTIPLE_USERS, _MULTIPLE_HINTS),
            (AffectedScope.SINGLE_USER, _SINGLE_HINTS),
        ):
            found = _matches(text, hints)
            if found:
                return scope, found
        return AffectedScope.UNKNOWN, []

    @staticmethod
    def _severity(scope: AffectedScope, security_signal: bool) -> Severity:
        if scope is AffectedScope.ORGANIZATION:
            return Severity.P1
        if security_signal or scope in {AffectedScope.DEPARTMENT, AffectedScope.MULTIPLE_USERS}:
            return Severity.P2
        if scope is AffectedScope.SINGLE_USER:
            return Severity.P3
        return Severity.P4
