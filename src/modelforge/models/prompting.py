"""Versioned IAM prompt loading, rendering, and strict result parsing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from modelforge.models._prompt_resource import IAM_TRIAGE_PROMPT_TEXT
from modelforge.models.base import ModelConfigurationError, ModelOutputError
from modelforge.schemas import TicketInput, TriageResult
from modelforge.schemas._normalization import contains_normalized_literal

IAM_PROMPT_VERSION = "iam_triage_v1"
_SYSTEM_MARKER = "\n--- system ---\n"
_USER_MARKER = "\n--- user ---\n"
_TICKET_PLACEHOLDER = "{{ticket_json}}"
_SCHEMA_PLACEHOLDER = "{{output_schema}}"


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    version: str
    system: str
    user_template: str
    sha256: str
    path: Path | None


def default_prompt_path() -> Path:
    return Path(__file__).resolve().parents[3] / "prompts" / "iam_triage_v1.txt"


def load_iam_prompt(path: Path | None = None) -> PromptDefinition:
    """Load the prompt as a versioned artifact rather than an inline global."""

    resolved = (path or default_prompt_path()).resolve()
    try:
        raw = resolved.read_text(encoding="utf-8")
        source_path: Path | None = resolved
    except OSError as exc:
        if path is not None:
            raise ModelConfigurationError("IAM prompt artifact is unavailable", cause=exc) from exc
        # Python modules are always packaged by setuptools. This exact copy is
        # a wheel-safe fallback for the human-editable root prompt; tests guard
        # against the two versions drifting.
        raw = IAM_TRIAGE_PROMPT_TEXT
        source_path = None

    if not raw.startswith("version: ") or _SYSTEM_MARKER not in raw or _USER_MARKER not in raw:
        raise ModelConfigurationError("IAM prompt artifact has an invalid structure")
    version_line, remainder = raw.split(_SYSTEM_MARKER, 1)
    version = version_line.removeprefix("version: ").strip()
    system, user_template = remainder.split(_USER_MARKER, 1)
    if version != IAM_PROMPT_VERSION:
        raise ModelConfigurationError(
            f"unsupported IAM prompt version {version!r}; expected {IAM_PROMPT_VERSION!r}"
        )
    if user_template.count(_TICKET_PLACEHOLDER) != 1:
        raise ModelConfigurationError("IAM prompt must contain one ticket placeholder")
    if user_template.count(_SCHEMA_PLACEHOLDER) != 1:
        raise ModelConfigurationError("IAM prompt must contain one schema placeholder")
    return PromptDefinition(
        version=version,
        system=system.strip(),
        user_template=user_template.strip(),
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        path=source_path,
    )


def render_iam_messages(
    ticket: TicketInput,
    *,
    prompt: PromptDefinition | None = None,
) -> list[dict[str, str]]:
    """Render trusted instructions separately from explicitly untrusted data."""

    definition = prompt or load_iam_prompt()
    ticket_json = json.dumps(
        ticket.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    schema_json = json.dumps(
        TriageResult.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    user = definition.user_template.replace(_TICKET_PLACEHOLDER, ticket_json).replace(
        _SCHEMA_PLACEHOLDER,
        schema_json,
    )
    return [
        {"role": "system", "content": definition.system},
        {"role": "user", "content": user},
    ]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def parse_triage_json(raw_output: str) -> TriageResult:
    """Accept exactly one standards-compliant JSON object and validate it.

    Markdown fences, explanatory prose, duplicate keys, NaN/Infinity, missing
    fields, and extra fields are all failures. Provider/model errors therefore
    remain distinct from legitimate (but wrong) classifications.
    """

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ModelOutputError(
            "model returned an empty structured response",
            raw_output=raw_output if isinstance(raw_output, str) else None,
        )
    try:
        payload = json.loads(
            raw_output,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_json,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ModelOutputError(
            "model response is not strict JSON",
            raw_output=raw_output,
            cause=exc,
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelOutputError(
            "model response must be one JSON object",
            raw_output=raw_output,
        )

    expected = set(TriageResult.model_fields)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ModelOutputError(
            f"model response fields do not match the contract; missing={missing}, extra={extra}",
            raw_output=raw_output,
        )
    try:
        # JSON enum values are strings. Python-mode ``strict=True`` would reject
        # those valid StrEnum inputs; the domain schema itself already rejects
        # extras and unsafe confidence coercion.
        return TriageResult.model_validate(payload)
    except ValidationError as exc:
        raise ModelOutputError(
            "model response failed the triage schema",
            raw_output=raw_output,
            cause=exc,
        ) from exc


def require_grounded_evidence(
    ticket: TicketInput,
    result: TriageResult,
    *,
    raw_output: str | None = None,
) -> None:
    """Reject evidence that is not a literal subject/body substring."""

    ticket_text = f"{ticket.subject}\n{ticket.body}"
    if any(
        not contains_normalized_literal(ticket_text, snippet)
        for snippet in result.evidence
    ):
        # Unsupported snippets can contain ticket data. Never copy them into
        # the exception text that an API or structured logger might expose.
        raise ModelOutputError(
            "model response contains evidence not present in the ticket",
            raw_output=raw_output,
        )
