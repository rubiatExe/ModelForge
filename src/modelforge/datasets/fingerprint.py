"""Canonical, content-addressed identities for dataset records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from modelforge.datasets.normalization import normalize_for_fingerprint
from modelforge.schemas import TicketInput, TriageLabel


def canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically and without insignificant spaces."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_ticket_payload(ticket: TicketInput) -> dict[str, str]:
    """Return fields that define input content, excluding mutable record metadata."""

    return {
        "body": normalize_for_fingerprint(ticket.body),
        "employee_department": normalize_for_fingerprint(ticket.employee_department),
        "subject": normalize_for_fingerprint(ticket.subject),
    }


def canonical_fingerprint(ticket: TicketInput) -> str:
    """Hash semantically exact ticket content independent of ID and timestamp."""

    payload = canonical_json(canonical_ticket_payload(ticket)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_label_fingerprint(label: TriageLabel) -> str:
    """Hash the complete gold decision, including normalized evidence snippets."""

    payload = label.model_dump(mode="json")
    payload["evidence"] = sorted(normalize_for_fingerprint(item) for item in payload["evidence"])
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def fingerprint_set_digest(fingerprints: Iterable[str]) -> str:
    """Hash a sorted set of record fingerprints for manifest/lock comparison."""

    unique = sorted(set(fingerprints))
    serialized = "".join(f"{fingerprint}\n" for fingerprint in unique).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()
