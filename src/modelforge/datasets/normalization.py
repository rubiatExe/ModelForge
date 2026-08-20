"""Canonical text normalization used only for matching and fingerprints."""

from __future__ import annotations

from modelforge.schemas._normalization import normalize_unicode_whitespace


def normalize_text(value: str, *, casefold: bool = False) -> str:
    """NFKC-normalize and collapse Unicode whitespace without lossy rewriting."""

    return normalize_unicode_whitespace(value, casefold=casefold)


def normalize_for_fingerprint(value: str) -> str:
    """Normalize casing as well as Unicode/whitespace for identity checks."""

    return normalize_text(value, casefold=True)
