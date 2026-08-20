"""Small normalization primitives shared by schemas and dataset code."""

from __future__ import annotations

import unicodedata


def normalize_unicode_whitespace(value: str, *, casefold: bool = False) -> str:
    """Apply NFKC and collapse every Unicode whitespace run to one space."""

    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    return normalized.casefold() if casefold else normalized


def contains_normalized_literal(haystack: str, needle: str) -> bool:
    """Return whether *needle* is a literal substring after safe normalization."""

    normalized_haystack = normalize_unicode_whitespace(haystack, casefold=True)
    normalized_needle = normalize_unicode_whitespace(needle, casefold=True)
    return bool(normalized_needle) and normalized_needle in normalized_haystack
