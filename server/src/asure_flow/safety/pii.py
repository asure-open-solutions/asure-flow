"""Regex-based PII detection and redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    type: str  # "email", "phone", "ssn", "credit_card"
    start: int
    end: int
    original: str
    redacted: str


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", _EMAIL_RE, "[EMAIL]"),
    ("phone", _PHONE_RE, "[PHONE]"),
    ("ssn", _SSN_RE, "[SSN]"),
    ("credit_card", _CREDIT_CARD_RE, "[CARD]"),
]


def detect_pii(text: str) -> list[PIIMatch]:
    """Detect PII patterns in text. May return overlapping matches (e.g. a phone
    pattern matching a sub-span of a credit-card number) — redact_pii resolves
    overlaps before applying replacements."""
    matches: list[PIIMatch] = []
    for pii_type, pattern, replacement in _PATTERNS:
        for m in pattern.finditer(text):
            matches.append(
                PIIMatch(
                    type=pii_type,
                    start=m.start(),
                    end=m.end(),
                    original=m.group(),
                    redacted=replacement,
                )
            )
    return sorted(matches, key=lambda m: m.start)


def _select_non_overlapping(matches: list[PIIMatch]) -> list[PIIMatch]:
    """Pick a non-overlapping subset: earliest start wins, ties broken by the
    longest span (so a credit-card match beats a phone sub-match), then greedily
    skip any match overlapping one already kept. Applying overlapping matches
    would corrupt the surrounding text and could leave real PII partly exposed."""
    ordered = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
    selected: list[PIIMatch] = []
    last_end = -1
    for m in ordered:
        if m.start >= last_end:
            selected.append(m)
            last_end = m.end
    return selected


def redact_pii(text: str) -> tuple[str, list[PIIMatch]]:
    """Redact PII in text. Returns (redacted_text, applied_matches)."""
    matches = detect_pii(text)
    if not matches:
        return text, []
    applied = _select_non_overlapping(matches)
    # Apply replacements from end to start to preserve offsets
    result = text
    for m in sorted(applied, key=lambda m: m.start, reverse=True):
        result = result[: m.start] + m.redacted + result[m.end :]
    return result, applied
