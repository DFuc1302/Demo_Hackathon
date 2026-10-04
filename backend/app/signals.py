from __future__ import annotations

import re

_SIGNAL_PATTERNS = (
    ("instruction override", re.compile(r"\b(ignore|disregard|override|cancel)\b.*\b(previous|prior|policy|instructions?)\b", re.IGNORECASE)),
    ("system prompt request", re.compile(r"\b(reveal|show|print|provide)\b.*\b(hidden|system|internal|private)\b.*\b(prompt|instructions?|configuration|rules?)\b", re.IGNORECASE)),
    ("safety bypass", re.compile(r"\b(bypass|evade|ignore|without)\b.*\b(safety|safeguards?|guardrails?|restrictions?|rules?)\b", re.IGNORECASE)),
    ("unrestricted persona", re.compile(r"\b(unrestricted|without restrictions|must never refuse)\b", re.IGNORECASE)),
    ("role-play evasion", re.compile(r"\b(role[- ]play|fictional persona|pretend)\b", re.IGNORECASE)),
)


def extract_signals(prompt: str) -> list[str]:
    return [name for name, pattern in _SIGNAL_PATTERNS if pattern.search(prompt)]
