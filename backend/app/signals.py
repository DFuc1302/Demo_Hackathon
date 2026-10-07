from __future__ import annotations

import re

_SIGNAL_PATTERNS = (
    ("instruction override", re.compile(r"\b(ignore|disregard|override|cancel)\b.*\b(previous|prior|policy|instructions?)\b", re.IGNORECASE)),
    ("system prompt request", re.compile(r"\b(reveal|show|print|provide)\b.*\b(hidden|system|internal|private)\b.*\b(prompt|instructions?|configuration|rules?)\b", re.IGNORECASE)),
    ("safety bypass", re.compile(r"\b(bypass|evade|ignore|without)\b.*\b(safety|safeguards?|guardrails?|restrictions?|rules?)\b", re.IGNORECASE)),
    ("unrestricted persona", re.compile(r"\b(unrestricted|without restrictions|must never refuse)\b", re.IGNORECASE)),
    ("role-play evasion", re.compile(r"\b(role[- ]play|fictional persona|pretend)\b", re.IGNORECASE)),
    ("markdown exfiltration", re.compile(r"!\[.*?\]\((https?://|[a-z0-9+.-]+://)[^\s)]+\)", re.IGNORECASE)),
    ("delimiter manipulation", re.compile(r"(---|===|###)\s*(end|start|new)?\s*(system|prompt|instructions?|context)\b", re.IGNORECASE)),
    ("counterfactual framing", re.compile(r"\b(in an alternate (reality|universe)|hypothetically speaking|purely hypothetical scenario)\b", re.IGNORECASE)),
    ("encoded payload instruction", re.compile(r"\b(decode|decrypt|translate)\b.*\b(base64|rot13|hex|cipher)\b", re.IGNORECASE)),
)

_SECRET_PATTERNS = (
    ("openai_api_key", re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}\b", re.IGNORECASE)),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b", re.IGNORECASE)),
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE)),
    ("generic_api_secret", re.compile(r"\b(?:api[_-]?key|secret[_-]?token|auth[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", re.IGNORECASE)),
)


def detect_secrets(text: str) -> list[str]:
    """Detect exposed credentials, tokens, or high-entropy secrets in text."""
    return [name for name, pattern in _SECRET_PATTERNS if pattern.search(text)]


def extract_signals(prompt: str) -> list[str]:
    return [name for name, pattern in _SIGNAL_PATTERNS if pattern.search(prompt)]
