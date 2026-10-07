from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from app.pipeline.language import identify_language

_SECRET_PATTERNS = [
    re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"),
    re.compile(r"\b(ghp_[a-zA-Z0-9]{36})\b"),
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\b(bearer\s+[a-zA-Z0-9_\-\.]{25,})\b", re.I),
    re.compile(r"\bpassword\s*[:=]\s*['\"]?[^\s'\"]{6,}['\"]?", re.I),
]

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "be", "been", "this", "that", "it", "with",
    "as", "by", "from", "your", "my", "our", "their", "have", "has", "had",
}


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    title: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class GroundednessValidationResult:
    is_grounded: bool
    grounding_score: float
    unsupported_spans: list[str]
    cited_chunk_ids: list[str]


@dataclass(frozen=True)
class GenAIAuditResult:
    query: str
    completion: str
    is_grounded: bool
    grounding_score: float
    unsupported_claims: list[str]
    cited_chunk_ids: list[str]
    is_safe: bool
    detected_leaks: list[str]
    language_estimate: dict[str, Any]
    overall_confidence: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseResponseGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class LocalEchoGenerator:
    """Deterministic local generation double for offline evaluation."""

    def __init__(self, canned_responses: dict[str, str] | None = None) -> None:
        self.canned_responses = canned_responses or {}

    def generate(self, prompt: str) -> str:
        if prompt in self.canned_responses:
            return self.canned_responses[prompt]
        return f"Verified response regarding: {prompt}"


class HTTPResponseGenerator:
    """Bounded, SSRF-safe HTTP client for remote LLM completions."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_bytes: int = 16384,
        allow_local: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.allow_local = allow_local
        self._validate_endpoint()

    def _validate_endpoint(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("endpoint must be an HTTP or HTTPS URL with a valid hostname")

        port = parsed.port or (80 if parsed.scheme == "http" else 443)
        try:
            addr_info = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"could not resolve endpoint hostname '{parsed.hostname}': {exc}") from exc

        for item in addr_info:
            ip = ipaddress.ip_address(item[4][0])
            if ip.is_link_local or (isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.IPv4Network("169.254.0.0/16")):
                raise ValueError(f"forbidden cloud metadata address: {ip}")
            if not self.allow_local and (ip.is_loopback or ip.is_private or ip.is_reserved):
                raise ValueError(f"forbidden private/loopback address: {ip}")

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read(self.max_bytes + 1)
                if len(raw_bytes) > self.max_bytes:
                    raise ValueError("response exceeded maximum size limit")
                data = json.loads(raw_bytes.decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"HTTP generation failed: {exc}") from exc

        for key in ("response", "text", "output", "completion"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
        raise ValueError("invalid generation response format")


class EvidenceRetriever:
    """Lightweight in-memory lexical retrieval engine over reference document chunks."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("chunks list cannot be empty")
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedEvidence]:
        query_words = set(re.findall(r"\b\w+\b", query.lower())) - _STOPWORDS
        if not query_words:
            return [RetrievedEvidence(chunk=c, score=0.0) for c in self.chunks[:top_k]]

        scored: list[RetrievedEvidence] = []
        for chunk in self.chunks:
            chunk_words = re.findall(r"\b\w+\b", (chunk.title + " " + chunk.text).lower())
            overlap = sum(w in query_words for w in chunk_words)
            # Length-normalized overlap score
            score = overlap / max(1, len(chunk_words) ** 0.5)
            scored.append(RetrievedEvidence(chunk=chunk, score=round(score, 4)))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


def validate_response_groundedness(
    response_text: str,
    evidence: list[RetrievedEvidence],
    threshold: float = 0.45,
) -> GroundednessValidationResult:
    """Validate whether response assertions are supported by evidence."""
    if not response_text.strip():
        return GroundednessValidationResult(is_grounded=True, grounding_score=1.0, unsupported_spans=[], cited_chunk_ids=[])

    if not evidence:
        return GroundednessValidationResult(
            is_grounded=False,
            grounding_score=0.0,
            unsupported_spans=[response_text.strip()],
            cited_chunk_ids=[],
        )

    evidence_text = " ".join(e.chunk.title + " " + e.chunk.text for e in evidence).lower()
    evidence_words = set(re.findall(r"\b\w+\b", evidence_text))

    sentences = re.split(r"[.!?\n]+", response_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    unsupported_spans: list[str] = []
    total_content_words = 0
    grounded_content_words = 0

    for sentence in sentences:
        words = [w.lower() for w in re.findall(r"\b\w+\b", sentence) if w.lower() not in _STOPWORDS]
        if not words:
            continue
        total_content_words += len(words)
        supported = sum(w in evidence_words for w in words)
        grounded_content_words += supported

        # A sentence with less than 35% supported content words is an ungrounded claim
        if len(words) >= 3 and (supported / len(words)) < 0.35:
            unsupported_spans.append(sentence)

    score = grounded_content_words / max(1, total_content_words)
    score = round(min(1.0, max(0.0, score)), 4)
    is_grounded = score >= threshold and len(unsupported_spans) == 0

    cited_ids = [e.chunk.chunk_id for e in evidence if e.score > 0.0]

    return GroundednessValidationResult(
        is_grounded=is_grounded,
        grounding_score=score,
        unsupported_spans=unsupported_spans,
        cited_chunk_ids=cited_ids,
    )


def scan_response_leakage(
    text: str,
    confidential_snippets: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Scan response text for leaked credentials or sensitive snippets."""
    leaks: list[str] = []

    for pat in _SECRET_PATTERNS:
        match = pat.search(text)
        if match:
            leaks.append(f"credential_pattern:{match.group(0)[:12]}...")

    if confidential_snippets:
        text_lower = text.lower()
        for snippet in confidential_snippets:
            clean_snippet = snippet.strip().lower()
            if clean_snippet and len(clean_snippet) >= 6 and clean_snippet in text_lower:
                leaks.append(f"confidential_snippet_leak:{clean_snippet[:20]}")

    is_safe = len(leaks) == 0
    return is_safe, sorted(set(leaks))


def audit_genai_completion(
    query: str,
    completion: str,
    evidence_documents: list[DocumentChunk] | None = None,
    confidential_snippets: list[str] | None = None,
    grounding_threshold: float = 0.45,
) -> GenAIAuditResult:
    """Run comprehensive reliability and grounding audit on generated completion."""
    retrieved: list[RetrievedEvidence] = []
    if evidence_documents:
        retriever = EvidenceRetriever(evidence_documents)
        retrieved = retriever.retrieve(query, top_k=3)

    grounding_res = validate_response_groundedness(completion, retrieved, threshold=grounding_threshold)
    is_safe, leaks = scan_response_leakage(completion, confidential_snippets=confidential_snippets)
    lang_res = identify_language(completion)

    # Confidence calculation: combines grounding score and safety
    conf = grounding_res.grounding_score if retrieved else 0.85
    if not is_safe:
        conf *= 0.50

    return GenAIAuditResult(
        query=query,
        completion=completion,
        is_grounded=grounding_res.is_grounded,
        grounding_score=grounding_res.grounding_score,
        unsupported_claims=grounding_res.unsupported_spans,
        cited_chunk_ids=grounding_res.cited_chunk_ids,
        is_safe=is_safe,
        detected_leaks=leaks,
        language_estimate={
            "language": lang_res.language,
            "confidence": lang_res.confidence,
            "script": lang_res.script,
            "is_uncertain": lang_res.is_uncertain,
        },
        overall_confidence=round(conf, 3),
        provenance={
            "audit_version": "genai_reliability_v1",
            "has_reference_evidence": bool(evidence_documents),
            "evidence_chunk_count": len(evidence_documents) if evidence_documents else 0,
        },
    )
