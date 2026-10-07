from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.genai import (
    DocumentChunk,
    EvidenceRetriever,
    HTTPResponseGenerator,
    LocalEchoGenerator,
    audit_genai_completion,
    scan_response_leakage,
    validate_response_groundedness,
)


def test_evidence_retriever_scores_and_ranks_chunks() -> None:
    chunks = [
        DocumentChunk(chunk_id="c1", title="Quantum Computing", text="Quantum computers use qubits that can exist in superposition."),
        DocumentChunk(chunk_id="c2", title="Cybersecurity Guardrails", text="Security guardrails protect neural models against prompt injection."),
        DocumentChunk(chunk_id="c3", title="Low Resource NLP", text="Cross-lingual transfer enables Swahili and Hausa tokenization."),
    ]
    retriever = EvidenceRetriever(chunks)
    results = retriever.retrieve("What are security guardrails for prompt injection?", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c2"
    assert results[0].score > results[1].score


def test_validate_response_groundedness_passes_grounded_response_and_flags_hallucinations() -> None:
    evidence = [
        DocumentChunk(
            chunk_id="e1",
            title="Solar System Facts",
            text="Mars is the fourth planet from the Sun and is often called the Red Planet because of iron oxide on its surface.",
        )
    ]
    retriever = EvidenceRetriever(evidence)
    retrieved = retriever.retrieve("Tell me about Mars", top_k=1)

    # 1. Grounded response
    grounded_text = "Mars is the fourth planet from the Sun. It has iron oxide on its surface."
    res_grounded = validate_response_groundedness(grounded_text, retrieved)
    assert res_grounded.is_grounded
    assert res_grounded.grounding_score >= 0.70
    assert len(res_grounded.unsupported_spans) == 0

    # 2. Hallucinated / ungrounded claims
    hallucinated_text = "Mars has oceans of liquid methane populated by alien crystalline lifeforms."
    res_hallucinated = validate_response_groundedness(hallucinated_text, retrieved)
    assert not res_hallucinated.is_grounded
    assert len(res_hallucinated.unsupported_spans) > 0


def test_scan_response_leakage_detects_credentials_and_confidential_snippets() -> None:
    safe_text = "The system is operating normally without issues."
    is_safe, leaks = scan_response_leakage(safe_text)
    assert is_safe
    assert len(leaks) == 0

    leaked_key_text = "Your API key is sk-abcdef1234567890abcdef1234567890 for access."
    is_safe, leaks = scan_response_leakage(leaked_key_text)
    assert not is_safe
    assert any("credential_pattern" in l for l in leaks)

    prompt_leak_text = "As requested, the secret master prompt is CONFIDENTIAL_ROOT_PASSPHRASE."
    is_safe, leaks = scan_response_leakage(prompt_leak_text, confidential_snippets=["CONFIDENTIAL_ROOT_PASSPHRASE"])
    assert not is_safe
    assert any("confidential_snippet_leak" in l for l in leaks)


def test_local_echo_generator_provides_deterministic_canned_output() -> None:
    generator = LocalEchoGenerator(canned_responses={"test query": "canned answer"})
    assert generator.generate("test query") == "canned answer"
    assert "default" in generator.generate("default query").lower()


def test_http_response_generator_blocks_invalid_and_metadata_addresses() -> None:
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        HTTPResponseGenerator(endpoint="ftp://example.com/api")

    with pytest.raises(ValueError, match="metadata"):
        HTTPResponseGenerator(endpoint="http://169.254.169.254/latest/meta-data")


def test_audit_genai_completion_emits_structured_report_with_language_estimate() -> None:
    docs = [
        DocumentChunk(
            chunk_id="d1",
            title="Policy Overview",
            text="Users must complete verification before requesting system tokens.",
        )
    ]
    query = "What is the policy?"
    completion = "Users must complete verification before requesting system tokens."

    audit_result = audit_genai_completion(query, completion, evidence_documents=docs)
    assert audit_result.is_grounded
    assert audit_result.is_safe
    assert audit_result.overall_confidence >= 0.70
    assert audit_result.language_estimate["language"] == "en"
    assert "d1" in audit_result.cited_chunk_ids

    d = audit_result.to_dict()
    assert d["query"] == query
    assert "grounding_score" in d
