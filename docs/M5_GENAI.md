# M5 GenAI Reliability and Grounding Guide

## Overview

The M5 GenAI reliability module (`backend/app/pipeline/genai.py`) provides interchangeable response generation abstractions, local lexical evidence retrieval, groundedness validation, secret/leakage scanning, and structured audit reporting.

## Core Components

### 1. Interchangeable Generators

- **`LocalEchoGenerator`**: Deterministic test double returning canned answers or echo responses. Essential for offline evaluation, reproducible test fixtures, and zero-cost CI validation.
- **`HTTPResponseGenerator`**: Remote model client enforcing strict transport boundaries:
  - Validates endpoints against Link-Local, AWS/GCP metadata (`169.254.169.254`), and RFC 1918 private subnets (unless `--allow-local` is explicitly configured).
  - Hard timeouts and response-size caps ($16\text{ KiB}$) to protect against resource exhaustion.
  - Zero external credentials in repository code or committed artifacts.

### 2. Lexical Evidence Retrieval

- **`DocumentChunk(chunk_id, title, text, source)`**: Unit of reference knowledge.
- **`EvidenceRetriever(chunks)`**: In-memory lexical search engine computing length-normalized query-document overlap, ranking and returning top-$k$ evidence passages with citations.

### 3. Response Groundedness Validation

Function: `validate_response_groundedness(response_text, evidence, threshold=0.45) -> GroundednessValidationResult`

- Breaks completions into sentence-level assertion spans.
- Computes content-word and phrase overlap against retrieved evidence chunks.
- Identifies ungrounded or unsupported claims: assertions with insufficient evidence support are flagged in `unsupported_spans`.
- Computes an aggregate `grounding_score` ($0.0$ to $1.0$).

### 4. Response Security and Leakage Scanner

Function: `scan_response_leakage(text, confidential_snippets=None) -> (is_safe, detected_leaks)`

- Regex detection for leaked API keys (`sk-...`, `ghp_...`, `AKIA...`).
- Substring detection for confidential developer prompts or sensitive system instructions.

## CLI Usage

Audit any generated completion for groundedness and safety:

```bash
# Basic safety and language audit
python backend/scripts/pipeline_genai_audit.py \
  --query "What are the rules?" \
  --completion "Users must authenticate before accessing the database."

# Grounding audit against a reference JSON evidence file
python backend/scripts/pipeline_genai_audit.py \
  --query "What are the rules?" \
  --completion "Users must authenticate before accessing the database." \
  --evidence-file path/to/evidence.json \
  --output-report backend/outputs/genai_audit.json
```
