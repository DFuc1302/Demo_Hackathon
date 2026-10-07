from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.genai import DocumentChunk, audit_genai_completion


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a GenAI completion for groundedness and safety")
    parser.add_argument("--query", type=str, required=True, help="Input user query or prompt")
    parser.add_argument("--completion", type=str, help="Generated model response text")
    parser.add_argument("--evidence-file", type=Path, help="Optional JSON file with reference DocumentChunk list")
    parser.add_argument("--output-report", type=Path, help="Optional output JSON report path")
    args = parser.parse_args()

    completion_text = args.completion
    if completion_text is None:
        if not sys.stdin.isatty():
            completion_text = sys.stdin.read().strip()
        else:
            parser.exit(2, "error: provide --completion or pipe via stdin\n")

    if not completion_text:
        parser.exit(2, "error: completion text cannot be empty\n")

    evidence_chunks: list[DocumentChunk] | None = None
    if args.evidence_file:
        try:
            raw_docs = json.loads(args.evidence_file.read_text(encoding="utf-8"))
            evidence_chunks = [
                DocumentChunk(
                    chunk_id=doc.get("chunk_id", f"c-{i}"),
                    title=doc.get("title", ""),
                    text=doc.get("text", ""),
                    source=doc.get("source", ""),
                )
                for i, doc in enumerate(raw_docs)
            ]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            parser.exit(2, f"error loading evidence file: {exc}\n")

    audit_res = audit_genai_completion(
        query=args.query,
        completion=completion_text,
        evidence_documents=evidence_chunks,
    )

    json_str = json.dumps(audit_res.to_dict(), indent=2, ensure_ascii=False)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json_str + "\n", encoding="utf-8")

    print(json_str)


if __name__ == "__main__":
    main()
