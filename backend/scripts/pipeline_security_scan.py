from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.security import analyze_security_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run security analysis on a prompt")
    parser.add_argument("--prompt", type=str, help="Prompt text to analyze")
    parser.add_argument("--model-dir", type=Path, help="Optional trained pipeline model directory")
    args = parser.parse_args()

    prompt_text = args.prompt
    if prompt_text is None:
        if not sys.stdin.isatty():
            prompt_text = sys.stdin.read().strip()
        else:
            parser.exit(2, "error: provide --prompt or pipe text via stdin\n")

    if not prompt_text:
        parser.exit(2, "error: prompt text cannot be empty\n")

    try:
        result = analyze_security_prompt(prompt_text, model_dir=args.model_dir)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
