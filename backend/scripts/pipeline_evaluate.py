from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.core import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline pipeline")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()

    try:
        result = evaluate(args.model_dir, input_path=args.input)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
