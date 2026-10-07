from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import InferenceEngine

DEFAULT_MODEL = PROJECT_ROOT / "models" / "jailbreak_transformer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one prompt with the calibrated V2 transformer")
    parser.add_argument("prompt")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = InferenceEngine(args.model_dir).analyze(args.prompt)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
