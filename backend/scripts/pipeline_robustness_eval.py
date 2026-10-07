from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.robustness import evaluate_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model robustness against controlled text perturbations")
    parser.add_argument("--model-dir", type=Path, required=True, help="Trained pipeline model directory")
    parser.add_argument("--input", type=Path, help="Optional labeled dataset CSV to evaluate against")
    parser.add_argument("--transformations", nargs="+", help="Subset of transformations to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for perturbations")
    parser.add_argument("--output-report", type=Path, help="Optional path to save JSON report")
    args = parser.parse_args()

    try:
        report = evaluate_robustness(
            model_dir=args.model_dir,
            input_path=args.input,
            transformations=args.transformations,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    json_str = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json_str + "\n", encoding="utf-8")
    print(json_str)


if __name__ == "__main__":
    main()
