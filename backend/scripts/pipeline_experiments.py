from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.experiments import EnsembleModel, ExperimentRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment tracking and ensembling utility")
    parser.add_argument("--scan-dir", type=Path, help="Directory to scan for run artifacts")
    parser.add_argument("--compare", nargs="+", type=Path, help="Model directories to compare side-by-side")
    parser.add_argument("--ensemble", nargs="+", type=Path, help="Member model directories to ensemble")
    parser.add_argument("--data", type=Path, help="Dataset CSV for ensemble evaluation")
    parser.add_argument("--method", choices=["average", "weighted", "majority_vote"], default="average", help="Ensemble aggregation method")
    args = parser.parse_args()

    if args.scan_dir:
        registry = ExperimentRegistry(args.scan_dir)
        runs = registry.scan_runs()
        print(json.dumps([r.to_dict() for r in runs], indent=2))
        return

    if args.compare:
        try:
            comparison = ExperimentRegistry.compare_runs(args.compare)
            print(json.dumps(comparison, indent=2))
        except (ValueError, OSError) as exc:
            parser.exit(2, f"error: {exc}\n")
        return

    if args.ensemble:
        if not args.data:
            parser.exit(2, "error: --ensemble requires --data for evaluation\n")
        try:
            ensemble = EnsembleModel(args.ensemble, method=args.method)
            eval_res = ensemble.evaluate(args.data)
            print(json.dumps(eval_res, indent=2))
        except (ValueError, OSError) as exc:
            parser.exit(2, f"error: {exc}\n")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
