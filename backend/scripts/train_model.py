from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training import train_transformer

DEFAULT_DATASET = PROJECT_ROOT / "data" / "v2"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "v2_manifest.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "jailbreak_transformer"
DEFAULT_SMOKE_MODEL = Path("/tmp/demo-v2-smoke")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the calibrated V2 jailbreak transformer")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    model_dir = args.model_dir
    if model_dir is None:
        model_dir = DEFAULT_SMOKE_MODEL if args.smoke else DEFAULT_MODEL
    elif args.smoke and model_dir.resolve() == DEFAULT_MODEL.resolve():
        sys.exit("smoke training cannot write to the release model directory")
    result = train_transformer(args.dataset_dir, model_dir, manifest_path=args.manifest, smoke=args.smoke)
    print(json.dumps({"model_dir": str(result.model_dir), "sample_counts": result.sample_counts, "metrics": result.metrics, "temperature": result.temperature, "model_version": result.model_version}, indent=2))


if __name__ == "__main__":
    main()
