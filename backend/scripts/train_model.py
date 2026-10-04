from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training import train_classifier


DEFAULT_DATASET = PROJECT_ROOT / "data" / "prompts.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "jailbreak_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V1 jailbreak classifier")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    result = train_classifier(args.dataset, args.model)
    print(f"samples: {result.sample_count}")
    print(f"class_counts: {result.class_counts}")
    print(f"roc_auc: {result.roc_auc:.4f}")
    print(f"model: {result.model_path}")


if __name__ == "__main__":
    main()
