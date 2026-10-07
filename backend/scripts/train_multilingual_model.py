from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.multilingual_training import (
    DEFAULT_DATA_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RELEASE_MODEL_DIR,
    DEFAULT_TRANSLATION_MODEL_PATH,
    train_multilingual_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train separate schema-3 multilingual classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to multilingual jsonl splits directory.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to schema-3 multilingual manifest.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_RELEASE_MODEL_DIR,
        help="Output model directory.",
    )
    parser.add_argument(
        "--translation-model-path",
        type=Path,
        default=DEFAULT_TRANSLATION_MODEL_PATH,
        help="Path to installed translation artifact.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke training to a temporary directory without publishing release.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic training seed.",
    )

    args = parser.parse_args()

    result = train_multilingual_model(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        model_dir=args.model_dir,
        translation_model_path=args.translation_model_path,
        smoke=args.smoke,
        seed=args.seed,
    )
    print(f"Successfully trained multilingual model to {result.model_dir}")
    print(f"Temperatures: {result.temperatures}")


if __name__ == "__main__":
    main()
