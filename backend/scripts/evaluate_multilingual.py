from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
import tempfile
from typing import Any

from sklearn.metrics import f1_score, recall_score

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import InferenceEngine, MultilingualInferenceEngine
from app.multilingual_guardrail import MultilingualGuardrailService
from app.translation import TranslationEngine
from app.model_metadata import validate_model_metadata

PILOT_LANGUAGES = ("sw", "ha", "bn")
RELEASE_GATES = {
    "translation": {"f1": 0.80, "recall": 0.85},
    "compare": {"f1": 0.80, "recall": 0.90},
}

def validate_release_gates(metrics: dict, languages: tuple[str, ...] = PILOT_LANGUAGES) -> None:
    for mode, gates in RELEASE_GATES.items():
        for language in languages:
            values = metrics.get(mode, {}).get("test", {}).get(language)
            if not isinstance(values, dict):
                raise ValueError(f"missing {mode} test metrics for {language}")
            for metric, minimum in gates.items():
                value = float(values.get(metric, -1))
                if value < minimum:
                    raise ValueError(f"{mode} mode gate failed for {language}: {metric}={value:.4f}, required={minimum:.2f}")


def evaluate_multilingual_modes(
    *,
    model_dir: str | Path = PROJECT_ROOT / "models" / "multilingual_jailbreak_transformer",
    translation_model_path: str | Path = PROJECT_ROOT / "models" / "translator" / "m2m100_418M",
    english_model_dir: str | Path = PROJECT_ROOT / "models" / "jailbreak_transformer",
    data_dir: str | Path = PROJECT_ROOT / "data" / "multilingual",
    manifest_path: str | Path = PROJECT_ROOT / "data" / "multilingual_manifest.json",
    split: str = "test",
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    multi_engine = MultilingualInferenceEngine(model_dir)
    trans_engine = TranslationEngine(translation_model_path)
    en_engine = InferenceEngine(english_model_dir)
    service = MultilingualGuardrailService(
        multilingual_engine=multi_engine,
        translation_engine=trans_engine,
        english_engine=en_engine,
    )

    data_dir = Path(data_dir)
    split_file = data_dir / f"{split}.jsonl"
    rows = [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    # We evaluate for sw, ha, bn
    pilot_languages = PILOT_LANGUAGES
    metrics: dict[str, dict[str, dict[str, float]]] = {
        "translation": {split: {}},
        "compare": {split: {}},
    }

    for lang in pilot_languages:
        lang_rows = [r for r in rows if r["language"] == lang]
        labels = [r["label"] for r in lang_rows]

        # Translation mode
        trans_preds = []
        compare_preds = []
        for r in lang_rows:
            # Translation mode evaluation
            t_res = service.analyze(r["text"], language=lang, mode="translation")
            trans_preds.append(1 if t_res.decision == "jailbreak" else 0)

            # Compare mode evaluation
            c_res = service.analyze(r["text"], language=lang, mode="compare")
            compare_preds.append(1 if c_res.decision == "jailbreak" else 0)

        t_f1 = float(f1_score(labels, trans_preds, zero_division=0))
        t_rec = float(recall_score(labels, trans_preds, zero_division=0))
        c_f1 = float(f1_score(labels, compare_preds, zero_division=0))
        c_rec = float(recall_score(labels, compare_preds, zero_division=0))

        metrics["translation"][split][lang] = {"f1": t_f1, "recall": t_rec}
        metrics["compare"][split][lang] = {"f1": c_f1, "recall": c_rec}

        if split == "test":
            validate_release_gates({"translation": {"test": {lang: {"f1": t_f1, "recall": t_rec}}}, "compare": {"test": {lang: {"f1": c_f1, "recall": c_rec}}}}, (lang,))
    if split == "test":
        validate_release_gates(metrics)
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        merged = json.loads(json.dumps(meta))
        for mode in ("translation", "compare"):
            merged.setdefault("metrics", {}).setdefault(mode, {}).update(metrics[mode])
        validate_model_metadata(merged, model_dir)
        payload = json.dumps(merged, indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(prefix="metadata.", suffix=".json", dir=model_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, meta_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multilingual, translation, and compare modes.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "multilingual_jailbreak_transformer",
    )
    parser.add_argument(
        "--translation-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "translator" / "m2m100_418M",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
    )
    args = parser.parse_args()
    results = evaluate_multilingual_modes(
        model_dir=args.model_dir,
        translation_model_path=args.translation_model_path,
        split=args.split,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
