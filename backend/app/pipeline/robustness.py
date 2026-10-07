from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.pipeline.baseline import infer, metric_score
from app.pipeline.core import load_artifact
from app.pipeline.data import Dataset, read_dataset
from app.pipeline.language import identify_language

SUPPORTED_TRANSFORMATIONS = (
    "unicode_variation",
    "spelling_noise",
    "spacing_noise",
    "code_switching",
    "paraphrase_synonyms",
)

_SYNONYMS: dict[str, list[str]] = {
    "please": ["kindly", "could you", "pray"],
    "summarize": ["outline", "condense", "recap"],
    "review": ["inspect", "examine", "check"],
    "security": ["protection", "safety", "defense"],
    "system": ["platform", "framework", "service"],
    "alert": ["notice", "warning", "alarm"],
    "normal": ["standard", "regular", "routine"],
    "report": ["summary", "brief", "document"],
    "help": ["assist", "guide", "support"],
    "user": ["operator", "client", "person"],
}

_CODE_SWITCH_MARKERS: dict[str, list[str]] = {
    "sw": ["kwa ufupi", "tafadhali", "kama kawaida", "habari"],
    "ha": ["don Allah", "a takaice", "yau da kullum", "sannu"],
    "bn": ["অনুগ্রহ করে", "সংক্ষেপে", "নিয়মিত", "নমস্কার"],
}


def apply_unicode_variation(text: str, seed: int = 42) -> str:
    """Inject benign Unicode variation (NFKD accents, zero-width spaces)."""
    if not text:
        return text
    rng = random.Random(seed ^ (len(text) << 4))
    decomposed = unicodedata.normalize("NFKD", text)
    # Randomly insert invisible zero-width space in 10% of spaces
    words = decomposed.split(" ")
    out = []
    for w in words:
        if rng.random() < 0.2:
            out.append(w + "\u200B")
        else:
            out.append(w)
    return " ".join(out)


def apply_spelling_noise(text: str, p: float = 0.15, seed: int = 42) -> str:
    """Apply benign spelling noise (letter duplicate, adjacent swap)."""
    if not text or p <= 0:
        return text
    rng = random.Random(seed ^ (len(text) << 5))
    words = text.split()
    out = []
    for word in words:
        if len(word) >= 4 and rng.random() < p:
            chars = list(word)
            idx = rng.randint(1, len(chars) - 2)
            if rng.random() < 0.5:
                # Duplicate a character
                chars.insert(idx, chars[idx])
            else:
                # Swap adjacent characters
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            out.append("".join(chars))
        else:
            out.append(word)
    return " ".join(out)


def apply_spacing_noise(text: str, seed: int = 42) -> str:
    """Apply irregular whitespace noise."""
    if not text:
        return text
    rng = random.Random(seed ^ (len(text) << 6))
    words = text.split()
    out = []
    for i, w in enumerate(words):
        out.append(w)
        if i < len(words) - 1:
            spaces = "  " if rng.random() < 0.3 else " "
            out.append(spaces)
    return "".join(out)


def apply_code_switching(text: str, target_lang: str = "sw", seed: int = 42) -> str:
    """Insert conversational code-switched phrases from target low-resource language."""
    if not text:
        return text
    rng = random.Random(seed ^ (len(text) << 7))
    markers = _CODE_SWITCH_MARKERS.get(target_lang, _CODE_SWITCH_MARKERS["sw"])
    marker = rng.choice(markers)
    if rng.random() < 0.5:
        return f"{marker}, {text}"
    return f"{text} ({marker})"


def apply_paraphrase_synonyms(text: str, seed: int = 42) -> str:
    """Apply conservative synonym substitution preserving semantics."""
    if not text:
        return text
    rng = random.Random(seed ^ (len(text) << 8))
    words = text.split()
    out = []
    for w in words:
        clean = re.sub(r"[^\w]", "", w).lower()
        if clean in _SYNONYMS and rng.random() < 0.4:
            syn = rng.choice(_SYNONYMS[clean])
            # Match title case if original was title case
            replacement = syn.capitalize() if w.istitle() else syn
            out.append(w.lower().replace(clean, replacement))
        else:
            out.append(w)
    return " ".join(out)


TRANSFORMATION_REGISTRY = {
    "unicode_variation": apply_unicode_variation,
    "spelling_noise": apply_spelling_noise,
    "spacing_noise": apply_spacing_noise,
    "code_switching": apply_code_switching,
    "paraphrase_synonyms": apply_paraphrase_synonyms,
}


def evaluate_robustness(
    model_dir: str | Path,
    input_path: str | Path | None = None,
    transformations: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate pipeline baseline model robustness against controlled perturbations."""
    artifact, config = load_artifact(model_dir)
    data_path = Path(input_path).resolve() if input_path is not None else config.train_csv
    dataset = read_dataset(data_path, config, labeled=True)

    if not config.text_column:
        raise ValueError("robustness evaluation requires a task configuration with text_column")

    # 1. Clean baseline evaluation
    clean_preds, clean_probs = infer(dataset, config, artifact["preprocessing"], artifact["model"])
    clean_score = metric_score(dataset, config, artifact["model"], clean_preds, clean_probs)

    chosen_transforms = transformations or list(SUPPORTED_TRANSFORMATIONS)
    results: dict[str, Any] = {}

    for transform_name in chosen_transforms:
        if transform_name not in TRANSFORMATION_REGISTRY:
            raise ValueError(f"unsupported transformation: {transform_name}")
        fn = TRANSFORMATION_REGISTRY[transform_name]

        # Perturb only the text column
        perturbed_rows = []
        for row in dataset.rows:
            p_row = dict(row)
            original_text = row[config.text_column]
            p_row[config.text_column] = fn(original_text, seed=seed)
            perturbed_rows.append(p_row)

        perturbed_dataset = Dataset(columns=dataset.columns, rows=perturbed_rows)
        p_preds, p_probs = infer(perturbed_dataset, config, artifact["preprocessing"], artifact["model"])
        p_score = metric_score(perturbed_dataset, config, artifact["model"], p_preds, p_probs)

        score_delta = round(clean_score - p_score, 4)
        retention = round(p_score / clean_score, 4) if clean_score != 0 else 1.0

        sample_idx = min(len(dataset.rows) - 1, 0)
        results[transform_name] = {
            "perturbed_score": round(p_score, 4),
            "score_delta": score_delta,
            "retention_ratio": retention,
            "sample": {
                "original": dataset.rows[sample_idx][config.text_column],
                "perturbed": perturbed_rows[sample_idx][config.text_column],
            },
        }

    # 2. Subgroup breakdown by detected language
    subgroups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset.rows):
        lang_res = identify_language(row[config.text_column])
        subgroups[lang_res.language].append(idx)

    subgroup_metrics: dict[str, Any] = {}
    for lang, idxs in subgroups.items():
        if len(idxs) >= 2:
            sub_dataset = dataset.subset(idxs)
            sub_preds, sub_probs = infer(sub_dataset, config, artifact["preprocessing"], artifact["model"])
            try:
                sub_score = metric_score(sub_dataset, config, artifact["model"], sub_preds, sub_probs)
                subgroup_metrics[lang] = {
                    "count": len(idxs),
                    "clean_score": round(sub_score, 4),
                }
            except Exception:
                pass

    return {
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "metric": config.metric,
        "clean_score": round(clean_score, 4),
        "seed": seed,
        "transformations": results,
        "subgroup_breakdown": subgroup_metrics,
        "provenance": {
            "run_id": artifact["run"]["run_id"],
            "model_kind": artifact["model"]["kind"],
        },
    }
