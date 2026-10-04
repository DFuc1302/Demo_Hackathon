from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SUPPORTED_LABELS = frozenset({"benign", "jailbreak"})
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainingResult:
    roc_auc: float
    sample_count: int
    class_counts: dict[str, int]
    model_path: Path


def load_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    dataset_path = Path(path)
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"text", "label"} - set(reader.fieldnames):
            raise ValueError("dataset must contain text and label columns")

        texts: list[str] = []
        labels: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text:
                raise ValueError(f"row {row_number} has empty text")
            if label not in SUPPORTED_LABELS:
                raise ValueError(f"row {row_number} has unsupported label: {label!r}")
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError("dataset must contain at least one row")
    return texts, labels


def _validate_labels(labels: Iterable[str]) -> dict[str, int]:
    counts = Counter(labels)
    if set(counts) != SUPPORTED_LABELS:
        raise ValueError("dataset must contain both classes: benign and jailbreak")
    if min(counts.values()) < 2:
        raise ValueError("each class must contain at least two examples")
    return dict(sorted(counts.items()))


def train_classifier(
    dataset_path: str | Path,
    model_path: str | Path,
    *,
    test_size: float = 0.25,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainingResult:
    texts, labels = load_dataset(dataset_path)
    class_counts = _validate_labels(labels)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    pipeline.fit(train_texts, train_labels)

    jailbreak_index = list(pipeline.classes_).index("jailbreak")
    probabilities = pipeline.predict_proba(test_texts)[:, jailbreak_index]
    roc_auc = float(roc_auc_score([label == "jailbreak" for label in test_labels], probabilities))

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "labels": sorted(SUPPORTED_LABELS),
            "random_state": random_state,
            "roc_auc": roc_auc,
        },
        output_path,
    )
    return TrainingResult(roc_auc, len(texts), class_counts, output_path)
