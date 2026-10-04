import csv
from pathlib import Path

import pytest

from app.training import load_dataset, train_classifier


DATASET_PATH = Path(__file__).parents[1] / "data" / "prompts.csv"


def test_load_dataset_has_both_labels_and_balanced_examples():
    texts, labels = load_dataset(DATASET_PATH)

    assert len(texts) == len(labels)
    assert len(texts) >= 20
    assert set(labels) == {"benign", "jailbreak"}
    assert labels.count("benign") == labels.count("jailbreak")


def test_training_is_reproducible_and_saves_reloadable_model(tmp_path):
    first_path = tmp_path / "first.joblib"
    second_path = tmp_path / "second.joblib"

    first = train_classifier(DATASET_PATH, first_path)
    second = train_classifier(DATASET_PATH, second_path)

    assert first.roc_auc == second.roc_auc
    assert first.sample_count == second.sample_count
    assert 0.0 <= first.roc_auc <= 1.0
    assert first_path.exists()
    assert second_path.exists()

    import joblib

    artifact = joblib.load(first_path)
    predictions = artifact["pipeline"].predict_proba(["please summarize this article"])
    assert predictions.shape == (1, 2)


def test_load_dataset_rejects_unknown_labels(tmp_path):
    dataset_path = tmp_path / "invalid.csv"
    with dataset_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"text": "normal request", "label": "benign"},
                {"text": "unknown category", "label": "other"},
            ]
        )

    with pytest.raises(ValueError, match="unsupported label"):
        load_dataset(dataset_path)


def test_training_rejects_dataset_without_both_classes(tmp_path):
    dataset_path = tmp_path / "one-class.csv"
    with dataset_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"text": "normal request one", "label": "benign"},
                {"text": "normal request two", "label": "benign"},
            ]
        )

    with pytest.raises(ValueError, match="both classes"):
        train_classifier(dataset_path, tmp_path / "model.joblib")
