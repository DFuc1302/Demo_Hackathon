from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from app.pipeline.core import train
from app.pipeline.robustness import (
    SUPPORTED_TRANSFORMATIONS,
    apply_code_switching,
    apply_paraphrase_synonyms,
    apply_spacing_noise,
    apply_spelling_noise,
    apply_unicode_variation,
    evaluate_robustness,
)


def test_transformations_are_deterministic_and_handle_empty_strings() -> None:
    text = "Please summarize the urgent security notice."
    for name, fn in [
        ("unicode", apply_unicode_variation),
        ("spelling", apply_spelling_noise),
        ("spacing", apply_spacing_noise),
        ("code_switch", apply_code_switching),
        ("paraphrase", apply_paraphrase_synonyms),
    ]:
        out1 = fn(text, seed=42)
        out2 = fn(text, seed=42)
        assert out1 == out2, f"{name} must be deterministic"
        assert len(out1) > 0

        # Empty string handling
        assert fn("", seed=42) == ""


def test_evaluate_robustness_computes_deltas_and_retention_ratios(tmp_path: Path) -> None:
    train_rows = [
        {"id": "1", "text": "urgent system security alert notice", "label": "alert"},
        {"id": "2", "text": "standard office daily recap document", "label": "normal"},
        {"id": "3", "text": "critical vulnerability security alert", "label": "alert"},
        {"id": "4", "text": "routine employee review summary", "label": "normal"},
        {"id": "5", "text": "danger breach intrusion attack alert", "label": "alert"},
        {"id": "6", "text": "regular project schedule timeline", "label": "normal"},
        {"id": "7", "text": "emergency patch required alert immediately", "label": "alert"},
        {"id": "8", "text": "weekly lunch and learn meeting", "label": "normal"},
    ]
    train_csv = tmp_path / "train.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        w.writeheader()
        w.writerows(train_rows)

    cfg = {
        "train_csv": "train.csv",
        "text_column": "text",
        "id_column": "id",
        "target_column": "label",
        "task_type": "binary_classification",
        "split_strategy": "stratified",
        "validation_fraction": 0.25,
        "seed": 42,
        "metric": "accuracy",
        "prediction_column": "prediction",
        "positive_label": "alert",
    }
    cfg_file = tmp_path / "task.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model_dir = tmp_path / "model"
    train(cfg_file, model_dir)

    # Evaluate robustness
    report = evaluate_robustness(model_dir=model_dir, transformations=["spelling_noise", "spacing_noise"])

    assert report["metric"] == "accuracy"
    assert "clean_score" in report
    assert 0.0 <= report["clean_score"] <= 1.0

    transforms = report["transformations"]
    assert "spelling_noise" in transforms
    assert "spacing_noise" in transforms

    for name in ("spelling_noise", "spacing_noise"):
        t_data = transforms[name]
        assert "perturbed_score" in t_data
        assert "score_delta" in t_data
        assert "retention_ratio" in t_data
        assert "sample" in t_data

    assert "subgroup_breakdown" in report


def test_evaluate_robustness_rejects_numeric_only_model(tmp_path: Path) -> None:
    train_rows = [{"x": str(i), "y": str(2 * i)} for i in range(8)]
    train_csv = tmp_path / "train.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y"])
        w.writeheader()
        w.writerows(train_rows)

    cfg = {
        "train_csv": "train.csv",
        "feature_columns": ["x"],
        "target_column": "y",
        "task_type": "regression",
        "split_strategy": "random",
        "validation_fraction": 0.25,
        "seed": 42,
        "metric": "rmse",
        "prediction_column": "prediction",
    }
    cfg_file = tmp_path / "task.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model_dir = tmp_path / "model"
    train(cfg_file, model_dir)

    with pytest.raises(ValueError, match="text_column"):
        evaluate_robustness(model_dir=model_dir)
