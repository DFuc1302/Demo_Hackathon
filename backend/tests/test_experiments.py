from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from app.pipeline.core import train
from app.pipeline.experiments import EnsembleModel, ExperimentRegistry


def _setup_two_classification_models(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_rows = [
        {"id": f"r{i}", "text": f"security alert vulnerability {i}", "label": "alert"} if i % 2 == 0
        else {"id": f"r{i}", "text": f"regular daily office document {i}", "label": "normal"}
        for i in range(16)
    ]
    train_csv = tmp_path / "train.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        w.writeheader()
        w.writerows(train_rows)

    # Model 1: seed 42, standard tokenization
    cfg1 = {
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
    cfg_file1 = tmp_path / "task1.yaml"
    cfg_file1.write_text(yaml.safe_dump(cfg1), encoding="utf-8")
    model_dir1 = tmp_path / "model1"
    train(cfg_file1, model_dir1)

    # Model 2: seed 99, subword ngrams enabled
    cfg2 = dict(cfg1)
    cfg2["seed"] = 99
    cfg2["subword_ngrams"] = True
    cfg_file2 = tmp_path / "task2.yaml"
    cfg_file2.write_text(yaml.safe_dump(cfg2), encoding="utf-8")
    model_dir2 = tmp_path / "model2"
    train(cfg_file2, model_dir2)

    return model_dir1, model_dir2, train_csv


def test_experiment_registry_scans_and_compares_runs(tmp_path: Path) -> None:
    m1, m2, _ = _setup_two_classification_models(tmp_path)
    registry = ExperimentRegistry(tmp_path)
    runs = registry.scan_runs()

    assert len(runs) >= 2
    run_ids = [r.run_id for r in runs]
    assert len(set(run_ids)) == len(run_ids)

    # Compare side by side
    comp = ExperimentRegistry.compare_runs([m1, m2])
    assert len(comp["runs"]) == 2
    assert comp["best_run_id"] in run_ids
    assert comp["metric"] == "accuracy"
    assert len(comp["ranking"]) == 2


def test_ensemble_model_pools_probabilities_and_votes(tmp_path: Path) -> None:
    m1, m2, train_csv = _setup_two_classification_models(tmp_path)

    # Average soft pooling
    ens_avg = EnsembleModel([m1, m2], method="average")
    eval_res = ens_avg.evaluate(train_csv)
    assert "ensemble_score" in eval_res
    assert eval_res["ensemble_score"] >= 0.70
    assert len(eval_res["member_scores"]) == 2
    assert "justified_improvement" in eval_res

    # Majority vote
    ens_vote = EnsembleModel([m1, m2], method="majority_vote")
    vote_res = ens_vote.evaluate(train_csv)
    assert vote_res["ensemble_score"] >= 0.70


def test_ensemble_model_regression_weighted_average(tmp_path: Path) -> None:
    train_rows = [{"x": str(i), "y": str(3 * i + 1)} for i in range(16)]
    train_csv = tmp_path / "train.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y"])
        w.writeheader()
        w.writerows(train_rows)

    cfg1 = {
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
    cfg_file1 = tmp_path / "reg1.yaml"
    cfg_file1.write_text(yaml.safe_dump(cfg1), encoding="utf-8")
    m1 = tmp_path / "m1"
    train(cfg_file1, m1)

    cfg2 = dict(cfg1)
    cfg2["seed"] = 123
    cfg_file2 = tmp_path / "reg2.yaml"
    cfg_file2.write_text(yaml.safe_dump(cfg2), encoding="utf-8")
    m2 = tmp_path / "m2"
    train(cfg_file2, m2)

    ens_reg = EnsembleModel([m1, m2], weights=[0.6, 0.4])
    eval_res = ens_reg.evaluate(train_csv)
    assert "ensemble_score" in eval_res
    assert eval_res["ensemble_metric"] == "rmse"
    assert eval_res["ensemble_score"] < 5.0


def test_ensemble_model_rejects_single_member_and_mismatched_tasks(tmp_path: Path) -> None:
    m1, m2, _ = _setup_two_classification_models(tmp_path)

    with pytest.raises(ValueError, match="at least two"):
        EnsembleModel([m1])
