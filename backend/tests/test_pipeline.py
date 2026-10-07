from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from app.pipeline.config import load_config
from app.pipeline.core import evaluate, predict, train
from app.pipeline.data import read_dataset, split_dataset


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    config: dict[str, object] = {
        "train_csv": "train.csv",
        "text_column": "text",
        "feature_columns": [],
        "id_column": "id",
        "target_column": "target",
        "task_type": "binary_classification",
        "split_strategy": "stratified",
        "validation_fraction": 0.25,
        "seed": 17,
        "metric": "accuracy",
        "prediction_column": "prediction",
        "positive_label": "yes",
    }
    config.update(overrides)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _classification_rows(labels: tuple[str, ...], count: int = 18) -> list[dict[str, object]]:
    return [
        {
            "id": f"row-{index:02d}",
            "text": f"{labels[index % len(labels)]} indicator token{index}",
            "target": labels[index % len(labels)],
        }
        for index in range(count)
    ]


def _train_binary(tmp_path: Path) -> tuple[Path, Path]:
    _write_csv(tmp_path / "train.csv", _classification_rows(("no", "yes")))
    config_path = _write_config(tmp_path)
    model_dir = tmp_path / "model"
    train(config_path, model_dir)
    return config_path, model_dir


def test_load_config_resolves_relative_paths_and_rejects_unknown_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(data_dir / "train.csv", _classification_rows(("no", "yes"), 8))
    config_path = _write_config(
        tmp_path,
        train_csv="data/train.csv",
        predict_csv="data/predict.csv",
        sample_submission="data/sample.csv",
    )

    config = load_config(config_path)

    assert config.train_csv == (data_dir / "train.csv").resolve()
    assert config.predict_csv == (data_dir / "predict.csv").resolve()
    assert config.sample_submission == (data_dir / "sample.csv").resolve()
    assert list(config.feature_columns) == []

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["surprise"] = True
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown|surprise"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task_type": "ranking"}, "unsupported"),
        ({"task_type": "regression", "metric": "f1", "positive_label": None}, "unsupported|metric"),
        ({"task_type": "multiclass_classification", "metric": "roc_auc", "positive_label": None}, "unsupported|metric"),
        ({"task_type": "regression", "metric": "rmse", "split_strategy": "stratified", "positive_label": None}, "unsupported|split"),
        ({"metric": "f1", "positive_label": None}, "positive_label"),
    ],
)
def test_load_config_rejects_unsupported_task_metric_combinations(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    _write_csv(tmp_path / "train.csv", _classification_rows(("no", "yes"), 8))

    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, **overrides))


@pytest.mark.parametrize("leaking_field", ["text_column", "feature_columns", "id_column"])
def test_load_config_rejects_target_as_a_feature_or_id(tmp_path: Path, leaking_field: str) -> None:
    _write_csv(tmp_path / "train.csv", _classification_rows(("no", "yes"), 8))
    value: object = ["target"] if leaking_field == "feature_columns" else "target"

    with pytest.raises(ValueError, match="target|overlap"):
        load_config(_write_config(tmp_path, **{leaking_field: value}))


@pytest.mark.parametrize(
    ("rows", "overrides", "message"),
    [
        ([{"id": "1", "text": "hello", "target": ""}], {}, "blank|null|target"),
        ([{"id": "1", "text": "hello", "number": "nan", "target": "yes"}], {"feature_columns": ["number"]}, "finite|number"),
        ([{"id": "1", "text": "hello", "target": "inf"}], {"task_type": "regression", "metric": "rmse", "split_strategy": "random", "positive_label": None}, "finite|target"),
        ([{"id": "same", "text": "one", "target": "no"}, {"id": "same", "text": "two", "target": "yes"}], {}, "unique|duplicate|id"),
        ([{"id": "1", "wrong": "hello", "target": "yes"}], {}, "text|column"),
    ],
)
def test_read_dataset_rejects_invalid_required_csv_values(
    tmp_path: Path,
    rows: list[dict[str, object]],
    overrides: dict[str, object],
    message: str,
) -> None:
    csv_path = _write_csv(tmp_path / "train.csv", rows)
    config = load_config(_write_config(tmp_path, **overrides))

    with pytest.raises(ValueError, match=message):
        read_dataset(csv_path, config)


def test_split_is_deterministic_and_partitions_every_row_once(tmp_path: Path) -> None:
    rows = _classification_rows(("no", "yes"), 12)
    _write_csv(tmp_path / "train.csv", rows)
    config = load_config(_write_config(tmp_path, split_strategy="random"))
    dataset = read_dataset(config.train_csv, config)

    first_train, first_validation = split_dataset(dataset, config)
    second_train, second_validation = split_dataset(dataset, config)

    first_train_list = [int(index) for index in first_train]
    first_validation_list = [int(index) for index in first_validation]
    assert first_train_list == [int(index) for index in second_train]
    assert first_validation_list == [int(index) for index in second_validation]
    assert set(first_train_list).isdisjoint(first_validation_list)
    assert sorted(first_train_list + first_validation_list) == list(range(len(rows)))


def test_split_rejects_normalized_feature_leakage(tmp_path: Path) -> None:
    rows = _classification_rows(("no", "yes"), 12)
    rows[0]["text"] = " Shared   Prompt "
    rows[2]["text"] = "shared prompt"
    _write_csv(tmp_path / "train.csv", rows)
    config = load_config(_write_config(tmp_path, split_strategy="random"))
    dataset = read_dataset(config.train_csv, config)

    with pytest.raises(ValueError, match="leak|duplicate"):
        split_dataset(dataset, config)


def test_training_fits_text_and_numeric_preprocessing_on_train_rows_only(tmp_path: Path) -> None:
    rows = [
        {"id": f"row-{index}", "text": f"exclusive{index}", "number": str(index * 10), "target": "yes" if index % 2 else "no"}
        for index in range(12)
    ]
    _write_csv(tmp_path / "train.csv", rows)
    config_path = _write_config(tmp_path, feature_columns=["number"], split_strategy="random")
    config = load_config(config_path)
    dataset = read_dataset(config.train_csv, config)
    expected_train, expected_validation = split_dataset(dataset, config)

    model_dir = tmp_path / "model"
    train(config_path, model_dir)
    artifact = json.loads((model_dir / "artifact.json").read_text(encoding="utf-8"))

    expected_train = [int(index) for index in expected_train]
    expected_validation = [int(index) for index in expected_validation]
    assert artifact["run"]["train_indices"] == expected_train
    assert artifact["run"]["validation_indices"] == expected_validation
    vocabulary = artifact["preprocessing"]["text"]["vocabulary"]
    assert all(f"exclusive{index}" in vocabulary for index in expected_train)
    assert all(f"exclusive{index}" not in vocabulary for index in expected_validation)
    train_mean = sum(float(rows[index]["number"]) for index in expected_train) / len(expected_train)
    assert artifact["preprocessing"]["numeric"]["mean"] == pytest.approx([train_mean])


@pytest.mark.parametrize(
    ("task_type", "metric", "labels", "feature_columns"),
    [
        ("binary_classification", "accuracy", ("no", "yes"), []),
        ("multiclass_classification", "f1_macro", ("red", "green", "blue"), []),
        ("regression", "rmse", (), ["number"]),
    ],
)
def test_real_models_predict_in_input_order_with_ids_and_valid_values(
    tmp_path: Path,
    task_type: str,
    metric: str,
    labels: tuple[str, ...],
    feature_columns: list[str],
) -> None:
    if task_type == "regression":
        train_rows = [
            {"id": f"train-{index}", "number": str(index), "target": str(3 * index + 2)}
            for index in range(16)
        ]
        predict_rows = [
            {"id": "z-last", "number": "9"},
            {"id": "a-first", "number": "1"},
            {"id": "middle", "number": "5"},
        ]
        text_column = None
        positive_label = None
    else:
        train_rows = _classification_rows(labels, 24)
        predict_rows = [
            {"id": "z-last", "text": f"{labels[-1]} unseen"},
            {"id": "a-first", "text": f"{labels[0]} unseen"},
            {"id": "middle", "text": f"{labels[1]} unseen"},
        ]
        text_column = "text"
        positive_label = "yes" if task_type == "binary_classification" else None

    _write_csv(tmp_path / "train.csv", train_rows)
    input_path = _write_csv(tmp_path / "predict.csv", predict_rows)
    config_path = _write_config(
        tmp_path,
        task_type=task_type,
        metric=metric,
        split_strategy="random" if task_type == "regression" else "stratified",
        text_column=text_column,
        feature_columns=feature_columns,
        positive_label=positive_label,
    )
    model_dir = tmp_path / "model"
    train(config_path, model_dir)
    output_path = tmp_path / "predictions.csv"

    prediction_result = predict(model_dir, output_path, input_path=input_path)
    with output_path.open(encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))

    assert prediction_result["rows"] == len(predict_rows)
    assert list(output_rows[0]) == ["id", "prediction"]
    assert [row["id"] for row in output_rows] == ["z-last", "a-first", "middle"]
    if task_type == "regression":
        assert [float(row["prediction"]) for row in output_rows] == pytest.approx([29, 5, 17], abs=2)
    else:
        assert [row["prediction"] for row in output_rows] == [labels[-1], labels[0], labels[1]]


def test_prediction_without_id_writes_only_prediction_column(tmp_path: Path) -> None:
    train_rows = [
        {"text": f"{'yes' if index % 2 else 'no'} token{index}", "target": "yes" if index % 2 else "no"}
        for index in range(12)
    ]
    _write_csv(tmp_path / "train.csv", train_rows)
    config_path = _write_config(tmp_path, id_column=None)
    model_dir = tmp_path / "model"
    train(config_path, model_dir)
    input_path = _write_csv(tmp_path / "predict.csv", [{"text": "yes novel"}, {"text": "no novel"}])
    output_path = tmp_path / "predictions.csv"

    predict(model_dir, output_path, input_path=input_path)

    with output_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["prediction"]
        assert [row["prediction"] for row in reader] == ["yes", "no"]


@pytest.mark.parametrize(
    ("sample_rows", "fieldnames", "message"),
    [
        ([{"id": "z-last", "prediction": ""}, {"id": "a-first", "prediction": ""}], ["prediction", "id"], "column|order"),
        ([{"id": "z-last", "prediction": ""}], ["id", "prediction"], "row|count"),
        ([{"id": "wrong", "prediction": ""}, {"id": "a-first", "prediction": ""}], ["id", "prediction"], "id|order"),
    ],
)
def test_sample_submission_mismatch_fails_before_output_creation(
    tmp_path: Path,
    sample_rows: list[dict[str, str]],
    fieldnames: list[str],
    message: str,
) -> None:
    _, model_dir = _train_binary(tmp_path)
    input_path = _write_csv(
        tmp_path / "predict.csv",
        [{"id": "z-last", "text": "yes example"}, {"id": "a-first", "text": "no example"}],
    )
    sample_path = tmp_path / "sample.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)
    output_path = tmp_path / "predictions.csv"

    with pytest.raises(ValueError, match=message):
        predict(model_dir, output_path, input_path=input_path, sample_submission=sample_path)

    assert not output_path.exists()


def test_predict_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    _, model_dir = _train_binary(tmp_path)
    input_path = _write_csv(tmp_path / "predict.csv", [{"id": "one", "text": "yes example"}])
    output_path = tmp_path / "predictions.csv"
    output_path.write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exist|overwrite"):
        predict(model_dir, output_path, input_path=input_path)

    assert output_path.read_text(encoding="utf-8") == "do not replace\n"


@pytest.mark.parametrize("metric", ["accuracy", "f1", "f1_macro", "roc_auc"])
def test_binary_metrics_use_explicit_positive_class_not_label_sort_order(tmp_path: Path, metric: str) -> None:
    _write_csv(tmp_path / "train.csv", _classification_rows(("no", "yes"), 24))
    config_path = _write_config(tmp_path, metric=metric, positive_label="no")
    run = train(config_path, tmp_path / "model")
    assert run["score"] == pytest.approx(1.0)
    assert evaluate(tmp_path / "model")["score"] == pytest.approx(1.0)


def test_heldout_evaluation_rejects_changed_source_and_fitted_rows(tmp_path: Path) -> None:
    _, model_dir = _train_binary(tmp_path)
    with pytest.raises(ValueError, match="leakage"):
        evaluate(model_dir, input_path=tmp_path / "train.csv")
    with (tmp_path / "train.csv").open("a") as stream:
        stream.write("new-row,no novel,no\n")
    with pytest.raises(ValueError, match="hash changed"):
        evaluate(model_dir)


def test_tampered_json_model_is_rejected_before_prediction(tmp_path: Path) -> None:
    _, model_dir = _train_binary(tmp_path)
    path = model_dir / "artifact.json"
    artifact = json.loads(path.read_text())
    artifact["model"]["intercept"][0] += 100
    path.write_text(json.dumps(artifact))
    input_path = _write_csv(tmp_path / "predict.csv", [{"id": "one", "text": "no example"}])
    with pytest.raises(ValueError, match="hash mismatch"):
        predict(model_dir, tmp_path / "predictions.csv", input_path=input_path)
    assert not (tmp_path / "predictions.csv").exists()


def test_training_refuses_existing_run_without_changing_it(tmp_path: Path) -> None:
    config_path, model_dir = _train_binary(tmp_path)
    before = (model_dir / "artifact.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        train(config_path, model_dir)
    assert (model_dir / "artifact.json").read_bytes() == before
