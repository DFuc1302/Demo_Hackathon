from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

METRICS = {
    "binary_classification": {"accuracy", "f1", "f1_macro", "roc_auc"},
    "multiclass_classification": {"accuracy", "f1_macro"},
    "regression": {"rmse", "mae", "r2"},
}


@dataclass(frozen=True)
class TaskConfig:
    train_csv: Path
    target_column: str
    task_type: str
    split_strategy: str
    validation_fraction: float
    seed: int
    metric: str
    prediction_column: str
    predict_csv: Path | None = None
    text_column: str | None = None
    feature_columns: tuple[str, ...] = ()
    id_column: str | None = None
    sample_submission: Path | None = None
    positive_label: str | None = None

    def record(self) -> dict:
        values = asdict(self)
        for key in ("train_csv", "predict_csv", "sample_submission"):
            values[key] = str(values[key]) if values[key] is not None else None
        values["feature_columns"] = list(self.feature_columns)
        return values


def config_from_dict(values: object, base: Path) -> TaskConfig:
    if not isinstance(values, dict):
        raise ValueError("task config must be a YAML mapping")
    allowed = set(TaskConfig.__dataclass_fields__)
    required = {"train_csv", "target_column", "task_type", "split_strategy",
                "validation_fraction", "seed", "metric", "prediction_column"}
    if set(values) - allowed or required - set(values):
        raise ValueError("task config has unknown or missing fields")
    values = dict(values)
    for key in ("target_column", "prediction_column", "task_type", "split_strategy", "metric"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonblank string")
    for key in ("text_column", "id_column", "positive_label"):
        value = values.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonblank string or null")
    features = values.get("feature_columns", [])
    if not isinstance(features, (list, tuple)) or any(not isinstance(c, str) or not c.strip() for c in features):
        raise ValueError("feature_columns must be a list of numeric column names")
    columns = [*features, *([values["text_column"]] if values.get("text_column") else [])]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("declare distinct text/numeric feature columns")
    if values["target_column"] in columns or values.get("id_column") in columns:
        raise ValueError("target/id columns cannot be features")
    if values.get("id_column") in (values["target_column"], values["prediction_column"]):
        raise ValueError("id column must differ from target/prediction columns")
    values["feature_columns"] = tuple(features)
    task, metric = values["task_type"], values["metric"]
    if task not in METRICS or metric not in METRICS[task]:
        raise ValueError("unsupported task/metric combination")
    if values["split_strategy"] not in ("random", "stratified") or (task == "regression" and values["split_strategy"] == "stratified"):
        raise ValueError("unsupported split strategy for this task")
    fraction = values["validation_fraction"]
    if type(fraction) not in (float, int) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if type(values["seed"]) is not int or not 0 <= values["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if metric in ("f1", "roc_auc") and values.get("positive_label") is None:
        raise ValueError("binary f1/roc_auc requires an explicit positive_label")
    if values.get("positive_label") is not None and task != "binary_classification":
        raise ValueError("positive_label is only supported for binary classification")
    for key in ("train_csv", "predict_csv", "sample_submission"):
        value = values.get(key)
        if value is None and key != "train_csv":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonblank file path")
        values[key] = (base / value).resolve()
    return TaskConfig(**values)


def load_config(path: str | Path) -> TaskConfig:
    path = Path(path).resolve()
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML task config: {path}") from exc
    return config_from_dict(values, path.parent)
