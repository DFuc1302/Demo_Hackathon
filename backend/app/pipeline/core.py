from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn

from app.pipeline.baseline import fit_baseline, infer, metric_score
from app.pipeline.config import TaskConfig, config_from_dict, load_config
from app.pipeline.data import feature_hashes, read_dataset, split_dataset, summary

REPO_ROOT = Path(__file__).resolve().parents[3]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_output(path: Path) -> None:
    path = path.resolve()
    if path.exists():
        raise ValueError(f"output already exists; choose a new path: {path}")
    if path.is_relative_to(REPO_ROOT) and not any(path.is_relative_to(REPO_ROOT / folder) for folder in ("backend/outputs", "backend/submissions")):
        raise ValueError("repository outputs must be under backend/outputs or backend/submissions; source/data/models are protected")


def train(config_path: str | Path, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir).resolve()
    _check_output(output_dir)
    started = time.monotonic()
    config = load_config(config_path)
    input_hash = _hash_file(config.train_csv)
    data = read_dataset(config.train_csv, config)
    train_indices, validation_indices = split_dataset(data, config)
    training, validation = data.subset(train_indices), data.subset(validation_indices)
    preprocessing, model = fit_baseline(training, config)
    predictions, probabilities = infer(validation, config, preprocessing, model)
    score = metric_score(validation, config, model, predictions, probabilities)
    config_record = config.record()
    model_hash = hashlib.sha256(_json_bytes({"config": config_record, "preprocessing": preprocessing, "model": model})).hexdigest()
    config_hash = hashlib.sha256(_json_bytes(config_record)).hexdigest()
    split = {"train_indices": train_indices.tolist(), "validation_indices": validation_indices.tolist()}
    run = {
        **split, "run_id": hashlib.sha256(_json_bytes({"input": input_hash, "config": config_hash, "model": model_hash, **split})).hexdigest(),
        "input_sha256": input_hash, "config_sha256": config_hash, "model_sha256": model_hash,
        "train_feature_hashes": feature_hashes(training, config),
        "summary": summary(data, config), "seed": config.seed,
        "metric": config.metric, "score": score, "higher_is_better": config.metric not in ("rmse", "mae"),
        "train_rows": len(training.rows), "validation_rows": len(validation.rows),
        "runtime_seconds": time.monotonic() - started,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    artifact = {"schema_version": 1, "config": config_record, "preprocessing": preprocessing, "model": model, "run": run}
    # Validate before publication; serialization contains only JSON data, never pickle.
    _validate_artifact(artifact)
    if _hash_file(config.train_csv) != input_hash:
        raise ValueError("training input changed during fitting; no artifact was published")
    content = _json_bytes(artifact)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()  # Exclusive creation protects prior runs, including concurrent writers.
    with (output_dir / "artifact.json").open("xb") as stream:
        stream.write(content)
    return run


def _validate_artifact(artifact: object) -> TaskConfig:
    try:
        if not isinstance(artifact, dict) or set(artifact) != {"schema_version", "config", "preprocessing", "model", "run"} or type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1:
            raise ValueError("unsupported artifact schema")
        config = config_from_dict(artifact["config"], Path.cwd())
        preprocessing, model, run = artifact["preprocessing"], artifact["model"], artifact["run"]
        dimension = len(config.feature_columns)
        numeric = preprocessing["numeric"]
        if config.feature_columns:
            mean, scale = np.asarray(numeric["mean"], dtype=float), np.asarray(numeric["scale"], dtype=float)
            if mean.shape != (dimension,) or scale.shape != (dimension,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("invalid numeric preprocessing state")
        elif numeric is not None:
            raise ValueError("unexpected numeric preprocessing state")
        text = preprocessing["text"]
        if config.text_column:
            vocabulary, idf = text["vocabulary"], np.asarray(text["idf"], dtype=float)
            if not isinstance(vocabulary, dict) or not vocabulary or any(not isinstance(k, str) or type(v) is not int for k, v in vocabulary.items()) or set(vocabulary.values()) != set(range(len(vocabulary))) or idf.shape != (len(vocabulary),) or not np.isfinite(idf).all() or (idf <= 0).any():
                raise ValueError("invalid text preprocessing state")
            dimension += len(vocabulary)
        elif text is not None:
            raise ValueError("unexpected text preprocessing state")
        classes = model["classes"]
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c.strip() for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid model classes")
        if config.task_type == "regression":
            rows = 1
            if model["kind"] != "ridge" or classes:
                raise ValueError("regression artifact requires ridge coefficients")
        else:
            if model["kind"] != "logistic_regression" or (config.task_type == "binary_classification" and len(classes) != 2) or (config.task_type == "multiclass_classification" and len(classes) < 3):
                raise ValueError("classification artifact has invalid class cardinality")
            if config.positive_label is not None and config.positive_label not in classes:
                raise ValueError("artifact is missing positive_label")
            rows = 1 if len(classes) == 2 else len(classes)
        coefficients, intercept = np.asarray(model["coefficients"], dtype=float), np.asarray(model["intercept"], dtype=float)
        if coefficients.shape != (rows, dimension) or intercept.shape != (rows,) or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
            raise ValueError("invalid coefficient/intercept shape or values")
        expected_hash = hashlib.sha256(_json_bytes({"config": artifact["config"], "preprocessing": preprocessing, "model": model})).hexdigest()
        if run["model_sha256"] != expected_hash or run["config_sha256"] != hashlib.sha256(_json_bytes(artifact["config"])).hexdigest():
            raise ValueError("artifact model/config hash mismatch")
        training, validation = run["train_indices"], run["validation_indices"]
        all_indices = training + validation
        if not training or not validation or any(type(i) is not int or i < 0 for i in all_indices) or len(set(all_indices)) != len(all_indices) or set(all_indices) != set(range(len(all_indices))):
            raise ValueError("invalid persisted split indices")
        if not isinstance(run["train_feature_hashes"], list) or len(run["train_feature_hashes"]) != len(training):
            raise ValueError("invalid training feature hash inventory")
        return config
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("invalid pipeline artifact structure") from exc


def load_artifact(model_dir: str | Path) -> tuple[dict, TaskConfig]:
    path = Path(model_dir) / "artifact.json"
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"invalid JSON pipeline artifact: {path}") from exc
    config = _validate_artifact(artifact)
    return artifact, config


def evaluate(model_dir: str | Path, input_path: str | Path | None = None) -> dict:
    artifact, config = load_artifact(model_dir)
    path = Path(input_path).resolve() if input_path is not None else config.train_csv
    data = read_dataset(path, config)
    run = artifact["run"]
    input_hash = _hash_file(path)
    if input_path is None:
        if input_hash != run["input_sha256"]:
            raise ValueError("original training CSV hash changed; cannot reproduce heldout evaluation")
        data = data.subset(run["validation_indices"])
    if set(feature_hashes(data, config)) & set(run["train_feature_hashes"]):
        raise ValueError("evaluation input has feature leakage from fitted training rows")
    if config.task_type != "regression" and not {row[config.target_column] for row in data.rows} <= set(artifact["model"]["classes"]):
        raise ValueError("evaluation input has target classes unknown to the fitted model")
    predictions, probabilities = infer(data, config, artifact["preprocessing"], artifact["model"])
    score = metric_score(data, config, artifact["model"], predictions, probabilities)
    return {"metric": config.metric, "score": score, "rows": len(data.rows), "input_sha256": input_hash,
            "run_id": run["run_id"], "higher_is_better": config.metric not in ("rmse", "mae")}


def predict(model_dir: str | Path, output_path: str | Path, input_path: str | Path | None = None,
            sample_submission: str | Path | None = None) -> dict:
    artifact, config = load_artifact(model_dir)
    path = Path(input_path).resolve() if input_path is not None else config.predict_csv
    if path is None:
        raise ValueError("prediction requires --input or predict_csv in the task config")
    output = Path(output_path).resolve()
    _check_output(output)
    data = read_dataset(path, config, labeled=False)
    predictions, _ = infer(data, config, artifact["preprocessing"], artifact["model"])
    columns = ([config.id_column] if config.id_column else []) + [config.prediction_column]
    sample_path = Path(sample_submission).resolve() if sample_submission is not None else config.sample_submission
    if sample_path is not None:
        with sample_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != columns:
                raise ValueError("sample submission column order/names do not match configured output")
            sample_rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in sample_rows):
            raise ValueError("sample submission has invalid row shape")
        if len(sample_rows) != len(data.rows):
            raise ValueError("sample submission row count does not match prediction input")
        if config.id_column and [row[config.id_column] for row in sample_rows] != [row[config.id_column] for row in data.rows]:
            raise ValueError("sample submission ID order does not match prediction input")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row, value in zip(data.rows, predictions):
        writer.writerow(([row[config.id_column]] if config.id_column else []) + [value])
    content = buffer.getvalue().encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(content)
    return {"rows": len(data.rows), "columns": columns, "output": str(output),
            "sha256": hashlib.sha256(content).hexdigest(), "run_id": artifact["run"]["run_id"]}
