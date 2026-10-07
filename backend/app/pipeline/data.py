from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from app.pipeline.config import TaskConfig


@dataclass(frozen=True)
class Dataset:
    columns: list[str]
    rows: list[dict[str, str]]

    def subset(self, indices) -> Dataset:
        return Dataset(self.columns, [self.rows[int(index)] for index in indices])


def read_dataset(path: str | Path, config: TaskConfig, labeled: bool = True) -> Dataset:
    path = Path(path)
    required = set(config.feature_columns)
    if config.text_column:
        required.add(config.text_column)
    if config.id_column:
        required.add(config.id_column)
    if labeled:
        required.add(config.target_column)
    rows = []
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns) or any(not c.strip() for c in columns):
            raise ValueError(f"{path}: CSV requires distinct nonblank column names")
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: row {number} has an invalid column count")
            for column in required:
                if not row[column].strip():
                    raise ValueError(f"{path}: row {number} column {column} is null/blank")
            numeric_columns = list(config.feature_columns)
            if labeled and config.task_type == "regression":
                numeric_columns.append(config.target_column)
            for column in numeric_columns:
                try:
                    value = float(row[column])
                except ValueError as exc:
                    raise ValueError(f"{path}: row {number} column {column} must be numeric") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}: row {number} column {column} must be finite")
            if config.id_column:
                identity = row[config.id_column]
                if identity in ids:
                    raise ValueError(f"{path}: duplicate ID at row {number}")
                ids.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: CSV has no data rows")
    data = Dataset(columns, rows)
    if labeled and config.task_type != "regression":
        classes = set(targets(data, config))
        if config.task_type == "binary_classification" and len(classes) != 2:
            raise ValueError("binary classification requires exactly two target classes")
        if config.task_type == "multiclass_classification" and len(classes) < 3:
            raise ValueError("multiclass classification requires at least three target classes")
        if config.positive_label is not None and config.positive_label not in classes:
            raise ValueError("positive_label is absent from target classes")
    return data


def targets(data: Dataset, config: TaskConfig) -> np.ndarray:
    values = [row[config.target_column] for row in data.rows]
    return np.asarray(values, dtype=float if config.task_type == "regression" else str)


def feature_hashes(data: Dataset, config: TaskConfig) -> list[str]:
    result = []
    for row in data.rows:
        values = [float(row[c]) for c in config.feature_columns]
        if config.text_column:
            # Same conservative whitespace/casefold key used for corpus leakage checks.
            values.append(" ".join(row[config.text_column].split()).casefold())
        raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
        result.append(hashlib.sha256(raw).hexdigest())
    return result


def split_dataset(data: Dataset, config: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    labels = targets(data, config)
    indices = np.arange(len(data.rows))
    stratify = labels if config.split_strategy == "stratified" else None
    try:
        train, validation = train_test_split(indices, test_size=config.validation_fraction,
                                            random_state=config.seed, stratify=stratify)
    except ValueError as exc:
        raise ValueError(f"cannot create requested train/validation split: {exc}") from exc
    if config.task_type != "regression":
        all_classes = set(labels)
        if set(labels[train]) != all_classes or set(labels[validation]) != all_classes:
            raise ValueError("each split must contain every target class; use stratified splitting or more rows")
    if config.metric == "r2" and len(validation) < 2:
        raise ValueError("r2 requires at least two validation rows")
    hashes = feature_hashes(data, config)
    if {hashes[i] for i in train} & {hashes[i] for i in validation}:
        raise ValueError("feature leakage across train/validation; deduplicate or use a task-specific grouped split")
    # Persist indices in source order for repeatable reports and heldout evaluation.
    return np.sort(train), np.sort(validation)


def summary(data: Dataset, config: TaskConfig) -> dict:
    result = {
        "rows": len(data.rows), "columns": data.columns,
        "null_counts": {c: sum(not row[c].strip() for row in data.rows) for c in data.columns},
        "text_column": config.text_column, "numeric_columns": list(config.feature_columns),
        "numeric_ranges": {},
    }
    for column in config.feature_columns:
        values = [float(row[column]) for row in data.rows]
        result["numeric_ranges"][column] = {"min": min(values), "max": max(values)}
    if config.task_type == "regression":
        values = targets(data, config)
        result["target_summary"] = {"min": float(values.min()), "max": float(values.max()),
                                    "mean": float(values.mean())}
    else:
        result["class_counts"] = dict(sorted(Counter(targets(data, config)).items()))
    return result
