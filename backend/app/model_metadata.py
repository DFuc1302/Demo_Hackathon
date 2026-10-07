from __future__ import annotations

import math
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "model_version",
    "model_id",
    "model_revision",
    "dataset_id",
    "dataset_revision",
    "dataset_split_hashes",
    "software_versions",
    "label_mapping",
    "max_token_length",
    "calibration_temperature",
    "classification_threshold",
    "risk_thresholds",
    "seed",
    "sample_counts",
    "class_counts",
    "metrics",
}

REQUIRED_SPLITS = {"train", "validation", "test"}
REQUIRED_SOFTWARE = {"python", "torch", "transformers", "datasets"}
REQUIRED_METRIC_NAMES = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "expected_calibration_error",
    "confusion_matrix",
}
EXPECTED_LABEL_MAPPING = {"0": "benign", "1": "jailbreak"}
HEX_CHARS = set("0123456789abcdef")


def _is_non_bool_int(val: object) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_non_bool_number(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def validate_model_metadata(metadata: object, model_path: str | Path) -> dict[str, Any]:
    path_str = str(model_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"model metadata in {path_str} has invalid metadata: must be a dict")

    schema_version = metadata.get("schema_version")
    if schema_version == 3:
        return _validate_schema_3_metadata(metadata, path_str)
    if not _is_non_bool_int(schema_version) or schema_version != 2:
        raise ValueError(f"model metadata in {path_str} has invalid schema_version: expected 2 or 3")

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in metadata:
            raise ValueError(f"model metadata in {path_str} is missing required field: {field}")
    # non-empty strings
    for str_field in ("model_version", "model_id", "model_revision", "dataset_id", "dataset_revision"):
        val = metadata[str_field]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"model metadata in {path_str} has invalid {str_field}: must be a non-empty string")

    # label_mapping
    if metadata["label_mapping"] != EXPECTED_LABEL_MAPPING:
        raise ValueError(f"model metadata in {path_str} has invalid label_mapping: expected {EXPECTED_LABEL_MAPPING}")

    # max_token_length
    max_token_length = metadata["max_token_length"]
    if not _is_non_bool_int(max_token_length) or max_token_length != 256:
        raise ValueError(f"model metadata in {path_str} has invalid max_token_length: expected 256")

    # calibration_temperature
    temperature = metadata["calibration_temperature"]
    if not _is_non_bool_number(temperature) or not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"model metadata in {path_str} has invalid calibration_temperature: must be finite and positive")

    # classification_threshold
    threshold = metadata["classification_threshold"]
    if not _is_non_bool_number(threshold) or not math.isfinite(float(threshold)) or not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"model metadata in {path_str} has invalid classification_threshold: must be finite in [0, 1]")

    # risk_thresholds
    risk_thresholds = metadata["risk_thresholds"]
    if not isinstance(risk_thresholds, dict) or set(risk_thresholds.keys()) != {"low", "high"}:
        raise ValueError(f"model metadata in {path_str} has invalid risk_thresholds: must contain low and high")
    low = risk_thresholds["low"]
    high = risk_thresholds["high"]
    if not _is_non_bool_number(low) or not _is_non_bool_number(high):
        raise ValueError(f"model metadata in {path_str} has invalid risk_thresholds: low and high must be numeric")
    low_f, high_f = float(low), float(high)
    if not math.isfinite(low_f) or not math.isfinite(high_f) or not (0.0 < low_f < high_f < 1.0):
        raise ValueError(f"model metadata in {path_str} has invalid risk_thresholds: must satisfy 0 < low < high < 1")

    # seed
    seed = metadata["seed"]
    if not _is_non_bool_int(seed):
        raise ValueError(f"model metadata in {path_str} has invalid seed: must be an integer")

    # dataset_split_hashes
    hashes = metadata["dataset_split_hashes"]
    if not isinstance(hashes, dict) or set(hashes.keys()) != REQUIRED_SPLITS:
        raise ValueError(f"model metadata in {path_str} has invalid dataset_split_hashes: must contain exact train/validation/test splits")
    for split, h in hashes.items():
        if not isinstance(h, str) or len(h) != 64 or not all(c in HEX_CHARS for c in h):
            raise ValueError(f"model metadata in {path_str} has invalid dataset_split_hashes: {split} must be 64-char lowercase hex")

    # sample_counts
    sample_counts = metadata["sample_counts"]
    if not isinstance(sample_counts, dict) or set(sample_counts.keys()) != REQUIRED_SPLITS:
        raise ValueError(f"model metadata in {path_str} has invalid sample_counts: must contain exact train/validation/test splits")
    for split, count in sample_counts.items():
        if not _is_non_bool_int(count) or count <= 0:
            raise ValueError(f"model metadata in {path_str} has invalid sample_counts: {split} must be a positive integer")

    # class_counts
    class_counts = metadata["class_counts"]
    if not isinstance(class_counts, dict) or set(class_counts.keys()) != REQUIRED_SPLITS:
        raise ValueError(f"model metadata in {path_str} has invalid class_counts: must contain exact train/validation/test splits")
    for split in REQUIRED_SPLITS:
        sc = class_counts[split]
        if not isinstance(sc, dict) or set(sc.keys()) != {"0", "1"}:
            raise ValueError(f"model metadata in {path_str} has invalid class_counts: {split} must contain 0 and 1 classes")
        c0 = sc["0"]
        c1 = sc["1"]
        if not _is_non_bool_int(c0) or c0 < 0 or not _is_non_bool_int(c1) or c1 < 0:
            raise ValueError(f"model metadata in {path_str} has invalid class_counts: {split} counts must be non-negative integers")
        if c0 + c1 != sample_counts[split]:
            raise ValueError(f"model metadata in {path_str} has invalid class_counts: {split} sum ({c0 + c1}) does not match sample count ({sample_counts[split]})")

    # software_versions
    software = metadata["software_versions"]
    if not isinstance(software, dict) or set(software.keys()) != REQUIRED_SOFTWARE:
        raise ValueError(f"model metadata in {path_str} has invalid software_versions: must contain exact python/torch/transformers/datasets keys")
    for pkg, version in software.items():
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"model metadata in {path_str} has invalid software_versions: {pkg} version must be a non-empty string")

    # metrics
    metrics = metadata["metrics"]
    if not isinstance(metrics, dict) or set(metrics.keys()) != {"validation", "test"}:
        raise ValueError(f"model metadata in {path_str} has invalid metrics: must contain exact validation and test blocks")
    for split in ("validation", "test"):
        block = metrics[split]
        if not isinstance(block, dict):
            raise ValueError(f"model metadata in {path_str} has invalid metrics: {split} must be a dict")
        if set(block.keys()) != REQUIRED_METRIC_NAMES:
            raise ValueError(f"model metadata in {path_str} has invalid metrics: {split} must contain all required metric names")
        for metric_name in REQUIRED_METRIC_NAMES:
            if metric_name == "confusion_matrix":
                cm = block["confusion_matrix"]
                if not isinstance(cm, (list, tuple)) or len(cm) != 2:
                    raise ValueError(f"model metadata in {path_str} has invalid metrics: {split} confusion_matrix must be 2x2")
                for row in cm:
                    if not isinstance(row, (list, tuple)) or len(row) != 2:
                        raise ValueError(f"model metadata in {path_str} has invalid metrics: {split} confusion_matrix must be 2x2")
                    for cell in row:
                        if not _is_non_bool_int(cell) or cell < 0:
                            raise ValueError(f"model metadata in {path_str} has invalid metrics: {split} confusion_matrix cells must be non-negative integers")
            else:
                mval = block[metric_name]
                if not _is_non_bool_number(mval) or not math.isfinite(float(mval)) or not (0.0 <= float(mval) <= 1.0):
                    raise ValueError(f"model metadata in {path_str} has invalid metrics: {split} {metric_name} must be finite in [0, 1]")

    return metadata
SCHEMA_3_RELEASE_GATES = {
    "translation": {"f1": 0.80, "recall": 0.85},
    "compare": {"f1": 0.80, "recall": 0.90},
}


def _validate_schema_3_metric_values(metrics: dict[str, Any], path_str: str) -> None:
    for mode, mode_metrics in metrics.items():
        for split, split_metrics in mode_metrics.items():
            for language, values in split_metrics.items():
                if not isinstance(values, dict):
                    raise ValueError(f"model metadata in {path_str} metrics[{mode}][{split}][{language}] must be a dict")
                for name in ("f1", "recall"):
                    if name in values and (not _is_non_bool_number(values[name]) or not 0.0 <= float(values[name]) <= 1.0):
                        raise ValueError(f"model metadata in {path_str} has invalid {mode} {split} {language} {name}")
                if mode in ("translation", "compare") and split == "test":
                    for name, minimum in SCHEMA_3_RELEASE_GATES[mode].items():
                        if name not in values or float(values[name]) < minimum:
                            raise ValueError(f"model metadata in {path_str} fails {mode} test gate for {language}: {name}")


REQUIRED_SCHEMA_3_TOP_LEVEL = {
    "schema_version",
    "model_version",
    "model_id",
    "model_revision",
    "dataset_id",
    "dataset_revision",
    "dataset_split_hashes",
    "software_versions",
    "label_mapping",
    "max_token_length",
    "languages",
    "calibration_temperatures",
    "classification_threshold",
    "risk_thresholds",
    "seed",
    "sample_counts",
    "class_counts",
    "translation_artifact",
    "metrics",
}


def _validate_schema_3_metadata(metadata: dict[str, Any], path_str: str) -> dict[str, Any]:
    for field in REQUIRED_SCHEMA_3_TOP_LEVEL:
        if field not in metadata:
            raise ValueError(f"model metadata in {path_str} is missing required field: {field}")

    # languages
    languages = metadata["languages"]
    if not isinstance(languages, list) or languages != ["en", "sw", "ha", "bn"]:
        raise ValueError(f"model metadata in {path_str} languages must be ['en', 'sw', 'ha', 'bn']")

    # calibration_temperatures
    temperatures = metadata["calibration_temperatures"]
    if not isinstance(temperatures, dict) or set(temperatures.keys()) != set(languages):
        raise ValueError(f"model metadata in {path_str} calibration_temperatures must contain all languages")
    for lang, temp in temperatures.items():
        if not _is_non_bool_number(temp) or not math.isfinite(float(temp)) or float(temp) <= 0.0:
            raise ValueError(f"model metadata in {path_str} has invalid calibration_temperature for {lang}")

    # translation_artifact provenance identity is part of the schema-3 contract.
    from app.translation_artifact import LANGUAGES, MODEL_ID, MODEL_REVISION
    ta = metadata["translation_artifact"]
    if not isinstance(ta, dict) or set(ta) != {"model_id", "model_revision", "license", "languages", "sha256"}:
        raise ValueError(f"model metadata in {path_str} has invalid translation_artifact")
    if ta["model_id"] != MODEL_ID or ta["model_revision"] != MODEL_REVISION or ta["license"] != "mit" or ta["languages"] != list(LANGUAGES):
        raise ValueError(f"model metadata in {path_str} has mismatched translation_artifact provenance")
    if not isinstance(ta["sha256"], dict) or not ta["sha256"] or any(not isinstance(value, str) or len(value) != 64 for value in ta["sha256"].values()):
        raise ValueError(f"model metadata in {path_str} has invalid translation_artifact hashes")

    # non-empty strings
    for str_field in ("model_version", "model_id", "model_revision", "dataset_id", "dataset_revision"):
        val = metadata[str_field]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"model metadata in {path_str} has invalid {str_field}: must be a non-empty string")

    # label_mapping
    if metadata["label_mapping"] != EXPECTED_LABEL_MAPPING:
        raise ValueError(f"model metadata in {path_str} has invalid label_mapping: expected {EXPECTED_LABEL_MAPPING}")

    # classification_threshold
    threshold = metadata["classification_threshold"]
    if not _is_non_bool_number(threshold) or not math.isfinite(float(threshold)) or not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"model metadata in {path_str} has invalid classification_threshold")

    # risk_thresholds
    risk_thresholds = metadata["risk_thresholds"]
    if not isinstance(risk_thresholds, dict) or set(risk_thresholds.keys()) != {"low", "high"}:
        raise ValueError(f"model metadata in {path_str} has invalid risk_thresholds")

    # sample_counts & class_counts per split and language
    sample_counts = metadata["sample_counts"]
    class_counts = metadata["class_counts"]
    if not isinstance(sample_counts, dict) or set(sample_counts.keys()) != REQUIRED_SPLITS:
        raise ValueError(f"model metadata in {path_str} has invalid sample_counts")

    # metrics: shaped as metrics[mode][split][language]
    metrics = metadata["metrics"]
    if not isinstance(metrics, dict) or not {"multilingual"}.issubset(metrics.keys()):
        raise ValueError(f"model metadata in {path_str} metrics must contain modes")
    for mode, mode_metrics in metrics.items():
        if mode not in ("multilingual", "translation", "compare") or not isinstance(mode_metrics, dict):
            raise ValueError(f"model metadata in {path_str} has invalid mode: {mode}")
        for split, split_metrics in mode_metrics.items():
            if split not in ("validation", "test") or not isinstance(split_metrics, dict):
                raise ValueError(f"model metadata in {path_str} has invalid split: {split}")
            for lang, lang_metrics in split_metrics.items():
                if lang not in languages or not isinstance(lang_metrics, dict):
                    raise ValueError(f"model metadata in {path_str} metrics has invalid language: {lang}")
    _validate_schema_3_metric_values(metrics, path_str)

    return metadata
