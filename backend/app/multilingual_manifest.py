from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.translation_artifact import LANGUAGES, file_sha256, validate_translation_artifact

CLASSIFIER_MODEL_ID = "FacebookAI/xlm-roberta-base"
CLASSIFIER_MODEL_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
SPLITS = ("train", "validation", "test")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "source_v2_manifest_sha256",
    "dataset_id",
    "dataset_revision",
    "languages",
    "model_id",
    "model_revision",
    "license",
    "translation_artifact",
    "splits",
}
_SPLIT_FIELDS = {
    "prepared_sha256",
    "rows",
    "source_groups",
    "language_label_counts",
    "decontamination",
}
_DECONTAMINATION_FIELDS = {
    "input_groups",
    "empty_groups",
    "collision_groups",
    "collision_groups_by_language",
    "rebalance_groups",
    "removed_groups",
}


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_source_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("source V2 manifest is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("source V2 manifest is missing or invalid")
    return manifest


def _validate_split(split_name: str, split: Any) -> None:
    prefix = f"multilingual {split_name} split"
    if not isinstance(split, dict) or set(split) != _SPLIT_FIELDS:
        raise ValueError(f"{prefix} has invalid fields")
    if not _is_sha256(split["prepared_sha256"]):
        raise ValueError(f"{prefix} prepared hash is invalid")

    rows = split["rows"]
    source_groups = split["source_groups"]
    if not _is_count(rows) or not _is_count(source_groups) or rows != 4 * source_groups:
        raise ValueError(f"{prefix} row and source-group counts are inconsistent")
    if source_groups == 0:
        raise ValueError(f"{prefix} must be nonempty")

    language_counts = split["language_label_counts"]
    if not isinstance(language_counts, dict) or set(language_counts) != set(LANGUAGES):
        raise ValueError(f"{prefix} language-label counts have invalid fields")
    expected_label_counts: dict[str, int] | None = None
    for language in LANGUAGES:
        counts = language_counts[language]
        if (
            not isinstance(counts, dict)
            or set(counts) != {"0", "1"}
            or not all(_is_count(value) for value in counts.values())
            or counts["0"] != counts["1"]
            or counts["0"] + counts["1"] != source_groups
        ):
            raise ValueError(f"{prefix} has inconsistent or unbalanced counts for {language}")
        if expected_label_counts is None:
            expected_label_counts = counts
        elif counts != expected_label_counts:
            raise ValueError(f"{prefix} language-label counts do not describe complete groups")

    decontamination = split["decontamination"]
    if not isinstance(decontamination, dict) or set(decontamination) != _DECONTAMINATION_FIELDS:
        raise ValueError(f"{prefix} decontamination has invalid fields")
    scalar_fields = _DECONTAMINATION_FIELDS - {"collision_groups_by_language"}
    if not all(_is_count(decontamination[field]) for field in scalar_fields):
        raise ValueError(f"{prefix} decontamination counts are invalid")
    collision_by_language = decontamination["collision_groups_by_language"]
    if (
        not isinstance(collision_by_language, dict)
        or set(collision_by_language) != set(LANGUAGES)
        or not all(_is_count(value) for value in collision_by_language.values())
        or any(value > decontamination["collision_groups"] for value in collision_by_language.values())
    ):
        raise ValueError(f"{prefix} collision counts by language are invalid")

    removed_groups = decontamination["removed_groups"]
    categorized_removals = (
        decontamination["empty_groups"]
        + decontamination["collision_groups"]
        + decontamination["rebalance_groups"]
    )
    if (
        removed_groups != categorized_removals
        or decontamination["input_groups"] - removed_groups != source_groups
    ):
        raise ValueError(f"{prefix} decontamination removal accounting is inconsistent")
    input_groups = decontamination["input_groups"]
    if input_groups and removed_groups / input_groups > 0.10:
        raise ValueError(f"{prefix} decontamination removed more than 10% of source groups")


def validate_multilingual_manifest(
    manifest: Any,
    *,
    source_v2_manifest_path: str | Path,
    translation_model_path: str | Path,
) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != _TOP_LEVEL_FIELDS:
        raise ValueError("multilingual manifest has invalid fields")
    if manifest["schema_version"] != 3:
        raise ValueError("multilingual manifest schema_version must be 3")
    if manifest["languages"] != list(LANGUAGES):
        raise ValueError("multilingual manifest languages mismatch")
    if (
        manifest["model_id"] != CLASSIFIER_MODEL_ID
        or manifest["model_revision"] != CLASSIFIER_MODEL_REVISION
        or manifest["license"] != "mit"
    ):
        raise ValueError("multilingual classifier identity, revision, or license mismatch")

    source_v2_manifest_path = Path(source_v2_manifest_path)
    if (
        not _is_sha256(manifest["source_v2_manifest_sha256"])
        or not source_v2_manifest_path.is_file()
        or file_sha256(source_v2_manifest_path) != manifest["source_v2_manifest_sha256"]
    ):
        raise ValueError("source V2 manifest hash mismatch")
    source_manifest = _load_source_manifest(source_v2_manifest_path)
    if (
        source_manifest.get("schema_version") != 2
        or source_manifest.get("license") != "mit"
        or source_manifest.get("language") != "en"
        or manifest["dataset_id"] != source_manifest.get("dataset_id")
        or manifest["dataset_revision"] != source_manifest.get("dataset_revision")
    ):
        raise ValueError("multilingual manifest source V2 provenance mismatch")

    artifact = validate_translation_artifact(translation_model_path)
    if manifest["translation_artifact"] != artifact:
        raise ValueError("multilingual manifest translation artifact does not match the installed artifact")

    splits = manifest["splits"]
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("multilingual manifest splits must be exactly train, validation, and test")
    for split_name in SPLITS:
        _validate_split(split_name, splits[split_name])
    return manifest
