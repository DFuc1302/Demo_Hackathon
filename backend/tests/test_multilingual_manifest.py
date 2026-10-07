from __future__ import annotations

import hashlib
import json

import pytest

from app.multilingual_manifest import (
    CLASSIFIER_MODEL_ID,
    CLASSIFIER_MODEL_REVISION,
    validate_multilingual_manifest,
)
from app.translation_artifact import LANGUAGES, MODEL_ID, MODEL_REVISION


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def manifest_inputs(tmp_path):
    v2_manifest = {
        "schema_version": 2,
        "dataset_id": "owner/dataset",
        "dataset_revision": "d" * 40,
        "license": "mit",
        "language": "en",
    }
    v2_path = tmp_path / "v2_manifest.json"
    v2_path.write_text(json.dumps(v2_manifest), encoding="utf-8")

    artifact_path = tmp_path / "translator"
    artifact_path.mkdir()
    files = {
        "model.safetensors": b"safe fixture",
        "config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "sentencepiece.bpe.model": b"fixture",
        "vocab.json": b"{}",
    }
    for name, content in files.items():
        (artifact_path / name).write_bytes(content)
    artifact = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": "mit",
        "languages": list(LANGUAGES),
        "sha256": {name: _sha256(content) for name, content in files.items()},
    }
    (artifact_path / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")

    split = {
        "prepared_sha256": "a" * 64,
        "rows": 8,
        "source_groups": 2,
        "language_label_counts": {
            language: {"0": 1, "1": 1} for language in LANGUAGES
        },
        "decontamination": {
            "input_groups": 2,
            "empty_groups": 0,
            "collision_groups": 0,
            "collision_groups_by_language": {language: 0 for language in LANGUAGES},
            "rebalance_groups": 0,
            "removed_groups": 0,
        },
    }
    manifest = {
        "schema_version": 3,
        "source_v2_manifest_sha256": _sha256(v2_path.read_bytes()),
        "dataset_id": v2_manifest["dataset_id"],
        "dataset_revision": v2_manifest["dataset_revision"],
        "languages": list(LANGUAGES),
        "model_id": CLASSIFIER_MODEL_ID,
        "model_revision": CLASSIFIER_MODEL_REVISION,
        "license": "mit",
        "translation_artifact": artifact,
        "splits": {name: json.loads(json.dumps(split)) for name in ("train", "validation", "test")},
    }
    return manifest, v2_path, artifact_path


def _validate(manifest_inputs):
    manifest, v2_path, artifact_path = manifest_inputs
    return validate_multilingual_manifest(
        manifest,
        source_v2_manifest_path=v2_path,
        translation_model_path=artifact_path,
    )


def test_accepts_exact_schema3_manifest_contract(manifest_inputs):
    manifest, _, _ = manifest_inputs
    assert _validate(manifest_inputs) == manifest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("dataset_id", "other/dataset"),
        ("dataset_revision", "e" * 40),
        ("languages", ["en", "sw", "bn", "ha"]),
        ("model_id", "other/model"),
        ("model_revision", "e" * 40),
        ("license", "apache-2.0"),
    ],
)
def test_rejects_schema_or_provenance_mismatch(manifest_inputs, field, value):
    manifest, _, _ = manifest_inputs
    manifest[field] = value
    with pytest.raises(ValueError):
        _validate(manifest_inputs)


def test_rejects_source_v2_manifest_hash_mismatch(manifest_inputs):
    manifest, _, _ = manifest_inputs
    manifest["source_v2_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source V2 manifest hash"):
        _validate(manifest_inputs)


def test_rejects_translation_inventory_not_identical_to_installed_artifact(manifest_inputs):
    manifest, _, _ = manifest_inputs
    manifest["translation_artifact"]["sha256"]["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="translation artifact"):
        _validate(manifest_inputs)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("rows",), 7),
        (("source_groups",), 3),
        (("language_label_counts", "bn", "1"), 0),
        (("decontamination", "input_groups"), 3),
        (("decontamination", "removed_groups"), 1),
        (("decontamination", "collision_groups_by_language", "sw"), -1),
    ],
)
def test_rejects_inconsistent_split_counts(manifest_inputs, path, value):
    manifest, _, _ = manifest_inputs
    target = manifest["splits"]["train"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="train"):
        _validate(manifest_inputs)


def test_rejects_more_than_ten_percent_removed_groups(manifest_inputs):
    manifest, _, _ = manifest_inputs
    split = manifest["splits"]["train"]
    split["source_groups"] = 16
    split["rows"] = 64
    split["language_label_counts"] = {
        language: {"0": 8, "1": 8} for language in LANGUAGES
    }
    split["decontamination"]["input_groups"] = 20
    split["decontamination"]["empty_groups"] = 4
    split["decontamination"]["removed_groups"] = 4
    with pytest.raises(ValueError, match="10%"):
        _validate(manifest_inputs)


def test_rejects_unknown_fields_at_every_manifest_level(manifest_inputs):
    manifest, _, _ = manifest_inputs
    manifest["splits"]["train"]["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        _validate(manifest_inputs)


def test_rejects_empty_split(manifest_inputs):
    manifest, _, _ = manifest_inputs
    split = manifest["splits"]["validation"]
    split["rows"] = 0
    split["source_groups"] = 0
    split["language_label_counts"] = {
        language: {"0": 0, "1": 0} for language in LANGUAGES
    }
    split["decontamination"]["input_groups"] = 0
    with pytest.raises(ValueError, match="nonempty"):
        _validate(manifest_inputs)


def test_accepts_split_mapping_regardless_of_key_order(manifest_inputs):
    manifest, _, _ = manifest_inputs
    splits = manifest["splits"]
    manifest["splits"] = {name: splits[name] for name in ("test", "train", "validation")}
    assert _validate(manifest_inputs) == manifest
