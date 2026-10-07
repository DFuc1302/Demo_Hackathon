from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_ID = "facebook/m2m100_418M"
MODEL_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
LANGUAGES = ("en", "sw", "ha", "bn")
UNSAFE_SUFFIXES = {".bin", ".pt", ".pkl", ".pickle", ".ot"}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_language_codes(tokenizer) -> None:
    original = tokenizer.src_lang
    try:
        ids = [tokenizer.get_lang_id(code) for code in LANGUAGES]
        if any(type(value) is not int for value in ids) or len(set(ids)) != len(LANGUAGES):
            raise ValueError("translator language tokens must be distinct integers")
        for code, token in zip(LANGUAGES, ids):
            tokenizer.src_lang = code
            encoded = tokenizer("test", truncation=False)["input_ids"]
            if not encoded or encoded[0] != token:
                raise ValueError(f"translator does not support exact source language {code}")
    except (KeyError, AttributeError) as exc:
        raise ValueError("translator is missing an exact bidirectional language code") from exc
    finally:
        tokenizer.src_lang = original


def validate_translation_artifact(path: str | Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("translator artifact must be a local directory")
    try:
        metadata_path = path / "artifact.json"
        if metadata_path.is_symlink():
            raise ValueError("translator inventory cannot be a symlink")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("translator artifact inventory is missing or invalid") from exc
    expected = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "license": "mit", "languages": list(LANGUAGES)}
    if not isinstance(metadata, dict) or set(metadata) != {*expected, "sha256"}:
        raise ValueError("translator artifact inventory has invalid fields")
    if any(metadata[key] != value for key, value in expected.items()):
        raise ValueError("translator artifact identity, revision, license, or languages mismatch")
    inventory = metadata["sha256"]
    required_files = {"model.safetensors", "config.json", "tokenizer_config.json",
                      "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.json"}
    if not isinstance(inventory, dict) or not required_files <= inventory.keys():
        raise ValueError("translator safetensors/config/tokenizer inventory is incomplete")
    files = set()
    for item in path.rglob("*"):
        if item.is_symlink() or item.suffix.lower() in UNSAFE_SUFFIXES:
            raise ValueError("translator contains a symlink or unsafe artifact format")
        if item.is_file() and item != metadata_path:
            name = item.relative_to(path).as_posix()
            files.add(name)
            if name not in inventory or file_sha256(item) != inventory[name]:
                raise ValueError(f"translator artifact hash mismatch: {name}")
    if files != inventory.keys():
        raise ValueError("translator inventory contains missing files")
    return metadata
