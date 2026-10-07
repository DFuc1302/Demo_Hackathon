from __future__ import annotations

import hashlib
import json

import pytest

from app.translation_artifact import (LANGUAGES, MODEL_ID, MODEL_REVISION,
                                      validate_language_codes, validate_translation_artifact)


class Tokenizer:
    src_lang = "en"

    def get_lang_id(self, language):
        return {code: 100 + index for index, code in enumerate(LANGUAGES)}[language]

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.get_lang_id(self.src_lang), 42, 2]}


@pytest.fixture
def artifact(tmp_path):
    files = {"model.safetensors": b"safe fixture", "config.json": b"{}",
             "tokenizer_config.json": b"{}", "special_tokens_map.json": b"{}",
             "sentencepiece.bpe.model": b"fixture", "vocab.json": b"{}"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    metadata = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "license": "mit", "languages": list(LANGUAGES),
                "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    (tmp_path / "artifact.json").write_text(json.dumps(metadata))
    return tmp_path


@pytest.mark.parametrize("field,value", [("model_id", "different/model"), ("model_revision", "a" * 40),
                                           ("license", "apache-2.0"), ("languages", ["en", "sw", "bn"])])
def test_rejects_wrong_provenance(artifact, field, value):
    path = artifact / "artifact.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate_translation_artifact(artifact)


@pytest.mark.parametrize("name", ["pytorch_model.bin", "weights.pt", "cache.pkl", "extra.json"])
def test_rejects_unsafe_or_uninventoried_files(artifact, name):
    (artifact / name).write_bytes(b"untrusted")
    with pytest.raises(ValueError):
        validate_translation_artifact(artifact)


def test_rejects_hash_mismatch(artifact):
    (artifact / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        validate_translation_artifact(artifact)


def test_rejects_symlink(artifact):
    (artifact / "config.json").unlink()
    (artifact / "config.json").symlink_to(artifact / "model.safetensors")
    with pytest.raises(ValueError):
        validate_translation_artifact(artifact)


def test_exact_bidirectional_language_codes_restore_source_language():
    tokenizer = Tokenizer()
    tokenizer.src_lang = "sw"
    validate_language_codes(tokenizer)
    assert tokenizer.src_lang == "sw"


def test_rejects_source_routing_that_ignores_language():
    class BrokenTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [100, 42, 2]}
    with pytest.raises(ValueError, match="source"):
        validate_language_codes(BrokenTokenizer())


def test_rejects_aliased_target_language_tokens():
    class BrokenTokenizer(Tokenizer):
        def get_lang_id(self, language):
            return 100
    with pytest.raises(ValueError, match="distinct"):
        validate_language_codes(BrokenTokenizer())


def test_rejects_missing_tokenizer_even_with_matching_inventory(artifact):
    metadata_path = artifact / "artifact.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["sha256"]["sentencepiece.bpe.model"]
    (artifact / "sentencepiece.bpe.model").unlink()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete"):
        validate_translation_artifact(artifact)


def test_rejects_missing_language_code():
    class MissingTokenizer(Tokenizer):
        def get_lang_id(self, language):
            if language == "ha":
                raise KeyError(language)
            return super().get_lang_id(language)
    with pytest.raises(ValueError, match="missing an exact bidirectional"):
        validate_language_codes(MissingTokenizer())
