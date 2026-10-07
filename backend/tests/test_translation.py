from __future__ import annotations

import pytest

from app.translation import TranslationEngine, TranslationResult


class FakeTokenizer:
    src_lang = "en"
    model_max_length = 1024

    def __init__(self):
        self._langs = {"en": 100, "sw": 101, "ha": 102, "bn": 103}

    def get_lang_id(self, code: str) -> int:
        return self._langs[code]

    def __call__(self, text: str, truncation: bool = False, return_tensors: str | None = None, **kwargs):
        tokens = [self._langs[self.src_lang]] + [ord(c) % 50 + 1 for c in text]
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([tokens])}
        return {"input_ids": tokens}

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        return ["translated sentence"]


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, *args, **kwargs):
        import torch
        return torch.tensor([[100, 20, 21, 2]])


def test_translation_engine_initialization_and_translate_to_english(tmp_path, monkeypatch):
    import json
    from app.translation_artifact import MODEL_ID, MODEL_REVISION, LANGUAGES

    # Create dummy artifact
    (tmp_path / "model.safetensors").write_bytes(b"dummy")
    for f in ["config.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.json"]:
        (tmp_path / f).write_bytes(b"{}")
    import hashlib
    sha256_map = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*") if p.is_file()}
    meta = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": "mit",
        "languages": list(LANGUAGES),
        "sha256": sha256_map,
    }
    (tmp_path / "artifact.json").write_text(json.dumps(meta))

    import transformers
    monkeypatch.setattr(transformers.M2M100Tokenizer, "from_pretrained", lambda *a, **kw: FakeTokenizer())
    monkeypatch.setattr(transformers.M2M100ForConditionalGeneration, "from_pretrained", lambda *a, **kw: FakeModel())

    engine = TranslationEngine(tmp_path)
    res = engine.translate_to_english("habari za asubuhi", language="sw")
    assert isinstance(res, TranslationResult)
    assert res.translated_windows == ["translated sentence"]
    assert res.input_truncated is False


def test_translation_engine_bidirectional(tmp_path, monkeypatch):
    import json
    from app.translation_artifact import MODEL_ID, MODEL_REVISION, LANGUAGES

    (tmp_path / "model.safetensors").write_bytes(b"dummy")
    for f in ["config.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.json"]:
        (tmp_path / f).write_bytes(b"{}")
    import hashlib
    sha256_map = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*") if p.is_file()}
    meta = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": "mit",
        "languages": list(LANGUAGES),
        "sha256": sha256_map,
    }
    (tmp_path / "artifact.json").write_text(json.dumps(meta))

    import transformers
    monkeypatch.setattr(transformers.M2M100Tokenizer, "from_pretrained", lambda *a, **kw: FakeTokenizer())
    monkeypatch.setattr(transformers.M2M100ForConditionalGeneration, "from_pretrained", lambda *a, **kw: FakeModel())

    engine = TranslationEngine(tmp_path)
    out = engine.translate("hello world", source_language="en", target_language="sw")
    assert out == "translated sentence"
