from __future__ import annotations

import pytest

from app.inference import InferenceEngine, InferenceResult, MultilingualInferenceEngine
from app.multilingual_guardrail import (
    BENCHMARK_WARNING,
    UNCALIBRATED_TRANSLATION_WARNING,
    MultilingualGuardrailService,
)
from app.translation import TranslationEngine, TranslationResult


class FakeTokenizer:
    src_lang = "en"
    model_max_length = 256

    def __init__(self):
        self._langs = {"en": 100, "sw": 101, "ha": 102, "bn": 103}

    def get_lang_id(self, code: str) -> int:
        return self._langs[code]

    def __call__(self, text: str, truncation: bool = False, return_tensors: str | None = None, **kwargs):
        import torch
        tokens = [self._langs[self.src_lang], 10, 20, 2]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens])}
        return {"input_ids": tokens}
    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        return ["translated prompt"]


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, *args, **kwargs):
        import torch
        return torch.tensor([[100, 10, 20, 2]])


class FakeMultiEngine:
    def analyze(self, prompt: str, language: str) -> InferenceResult:
        is_attack = "attack" in prompt.lower()
        return InferenceResult(
            label="jailbreak" if is_attack else "benign",
            jailbreak_probability=0.95 if is_attack else 0.05,
            risk_level="high" if is_attack else "low",
            heuristic_signals=[],
            input_truncated=False,
            model_version="test-multi",
            calibrated=True,
        )


class FakeEnglishEngine:
    def analyze(self, prompt: str) -> InferenceResult:
        is_attack = "attack" in prompt.lower() or "translated" in prompt.lower()
        return InferenceResult(
            label="jailbreak" if is_attack else "benign",
            jailbreak_probability=0.85 if is_attack else 0.10,
            risk_level="high" if is_attack else "low",
            heuristic_signals=[],
            input_truncated=False,
            model_version="test-en",
            calibrated=True,
        )


@pytest.fixture
def fake_guardrail_service(tmp_path, monkeypatch):
    import json
    from app.translation_artifact import MODEL_ID, MODEL_REVISION, LANGUAGES
    import hashlib

    # Dummy translator artifact
    (tmp_path / "model.safetensors").write_bytes(b"dummy")
    for f in ["config.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.json"]:
        (tmp_path / f).write_bytes(b"{}")
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

    trans_engine = TranslationEngine(tmp_path)
    service = MultilingualGuardrailService(
        multilingual_engine=FakeMultiEngine(),  # type: ignore
        translation_engine=trans_engine,
        english_engine=FakeEnglishEngine(),    # type: ignore
    )
    return service


def test_guardrail_multilingual_mode(fake_guardrail_service):
    res = fake_guardrail_service.analyze("safe prompt", language="sw", mode="multilingual")
    assert res.mode == "multilingual"
    assert res.decision == "benign"
    assert res.multilingual_assessment is not None
    assert res.translation_assessment is None
    assert BENCHMARK_WARNING in res.warnings
    assert UNCALIBRATED_TRANSLATION_WARNING not in res.warnings


def test_guardrail_translation_mode_uncalibrated(fake_guardrail_service):
    res = fake_guardrail_service.analyze("safe prompt", language="ha", mode="translation")
    assert res.mode == "translation"
    assert res.multilingual_assessment is None
    assert res.translation_assessment is not None
    assert res.translation_assessment.analysis.calibrated is False
    assert UNCALIBRATED_TRANSLATION_WARNING in res.warnings


def test_guardrail_compare_mode_conservative_or(fake_guardrail_service):
    # Prompt is safe in source (FakeMulti -> benign), but translated triggers attack in FakeEnglish -> jailbreak
    res = fake_guardrail_service.analyze("benign prompt", language="bn", mode="compare")
    assert res.mode == "compare"
    assert res.multilingual_assessment.label == "benign"
    assert res.translation_assessment.analysis.label == "jailbreak"
    assert res.decision == "jailbreak"
    assert res.risk_level == "high"
    assert BENCHMARK_WARNING in res.warnings
    assert UNCALIBRATED_TRANSLATION_WARNING in res.warnings
