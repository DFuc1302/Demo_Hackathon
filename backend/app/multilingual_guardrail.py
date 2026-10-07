from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.inference import InferenceEngine, InferenceResult, MultilingualInferenceEngine
from app.translation import TranslationEngine, TranslationResult

BENCHMARK_WARNING = (
    "Metrics use machine-translated prompts and do not establish native-speaker performance."
)
UNCALIBRATED_TRANSLATION_WARNING = (
    "Translation-assisted scores are not calibrated for the source language."
)

RISK_RANKS = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class TranslatedAssessmentResult:
    analysis: InferenceResult
    translation_model: str
    translation_input_truncated: bool


@dataclass(frozen=True)
class MultilingualGuardrailResult:
    language: str
    mode: str
    decision: str
    risk_level: str
    warnings: list[str]
    multilingual_assessment: InferenceResult | None
    translation_assessment: TranslatedAssessmentResult | None


class MultilingualGuardrailService:
    def __init__(
        self,
        *,
        multilingual_engine: MultilingualInferenceEngine,
        translation_engine: TranslationEngine,
        english_engine: InferenceEngine,
    ):
        self.multilingual_engine = multilingual_engine
        self.translation_engine = translation_engine
        self.english_engine = english_engine

    def analyze(
        self,
        prompt: str,
        language: Literal["sw", "ha", "bn"],
        mode: Literal["multilingual", "translation", "compare"] = "compare",
    ) -> MultilingualGuardrailResult:
        if language not in ("sw", "ha", "bn"):
            raise ValueError(f"unsupported language: {language}")
        if mode not in ("multilingual", "translation", "compare"):
            raise ValueError(f"unsupported mode: {mode}")

        warnings = [BENCHMARK_WARNING]
        multi_assessment: InferenceResult | None = None
        trans_assessment: TranslatedAssessmentResult | None = None

        if mode in ("multilingual", "compare"):
            multi_assessment = self.multilingual_engine.analyze(prompt, language=language)

        if mode in ("translation", "compare"):
            warnings.append(UNCALIBRATED_TRANSLATION_WARNING)
            t_res: TranslationResult = self.translation_engine.translate_to_english(
                prompt, language=language
            )

            worst_analysis: InferenceResult | None = None
            for window in t_res.translated_windows:
                if not window:
                    continue
                analysis = self.english_engine.analyze(window)
                # Uncalibrated for translation mode
                uncalibrated_analysis = InferenceResult(
                    label=analysis.label,
                    jailbreak_probability=analysis.jailbreak_probability,
                    risk_level=analysis.risk_level,
                    heuristic_signals=analysis.heuristic_signals,
                    input_truncated=analysis.input_truncated or t_res.input_truncated,
                    model_version=analysis.model_version,
                    calibrated=False,
                )
                if worst_analysis is None or uncalibrated_analysis.jailbreak_probability > worst_analysis.jailbreak_probability:
                    worst_analysis = uncalibrated_analysis

            if worst_analysis is None:
                worst_analysis = self.english_engine.analyze("empty")
                worst_analysis = InferenceResult(
                    label=worst_analysis.label,
                    jailbreak_probability=worst_analysis.jailbreak_probability,
                    risk_level=worst_analysis.risk_level,
                    heuristic_signals=worst_analysis.heuristic_signals,
                    input_truncated=t_res.input_truncated,
                    model_version=worst_analysis.model_version,
                    calibrated=False,
                )

            trans_assessment = TranslatedAssessmentResult(
                analysis=worst_analysis,
                translation_model="facebook/m2m100_418M",
                translation_input_truncated=t_res.input_truncated,
            )

        # Decide top-level decision and risk
        if mode == "multilingual":
            assert multi_assessment is not None
            decision = multi_assessment.label
            r_level = multi_assessment.risk_level
        elif mode == "translation":
            assert trans_assessment is not None
            decision = trans_assessment.analysis.label
            r_level = trans_assessment.analysis.risk_level
        else:  # compare
            assert multi_assessment is not None and trans_assessment is not None
            decision = (
                "jailbreak"
                if multi_assessment.label == "jailbreak" or trans_assessment.analysis.label == "jailbreak"
                else "benign"
            )
            # max risk level: low < medium < high
            rank1 = RISK_RANKS.get(multi_assessment.risk_level, 1)
            rank2 = RISK_RANKS.get(trans_assessment.analysis.risk_level, 1)
            r_level = multi_assessment.risk_level if rank1 >= rank2 else trans_assessment.analysis.risk_level

        return MultilingualGuardrailResult(
            language=language,
            mode=mode,
            decision=decision,
            risk_level=r_level,
            warnings=warnings,
            multilingual_assessment=multi_assessment,
            translation_assessment=trans_assessment,
        )
