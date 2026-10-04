from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from app.signals import extract_signals

MAX_PROMPT_LENGTH = 5000
LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.70


@dataclass(frozen=True)
class InferenceResult:
    label: str
    jailbreak_probability: float
    risk_level: str
    detected_signals: list[str]


def risk_level(jailbreak_probability: float) -> str:
    if not 0.0 <= jailbreak_probability <= 1.0:
        raise ValueError("jailbreak probability must be between 0 and 1")
    if jailbreak_probability < LOW_RISK_THRESHOLD:
        return "low"
    if jailbreak_probability < HIGH_RISK_THRESHOLD:
        return "medium"
    return "high"


class InferenceEngine:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"model artifact not found: {self.model_path}")
        try:
            artifact: Any = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"could not load model artifact: {self.model_path}") from exc

        pipeline = artifact.get("pipeline") if isinstance(artifact, dict) else None
        if pipeline is None or not hasattr(pipeline, "predict_proba"):
            raise ValueError(f"model artifact has no compatible pipeline: {self.model_path}")
        self._pipeline = pipeline

    def analyze(self, prompt: str) -> InferenceResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("prompt cannot be empty")
        if len(normalized_prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")

        probabilities = self._pipeline.predict_proba([normalized_prompt])[0]
        classes = list(self._pipeline.classes_)
        jailbreak_probability = float(probabilities[classes.index("jailbreak")])
        label = str(self._pipeline.predict([normalized_prompt])[0])
        return InferenceResult(
            label=label,
            jailbreak_probability=jailbreak_probability,
            risk_level=risk_level(jailbreak_probability),
            detected_signals=extract_signals(normalized_prompt),
        )
