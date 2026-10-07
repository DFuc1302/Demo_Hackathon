from __future__ import annotations

import base64
import codecs
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from app.pipeline.language import identify_language

SECURITY_CATEGORIES = (
    "direct_instruction_override",
    "system_prompt_extraction",
    "persona_adoption",
    "counterfactual_simulation",
    "obfuscated_encoding",
    "data_poisoning_signal",
    "benign",
)
DEFAULT_RISK_THRESHOLDS = {"low": 0.20, "high": 0.80}

_OVERRIDE_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget|bypass|override)\b.*\b(instructions|prompt|rules|constraints|safety|policy|security)\b", re.I),
    re.compile(r"\bdo\s+not\s+follow\b.*\b(rules|instructions|policy|security)\b", re.I),
]

_EXTRACTION_PATTERNS = [
    re.compile(r"\b(reveal|output|display|print|show|repeat|leak)\b.*\b(system\s+prompt|hidden\s+prompt|internal\s+rules|developer\s+instructions)\b", re.I),
    re.compile(r"\bwhat\s+(are|is)\s+your\s+(original|initial|hidden|system)\s+(instructions|prompt|rules)\b", re.I),
]

_PERSONA_PATTERNS = [
    re.compile(r"\b(pretend|act\s+as|roleplay\s+as|simulate)\b.*\b(unrestricted|dan\s+mode|jailbroken|no\s+safety|evil\s+twin)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(unrestricted|free\s+of\s+rules|dan)\b", re.I),
]

_SIMULATION_PATTERNS = [
    re.compile(r"\bhypothetical(ly)?\b.*\b(scenario|world|case)\b.*\b(without\s+safety|no\s+rules|unfiltered)\b", re.I),
    re.compile(r"\bin\s+a\s+fictional\s+(story|game)\b.*\b(bypass|disregard\s+safety)\b", re.I),
]

_POISONING_PATTERNS = [
    re.compile(r"\b\[label\s*:\s*(benign|clean|safe)\]\b", re.I),
    re.compile(r"\btrigger_word_[a-z0-9]+\b", re.I),
    re.compile(r"\b(inject_backdoor|trojan_token)\b", re.I),
]

_BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")


@dataclass(frozen=True)
class SecurityAnalysisResult:
    prompt: str
    score: float
    predicted_category: str
    risk_level: Literal["low", "medium", "high"]
    heuristic_signals: list[str]
    language_estimate: dict[str, Any]
    component_confidence: float
    calibrated: bool
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_security_signals(text: str) -> tuple[list[str], str | None]:
    """Extract explainable heuristic signals from prompt text.
    
    Returns (signals, suggested_category).
    Signals are context indicators, NOT ground-truth model attribution.
    """
    signals: list[str] = []
    suggested_category: str | None = None

    # Obfuscation checks
    if _ZERO_WIDTH_PATTERN.search(text):
        signals.append("obfuscation:zero_width_characters")
        suggested_category = suggested_category or "obfuscated_encoding"

    # Base64 decode check
    for match in _BASE64_PATTERN.finditer(text):
        b64_candidate = match.group()
        if len(b64_candidate) % 4 != 0:
            b64_candidate += "=" * (4 - (len(b64_candidate) % 4))
        try:
            decoded = base64.b64decode(b64_candidate, validate=True).decode("utf-8", errors="ignore").strip()
            if len(decoded) >= 8 and any(c.isalpha() for c in decoded):
                signals.append(f"obfuscation:base64_payload({decoded[:30]}...)")
                suggested_category = suggested_category or "obfuscated_encoding"
                # Recursively check decoded payload for injection
                inner_signals, _ = extract_security_signals(decoded)
                signals.extend([f"nested:{s}" for s in inner_signals])
        except Exception:
            pass

    # Rot13 check
    try:
        rot13 = codecs.decode(text, "rot_13")
        if rot13 != text and any(w in rot13.lower() for w in ("ignore", "system", "prompt", "dan", "bypass")):
            signals.append("obfuscation:rot13_candidate")
            suggested_category = suggested_category or "obfuscated_encoding"
    except Exception:
        pass

    # Pattern categories
    for pat in _OVERRIDE_PATTERNS:
        if pat.search(text):
            signals.append("attack_vector:instruction_override")
            suggested_category = suggested_category or "direct_instruction_override"

    for pat in _EXTRACTION_PATTERNS:
        if pat.search(text):
            signals.append("attack_vector:system_prompt_extraction")
            suggested_category = suggested_category or "system_prompt_extraction"

    for pat in _PERSONA_PATTERNS:
        if pat.search(text):
            signals.append("attack_vector:persona_adoption")
            suggested_category = suggested_category or "persona_adoption"

    for pat in _SIMULATION_PATTERNS:
        if pat.search(text):
            signals.append("attack_vector:counterfactual_simulation")
            suggested_category = suggested_category or "counterfactual_simulation"

    for pat in _POISONING_PATTERNS:
        if pat.search(text):
            signals.append("data_quality:poisoning_marker")
            suggested_category = suggested_category or "data_poisoning_signal"

    return sorted(set(signals)), suggested_category


def calculate_risk_level(
    score: float, thresholds: dict[str, float] | None = None
) -> Literal["low", "medium", "high"]:
    thresh = thresholds or DEFAULT_RISK_THRESHOLDS
    low = float(thresh["low"])
    high = float(thresh["high"])
    if score < low:
        return "low"
    if score < high:
        return "medium"
    return "high"


def analyze_security_prompt(
    text: str,
    model_dir: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> SecurityAnalysisResult:
    """Analyze prompt for security risks, returning a structured verifiable report."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text cannot be empty")

    signals, suggested_category = extract_security_signals(text)
    lang_res = identify_language(text)

    score: float = 0.0
    calibrated: bool = False
    model_category: str | None = None
    provenance: dict[str, Any] = {
        "analysis_type": "security_pipeline_v1",
        "has_model_inference": False,
    }

    # Model inference integration if model_dir provided
    if model_dir is not None:
        from app.pipeline.core import load_artifact, infer
        from app.pipeline.data import Dataset
        artifact, config = load_artifact(model_dir)
        dummy_data = Dataset(columns=[config.text_column or "text"], rows=[{config.text_column or "text": text}])
        preds, probs = infer(dummy_data, config, artifact["preprocessing"], artifact["model"])
        
        if probs is not None:
            # Classification probability
            classes = artifact["model"]["classes"]
            pos_label = config.positive_label or (classes[1] if len(classes) == 2 else classes[0])
            if pos_label in classes:
                pos_idx = classes.index(pos_label)
                score = float(probs[0, pos_idx])
            else:
                score = float(max(probs[0]))
            model_category = str(preds[0])
        else:
            # Regression continuous score normalized via sigmoid
            raw_val = float(preds[0])
            import math
            score = 1.0 / (1.0 + math.exp(-raw_val))
            model_category = "continuous_risk"

        calibrated = False  # generic baseline is uncalibrated unless explicitly calibrated
        provenance.update({
            "has_model_inference": True,
            "model_kind": artifact["model"]["kind"],
            "run_id": artifact["run"]["run_id"],
        })
    else:
        # Heuristic-only baseline score estimation
        if signals:
            score = min(0.95, 0.40 + 0.20 * len(signals))
        else:
            score = 0.05
        calibrated = False
        provenance["scoring_mode"] = "heuristic_baseline"

    # Category resolution: model prediction if available, else suggested heuristic category, else benign
    final_category = model_category or suggested_category or "benign"
    risk = calculate_risk_level(score, thresholds)

    # Component confidence combines language certainty and signal count
    confidence = 0.90 if not lang_res.is_uncertain else 0.65
    if signals and not model_category:
        confidence = min(0.95, confidence + 0.05)

    return SecurityAnalysisResult(
        prompt=text,
        score=round(score, 4),
        predicted_category=final_category,
        risk_level=risk,
        heuristic_signals=signals,
        language_estimate={
            "language": lang_res.language,
            "confidence": lang_res.confidence,
            "script": lang_res.script,
            "is_uncertain": lang_res.is_uncertain,
            "code_switched": lang_res.code_switched,
        },
        component_confidence=round(confidence, 3),
        calibrated=calibrated,
        provenance=provenance,
    )
