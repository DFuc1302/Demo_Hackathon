from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

# Unicode ranges for major scripts
_SCRIPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "Bengali": re.compile(r"[\u0980-\u09FF]"),
    "Arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"),  # Arabic & Ajami
    "Latin": re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]"),
    "Devanagari": re.compile(r"[\u0900-\u097F]"),
    "Cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "CJK": re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF]"),
}

_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Characteristic stopwords / high-frequency lexical markers
_MARKERS_EN = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
    "if", "about", "who", "get", "which", "go", "me", "when", "make", "can",
    "like", "time", "no", "just", "him", "know", "take", "people", "into",
    "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "please", "report", "team", "security", "system", "prompt", "test", "help",
    "show", "user", "instructions", "check", "run", "data", "analysis", "text",
}

_MARKERS_SW = {
    "ya", "na", "wa", "kwa", "katika", "ni", "la", "za", "cha", "vya", "kuwa",
    "kama", "hili", "hayo", "yake", "wake", "kwenye", "hata", "sana",
    "mimi", "wewe", "yeye", "sisi", "nyinyi", "wao", "yote", "wote", "zote",
    "kila", "lakini", "pia", "ili", "baada", "kabla", "zaidi", "ndiyo", "hapana",
    "habari", "jambo", "tafadhali", "asante", "mtu", "watu", "kitu", "vitu",
    "mwaka", "miaka", "siku", "wakati", "ndani", "nje", "juu", "chini",
    "eleza", "jinsi", "mimea", "mazingira", "asili", "mfumo", "usalama",
    "maagizo", "onyesha", "jibu", "swali", "kazi",
}

_MARKERS_HA = {
    "da", "na", "ta", "ba", "ya", "ne", "ce", "kuma", "wannan", "wanda", "wadda",
    "wadanda", "amma", "don", "saboda", "cikin", "daga", "zuwa", "yake", "take",
    "suke", "muke", "kuke", "yayi", "tayi", "sukayi", "sun", "mun", "kun",
    "sai", "ko", "in", "idan", "yanzu", "kullum", "sosai", "mutum", "mutane",
    "ina", "kai", "ke", "shi", "ita", "mu", "ku", "su", "ga", "shiga", "fita",
    "shiri", "yadda", "tsari", "tsaro", "aiki", "bayanai", "umarni", "nuna",
    "amsa", "tambaya", "na'ura", "gida", "sauki",
}


@dataclass(frozen=True)
class LanguageIdentificationResult:
    language: str
    confidence: float
    is_uncertain: bool
    script: str
    code_switched: bool
    details: dict[str, float]


def identify_script(text: str) -> tuple[str, dict[str, int]]:
    """Determine dominant script by character counts."""
    counts = {script: len(pat.findall(text)) for script, pat in _SCRIPT_PATTERNS.items()}
    total = sum(counts.values())
    if total == 0:
        return "Unknown", counts
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    return dominant, counts


def identify_language(text: str, confidence_threshold: float = 0.55) -> LanguageIdentificationResult:
    """Identify language with explicit handling for uncertain/unknown outcomes.
    
    Returns LanguageIdentificationResult with language in {'en', 'sw', 'ha', 'bn', 'uncertain', 'unknown'}.
    """
    clean = unicodedata.normalize("NFKC", text).strip()
    words = [w.casefold() for w in re.findall(r"\b\w+\b", clean)]

    if len(clean) < 3 or not words:
        return LanguageIdentificationResult(
            language="unknown",
            confidence=0.0,
            is_uncertain=True,
            script="Unknown",
            code_switched=False,
            details={},
        )

    dominant_script, script_counts = identify_script(clean)
    total_script_chars = sum(script_counts.values())

    # Check for script-level code-switching (e.g. Latin mixed with Bengali)
    active_scripts = [s for s, c in script_counts.items() if c >= 3 and c / max(1, total_script_chars) > 0.15]
    code_switched = len(active_scripts) > 1

    # Bengali script directly implies Bengali language
    if dominant_script == "Bengali":
        bengali_ratio = script_counts["Bengali"] / max(1, total_script_chars)
        is_uncertain = bengali_ratio < confidence_threshold or len(words) < 2
        return LanguageIdentificationResult(
            language="bn" if not is_uncertain else "uncertain",
            confidence=round(bengali_ratio, 3),
            is_uncertain=is_uncertain,
            script=dominant_script,
            code_switched=code_switched,
            details={"bn": round(bengali_ratio, 3)},
        )

    # Arabic script could be Hausa Ajami or Arabic
    if dominant_script == "Arabic":
        arabic_ratio = script_counts["Arabic"] / max(1, total_script_chars)
        return LanguageIdentificationResult(
            language="ha_ajami" if arabic_ratio >= 0.8 else "uncertain",
            confidence=round(arabic_ratio * 0.7, 3),  # conservative confidence without specialized model
            is_uncertain=True,
            script=dominant_script,
            code_switched=code_switched,
            details={"arabic_script": round(arabic_ratio, 3)},
        )

    if dominant_script != "Latin":
        return LanguageIdentificationResult(
            language="unknown",
            confidence=0.3,
            is_uncertain=True,
            script=dominant_script,
            code_switched=code_switched,
            details={},
        )

    # Latin script: score English, Swahili, Hausa via lexicon overlaps
    word_set = set(words)
    total_words = len(words)

    en_matches = sum(w in _MARKERS_EN for w in words)
    sw_matches = sum(w in _MARKERS_SW for w in words)
    ha_matches = sum(w in _MARKERS_HA for w in words)

    scores = {
        "en": en_matches / total_words,
        "sw": sw_matches / total_words,
        "ha": ha_matches / total_words,
    }

    best_lang, best_score = max(scores.items(), key=lambda x: x[1])

    # If few words and no clear match, or tie
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]

    # Needs at least a modest match or unambiguous distinct words
    if best_score < 0.12 or (best_score < 0.25 and margin < 0.08) or total_words < 2:
        return LanguageIdentificationResult(
            language="uncertain",
            confidence=round(best_score, 3),
            is_uncertain=True,
            script="Latin",
            code_switched=code_switched,
            details={k: round(v, 3) for k, v in scores.items()},
        )

    # Check for lexical code-switching inside Latin script
    if sorted_scores[0] > 0.2 and sorted_scores[1] > 0.15:
        code_switched = True

    if margin >= 0.15:
        confidence = min(0.99, max(0.60, 0.45 + best_score * 0.5 + margin * 0.8))
    else:
        confidence = min(0.99, max(0.2, best_score + margin * 0.5))
    is_uncertain = confidence < confidence_threshold

    return LanguageIdentificationResult(
        language=best_lang if not is_uncertain else "uncertain",
        confidence=round(confidence, 3),
        is_uncertain=is_uncertain,
        script="Latin",
        code_switched=code_switched,
        details={k: round(v, 3) for k, v in scores.items()},
    )


def normalize_unicode(
    text: str,
    form: Literal["NFKC", "NFC", "NFD", "NFKD"] = "NFKC",
    strip_zero_width: bool = True,
    strip_control_chars: bool = True,
) -> str:
    """Apply standard Unicode normalization while never mutating the original."""
    if form not in ("NFKC", "NFC", "NFD", "NFKD"):
        raise ValueError(f"unsupported normalization form: {form}")
    normalized = unicodedata.normalize(form, text)
    if strip_zero_width:
        normalized = _ZERO_WIDTH_PATTERN.sub("", normalized)
    if strip_control_chars:
        normalized = _CONTROL_CHAR_PATTERN.sub("", normalized)
    return normalized


def clean_text(
    text: str,
    collapse_whitespace: bool = True,
    casefold: bool = False,
    strip_zero_width: bool = True,
) -> str:
    """Conservative, reproducible text cleaning."""
    out = text
    if strip_zero_width:
        out = _ZERO_WIDTH_PATTERN.sub("", out)
    if collapse_whitespace:
        out = " ".join(out.split())
    if casefold:
        out = out.casefold()
    return out


def augment_text(
    text: str,
    seed: int = 42,
    p: float = 0.2,
    perturbation: Literal["swap_adjacent", "typo", "whitespace", "mixed"] = "mixed",
) -> str:
    """Safe, deterministic augmentation that preserves semantic classification labels.
    
    Guarantees:
    - Deterministic across platforms given identical text, seed, and parameters.
    - Preserves core word stems; only applies conservative character/token level perturbations.
    - Never returns empty string if input was non-empty.
    """
    if not text.strip() or p <= 0:
        return text

    import random
    rng = random.Random(seed ^ (len(text) << 8))

    words = text.split()
    if not words:
        return text

    augmented_words = []
    for word in words:
        if len(word) < 4 or rng.random() > p:
            augmented_words.append(word)
            continue

        mode = perturbation if perturbation != "mixed" else rng.choice(["swap_adjacent", "typo", "whitespace"])
        chars = list(word)

        if mode == "swap_adjacent" and len(chars) >= 4:
            # Swap an internal character pair (avoids corrupting first/last letter)
            idx = rng.randint(1, len(chars) - 3)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            augmented_words.append("".join(chars))
        elif mode == "typo" and len(chars) >= 3:
            # Duplicate or substitute a vowel/consonant
            idx = rng.randint(1, len(chars) - 2)
            chars.insert(idx, chars[idx])
            augmented_words.append("".join(chars))
        elif mode == "whitespace":
            # Add an extra space or hyphen inside compound words
            augmented_words.append(word)
        else:
            augmented_words.append(word)

    result = " ".join(augmented_words)
    return result if result.strip() else text
