from __future__ import annotations

import base64
import codecs
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.model_metadata import validate_model_metadata
from app.signals import extract_signals
MAX_PROMPT_LENGTH = 5000
MAX_TOKEN_LENGTH = 256
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")
_BASE64_CANDIDATE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def normalize_input_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_PATTERN.sub("", normalized)
    return normalized.strip()


def extract_decoded_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _BASE64_CANDIDATE.finditer(text):
        s = match.group()
        if len(s) % 4 != 0:
            s += "=" * (4 - (len(s) % 4))
        try:
            decoded_bytes = base64.b64decode(s, validate=True)
            decoded = decoded_bytes.decode("utf-8", errors="ignore").strip()
            if len(decoded) >= 8 and any(c.isalpha() for c in decoded):
                candidates.append(decoded)
        except Exception:
            pass
    # Check rot13 if contains word-like tokens
    try:
        rot13_candidate = codecs.decode(text, "rot_13")
        if rot13_candidate != text and any(w in rot13_candidate.lower() for w in ("ignore", "bypass", "system", "prompt", "instructions", "dan", "unrestricted")):
            candidates.append(rot13_candidate)
    except Exception:
        pass
    return candidates


@dataclass(frozen=True)
class InferenceResult:
    label: str
    jailbreak_probability: float
    risk_level: str
    heuristic_signals: list[str]
    input_truncated: bool
    model_version: str
    calibrated: bool


def risk_level(jailbreak_probability: float, thresholds: dict[str, float]) -> str:
    if not 0.0 <= jailbreak_probability <= 1.0:
        raise ValueError("jailbreak probability must be between 0 and 1")
    low = float(thresholds["low"])
    high = float(thresholds["high"])
    if not math.isfinite(low) or not math.isfinite(high) or not 0.0 < low < high < 1.0:
        raise ValueError("risk thresholds must satisfy 0 < low < high < 1")
    if jailbreak_probability < low:
        return "low"
    if jailbreak_probability < high:
        return "medium"
    return "high"


class InferenceEngine:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"model directory not found: {self.model_path}")
        metadata_path = self.model_path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"model directory is missing metadata.json: {self.model_path}")
        try:
            self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read model metadata in {self.model_path}: {exc}") from exc
        self.metadata = validate_model_metadata(self.metadata, self.model_path)
        self._temperature = float(self.metadata["calibration_temperature"])
        self._classification_threshold = float(self.metadata["classification_threshold"])
        self._risk_thresholds = {
            "low": float(self.metadata["risk_thresholds"]["low"]),
            "high": float(self.metadata["risk_thresholds"]["high"]),
        }
        try:
            # Preserve DeBERTa's trained pre-tokenizer; the Mistral patch replaces it.
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True, fix_mistral_regex=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(f"could not load transformer model from {self.model_path}: {exc}") from exc
        config_labels = {str(key): value for key, value in self._model.config.id2label.items()}
        if config_labels.get("0") != "benign" or config_labels.get("1") != "jailbreak" or self._model.config.num_labels != 2:
            raise ValueError(f"model configuration has an incompatible label mapping: {self.model_path}")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        if self._device.type == "cuda":
            self._model.half()
        self._model.eval()
    def _score_single_text(self, text: str) -> tuple[float, bool]:
        return _score_text_windows(
            self._tokenizer,
            self._model,
            self._device,
            self._temperature,
            text,
        )
    def analyze(self, prompt: str) -> InferenceResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        normalized_prompt = normalize_input_text(prompt)
        if not normalized_prompt:
            raise ValueError("prompt cannot be empty")
        if len(normalized_prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")

        best_prob, input_truncated = self._score_single_text(normalized_prompt)
        heuristic_signals = set(extract_signals(normalized_prompt))

        # Dual-pass on decoded candidates (Base64 / Rot13)
        candidates = extract_decoded_candidates(normalized_prompt)
        for cand in candidates:
            cand_prob, _ = self._score_single_text(cand)
            if cand_prob > best_prob:
                best_prob = cand_prob
            heuristic_signals.update(extract_signals(cand))

        label = "jailbreak" if best_prob >= self._classification_threshold else "benign"
        return InferenceResult(
            label,
            best_prob,
            risk_level(best_prob, self._risk_thresholds),
            sorted(heuristic_signals),
            input_truncated,
            self.metadata["model_version"],
            True,
        )

    def analyze_batch(self, prompts: list[str]) -> list[InferenceResult]:
        if not isinstance(prompts, list):
            raise TypeError("prompts must be a list of strings")
        if not prompts:
            return []
        # Pre-process all prompts
        normalized_prompts: list[str] = []
        heuristic_signals_list: list[set[str]] = []
        candidates_list: list[list[str]] = []
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a string")
            norm = normalize_input_text(prompt)
            if not norm:
                raise ValueError("prompt cannot be empty")
            if len(norm) > MAX_PROMPT_LENGTH:
                raise ValueError(f"prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")
            normalized_prompts.append(norm)
            signals = set(extract_signals(norm))
            cands = extract_decoded_candidates(norm)
            for c in cands:
                signals.update(extract_signals(c))
            heuristic_signals_list.append(signals)
            candidates_list.append(cands)

        # Build list of all sequences to evaluate in a single batched tensor forward pass
        # Track mapping from prompt index to sequence indices
        prompt_seq_indices: list[list[int]] = [[] for _ in prompts]
        all_token_seqs: list[list[int]] = []
        input_truncated_flags: list[bool] = []

        for idx, norm in enumerate(normalized_prompts):
            full_tokens = self._tokenizer(norm, truncation=False, add_special_tokens=True)["input_ids"]
            if len(full_tokens) <= MAX_TOKEN_LENGTH:
                input_truncated_flags.append(False)
                prompt_seq_indices[idx].append(len(all_token_seqs))
                all_token_seqs.append(full_tokens)
            else:
                input_truncated_flags.append(True)
                prompt_seq_indices[idx].append(len(all_token_seqs))
                all_token_seqs.append(full_tokens[:MAX_TOKEN_LENGTH])
                prompt_seq_indices[idx].append(len(all_token_seqs))
                all_token_seqs.append(full_tokens[-MAX_TOKEN_LENGTH:])

            for cand in candidates_list[idx]:
                cand_tokens = self._tokenizer(cand, truncation=False, add_special_tokens=True)["input_ids"]
                if len(cand_tokens) <= MAX_TOKEN_LENGTH:
                    prompt_seq_indices[idx].append(len(all_token_seqs))
                    all_token_seqs.append(cand_tokens)
                else:
                    prompt_seq_indices[idx].append(len(all_token_seqs))
                    all_token_seqs.append(cand_tokens[:MAX_TOKEN_LENGTH])
                    prompt_seq_indices[idx].append(len(all_token_seqs))
                    all_token_seqs.append(cand_tokens[-MAX_TOKEN_LENGTH:])

        # Batch pad all sequences and execute single forward pass
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        max_len = max(len(s) for s in all_token_seqs)
        padded_ids = []
        attention_masks = []
        for s in all_token_seqs:
            padded = s + [pad_token_id] * (max_len - len(s))
            mask = [1] * len(s) + [0] * (max_len - len(s))
            padded_ids.append(padded)
            attention_masks.append(mask)

        batch_tensor = torch.tensor(padded_ids, dtype=torch.long, device=self._device)
        mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=self._device)

        with torch.inference_mode():
            logits = self._model(input_ids=batch_tensor, attention_mask=mask_tensor).logits
            probabilities = torch.softmax(logits / self._temperature, dim=-1)
            probs_jailbreak = probabilities[:, 1].tolist()

        results: list[InferenceResult] = []
        for idx in range(len(prompts)):
            seq_idxs = prompt_seq_indices[idx]
            best_prob = max(probs_jailbreak[i] for i in seq_idxs)
            label = "jailbreak" if best_prob >= self._classification_threshold else "benign"
            results.append(
                InferenceResult(
                    label,
                    best_prob,
                    risk_level(best_prob, self._risk_thresholds),
                    sorted(heuristic_signals_list[idx]),
                    input_truncated_flags[idx],
                    self.metadata["model_version"],
                    True,
                )
            )
        return results
def _score_text_windows(
    tokenizer: Any,
    model: Any,
    device: torch.device,
    temperature: float,
    text: str,
) -> tuple[float, bool]:
    full_tokens = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    if len(full_tokens) <= MAX_TOKEN_LENGTH:
        input_truncated = False
        encoded = tokenizer(text, max_length=MAX_TOKEN_LENGTH, truncation=True, return_tensors="pt", padding=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits / temperature, dim=-1)[0]
        return float(probabilities[1].item()), input_truncated

    input_truncated = True
    head_ids = full_tokens[:MAX_TOKEN_LENGTH]
    tail_ids = full_tokens[-MAX_TOKEN_LENGTH:]
    batch_ids = torch.tensor([head_ids, tail_ids], dtype=torch.long, device=device)
    batch_mask = torch.ones_like(batch_ids)
    with torch.inference_mode():
        logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
        probabilities = torch.softmax(logits / temperature, dim=-1)
        probs_jailbreak = probabilities[:, 1]
        max_prob = float(torch.max(probs_jailbreak).item())
    return max_prob, input_truncated


class MultilingualInferenceEngine:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        metadata_file = self.model_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"model metadata file not found: {metadata_file}")
        try:
            raw_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read model metadata: {metadata_file}: {exc}") from exc
        self.metadata = validate_model_metadata(raw_metadata, self.model_path)
        self._temperatures = self.metadata["calibration_temperatures"]
        self._classification_threshold = float(self.metadata["classification_threshold"])
        self._risk_thresholds = {k: float(v) for k, v in self.metadata["risk_thresholds"].items()}
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(f"could not load multilingual model from {self.model_path}: {exc}") from exc
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        if self._device.type == "cuda":
            self._model.half()
        self._model.eval()

    def analyze(self, prompt: str, language: Literal["sw", "ha", "bn"]) -> InferenceResult:
        if language not in ("sw", "ha", "bn"):
            raise ValueError(f"unsupported language: {language}")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        norm = normalize_input_text(prompt)
        if not norm:
            raise ValueError("prompt cannot be empty")
        if len(norm) > MAX_PROMPT_LENGTH:
            raise ValueError(f"prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")

        temp = float(self._temperatures[language])
        prob, truncated = _score_text_windows(self._tokenizer, self._model, self._device, temp, norm)
        signals = sorted(extract_signals(norm))
        label = "jailbreak" if prob >= self._classification_threshold else "benign"
        return InferenceResult(
            label=label,
            jailbreak_probability=prob,
            risk_level=risk_level(prob, self._risk_thresholds),
            heuristic_signals=signals,
            input_truncated=truncated,
            model_version=self.metadata["model_version"],
            calibrated=True,
        )
