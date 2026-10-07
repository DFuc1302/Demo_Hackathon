from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from app.inference import normalize_input_text
from app.translation_artifact import (
    LANGUAGES,
    validate_language_codes,
    validate_translation_artifact,
)


@dataclass(frozen=True)
class TranslationResult:
    translated_windows: list[str]
    input_truncated: bool


class TranslationEngine:
    _instance: TranslationEngine | None = None
    _init_lock = threading.Lock()

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        validate_translation_artifact(self.model_path)
        self._lock = threading.Lock()

        # Load on GPU if available, else CPU (per user instruction: if any tasks can be done faster by gpu, do it)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = M2M100Tokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        validate_language_codes(self.tokenizer)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        self.max_tokens = getattr(self.tokenizer, "model_max_length", 1024)
        if self.max_tokens > 10000:  # handle unbounded default
            self.max_tokens = 1024

    def translate_to_english(
        self,
        text: str,
        language: Literal["sw", "ha", "bn"],
    ) -> TranslationResult:
        if language not in ("sw", "ha", "bn"):
            raise ValueError(f"unsupported source language: {language}")
        normalized = normalize_input_text(text)
        if not normalized:
            return TranslationResult(translated_windows=[""], input_truncated=False)

        with self._lock:
            self.tokenizer.src_lang = language
            encoded = self.tokenizer(normalized, truncation=False, return_tensors="pt")
            input_ids = encoded["input_ids"][0]
            target_bos = self.tokenizer.get_lang_id("en")

            input_ids = input_ids.to(self.device)
            if len(input_ids) <= self.max_tokens:
                # fits within context
                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids.unsqueeze(0),
                        forced_bos_token_id=target_bos,
                        do_sample=False,
                        num_beams=4,
                        max_new_tokens=256,
                    )
                translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                return TranslationResult(translated_windows=[translated], input_truncated=False)

            # head/tail windowing
            head_ids = input_ids[: self.max_tokens].unsqueeze(0).to(self.device)
            tail_ids = input_ids[-self.max_tokens :].unsqueeze(0).to(self.device)
            with torch.inference_mode():
                head_out = self.model.generate(
                    head_ids,
                    forced_bos_token_id=target_bos,
                    do_sample=False,
                    num_beams=4,
                    max_new_tokens=256,
                )
                tail_out = self.model.generate(
                    tail_ids,
                    forced_bos_token_id=target_bos,
                    do_sample=False,
                    num_beams=4,
                    max_new_tokens=256,
                )
            head_translated = self.tokenizer.batch_decode(head_out, skip_special_tokens=True)[0].strip()
            tail_translated = self.tokenizer.batch_decode(tail_out, skip_special_tokens=True)[0].strip()
            return TranslationResult(
                translated_windows=[head_translated, tail_translated],
                input_truncated=True,
            )

    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        if source_language not in LANGUAGES or target_language not in LANGUAGES:
            raise ValueError("unsupported translation language")
        normalized = normalize_input_text(text)
        if not normalized:
            return ""

        with self._lock:
            self.tokenizer.src_lang = source_language
            target_bos = self.tokenizer.get_lang_id(target_language)
            encoded = self.tokenizer(normalized, truncation=False, return_tensors="pt")
            input_ids = encoded["input_ids"][0]

            input_ids = input_ids.to(self.device)
            if len(input_ids) <= self.max_tokens:
                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids.unsqueeze(0),
                        forced_bos_token_id=target_bos,
                        do_sample=False,
                        num_beams=4,
                        max_new_tokens=256,
                    )
                return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            # head/tail windowing
            head_ids = input_ids[: self.max_tokens].unsqueeze(0).to(self.device)
            tail_ids = input_ids[-self.max_tokens :].unsqueeze(0).to(self.device)
            with torch.inference_mode():
                head_out = self.model.generate(
                    head_ids,
                    forced_bos_token_id=target_bos,
                    do_sample=False,
                    num_beams=4,
                    max_new_tokens=256,
                )
                tail_out = self.model.generate(
                    tail_ids,
                    forced_bos_token_id=target_bos,
                    do_sample=False,
                    num_beams=4,
                    max_new_tokens=256,
                )
            head_tr = self.tokenizer.batch_decode(head_out, skip_special_tokens=True)[0].strip()
            tail_tr = self.tokenizer.batch_decode(tail_out, skip_special_tokens=True)[0].strip()
            if head_tr == tail_tr:
                return head_tr
            return f"{head_tr} {tail_tr}".strip()
    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        batch_size: int = 8,
    ) -> list[str]:
        if source_language not in LANGUAGES or target_language not in LANGUAGES:
            raise ValueError("unsupported translation language")
        if not texts:
            return []

        results = []
        with self._lock:
            self.tokenizer.src_lang = source_language
            target_bos = self.tokenizer.get_lang_id(target_language)

            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                normalized_chunk = [normalize_input_text(t) for t in chunk]
                valid_indices = [idx for idx, t in enumerate(normalized_chunk) if t]
                if not valid_indices:
                    results.extend([""] * len(chunk))
                    continue

                valid_texts = [normalized_chunk[idx] for idx in valid_indices]
                encoded_list = [
                    self.tokenizer(t, truncation=False)["input_ids"]
                    for t in valid_texts
                ]

                standard_indices = [idx for idx, ids in enumerate(encoded_list) if len(ids) <= self.max_tokens]
                long_indices = [idx for idx, ids in enumerate(encoded_list) if len(ids) > self.max_tokens]

                translated_valid: list[str] = [""] * len(valid_texts)

                if standard_indices:
                    std_texts = [valid_texts[idx] for idx in standard_indices]
                    std_inputs = self.tokenizer(
                        std_texts,
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.inference_mode():
                        std_out = self.model.generate(
                            **std_inputs,
                            forced_bos_token_id=target_bos,
                            do_sample=False,
                            num_beams=4,
                            max_new_tokens=256,
                        )
                    decoded_std = self.tokenizer.batch_decode(std_out, skip_special_tokens=True)
                    for std_idx, text_out in zip(standard_indices, decoded_std):
                        translated_valid[std_idx] = text_out.strip()

                for long_idx in long_indices:
                    ids = encoded_list[long_idx]
                    head_ids = torch.tensor([ids[: self.max_tokens]], device=self.device)
                    tail_ids = torch.tensor([ids[-self.max_tokens :]], device=self.device)
                    with torch.inference_mode():
                        head_out = self.model.generate(
                            head_ids,
                            forced_bos_token_id=target_bos,
                            do_sample=False,
                            num_beams=4,
                            max_new_tokens=256,
                        )
                        tail_out = self.model.generate(
                            tail_ids,
                            forced_bos_token_id=target_bos,
                            do_sample=False,
                            num_beams=4,
                            max_new_tokens=256,
                        )
                    head_tr = self.tokenizer.batch_decode(head_out, skip_special_tokens=True)[0].strip()
                    tail_tr = self.tokenizer.batch_decode(tail_out, skip_special_tokens=True)[0].strip()
                    translated_valid[long_idx] = head_tr if head_tr == tail_tr else f"{head_tr} {tail_tr}".strip()

                chunk_res = [""] * len(chunk)
                for val_idx, text_out in zip(valid_indices, translated_valid):
                    chunk_res[val_idx] = text_out
                results.extend(chunk_res)

        return results
