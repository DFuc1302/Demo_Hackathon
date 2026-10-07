# M2 Multilingual & Low-Resource Text Preparation Guide

## Overview

The M2 module (`backend/app/pipeline/language.py`) provides reusable, task-independent text preprocessing, script identification, language detection with explicit uncertainty bounds, conservative cleanup, subword representations, and semantic label-preserving augmentation.

## Core Capabilities

### 1. Language & Script Identification

Function: `identify_language(text: str, confidence_threshold: float = 0.55) -> LanguageIdentificationResult`

- **Script Detection**: Multi-script character counting covering Latin, Bengali (`\u0980-\u09FF`), Arabic/Ajami (`\u0600-\u06FF`), Devanagari, Cyrillic, and CJK.
- **Lexical Profiling**: Statistical overlap against validated lexical markers for English (`en`), Swahili (`sw`), and Hausa (`ha`).
- **Uncertainty & Unknown Handling**: Short texts ($<3$ characters or $<2$ tokens) or ambiguous lexicon scores return `is_uncertain=True` and `language="uncertain"` or `"unknown"`.
- **Code-Switching Detection**: Identifies multi-script inputs or multi-language lexical marker mixing (`code_switched=True`).

### 2. Unicode Normalization & Cleaning

- `normalize_unicode(text, form="NFKC", strip_zero_width=True, strip_control_chars=True)`:
  - Supports `NFKC`, `NFC`, `NFD`, `NFKD`.
  - Removes zero-width joiners/spaces (`\u200B-\u200D`, `\uFEFF`, bidirectional controls).
  - Removes ASCII non-printable control characters (`\x00-\x1F`).
  - Preserves original input strings without in-place mutation.
- `clean_text(text, collapse_whitespace=True, casefold=False)`:
  - Collapses runs of whitespace, tabs, and newlines.

### 3. Multilingual Subword Representations

- `subword_ngrams: true` in task configuration enables character n-grams within word boundaries (`analyzer="char_wb", ngram_range=(3, 5)`).
- Critical for low-resource and morphologically rich languages (Bengali, Swahili, Hausa) where fixed word vocabularies suffer from high out-of-vocabulary rates.

### 4. Label-Preserving Safe Augmentation

Function: `augment_text(text: str, seed: int = 42, p: float = 0.2, perturbation="mixed") -> str`

- **Deterministic**: Controlled by RNG seed for reproducible experiments.
- **Conservative**: Modifies internal character pairs or duplicates letters without altering semantic keywords or sentence-level labels.
- **Safety Guarantee**: Never yields empty strings or corrupts label assignments.

## Integration into Task Configurations

Add the following options to any task YAML under `backend/configs/`:

```yaml
train_csv: fixtures/multilingual_train.csv
predict_csv: fixtures/multilingual_test.csv
text_column: text
target_column: label
task_type: binary_classification
split_strategy: stratified
validation_fraction: 0.25
seed: 42
metric: accuracy
prediction_column: prediction
positive_label: alert
language_normalize: NFKC
strip_zero_width: true
clean_text: true
subword_ngrams: true
```

## Running the Offline Multilingual Workflow

```bash
# 1. Train on multilingual dataset
python backend/scripts/pipeline_train.py \
  --config backend/configs/multilingual_text.yaml \
  --output-dir backend/outputs/m2_multilingual

# 2. Evaluate against held-out validation split
python backend/scripts/pipeline_evaluate.py \
  --model-dir backend/outputs/m2_multilingual

# 3. Predict on unseen multilingual test prompts
python backend/scripts/pipeline_predict.py \
  --model-dir backend/outputs/m2_multilingual \
  --output backend/submissions/m2_multilingual_submission.csv
```
