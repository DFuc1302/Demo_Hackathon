# Hackathon Challenge Adapter Guide

## Overview

This guide details how to adapt any hackathon challenge brief into the pipeline infrastructure on competition day without rewriting core codebase modules.

## Challenge Triage & Parameter Mapping

When the official challenge brief and dataset arrive, follow this translation protocol:

### 1. Task Type & Objective Mapping

| Challenge Task Shape | Pipeline `task_type` | Supported Metrics (`metric`) | Mandatory Constraints |
|---|---|---|---|
| Binary classification (0/1, benign/attack, True/False) | `binary_classification` | `accuracy`, `f1`, `f1_macro`, `roc_auc` | If `metric` is `f1` or `roc_auc`, specify explicit `positive_label` |
| Multi-class categorization (3+ categories) | `multiclass_classification` | `accuracy`, `f1_macro` | Must have $\ge 3$ distinct target classes |
| Numeric score / regression / continuous index | `regression` | `rmse`, `mae`, `r2` | Target must be continuous float; `split_strategy` must be `random` |

### 2. Dataset Schema Mapping

Inspect the official training and test CSV headers:
- `id_column`: Identifier column preserved through prediction (e.g. `id`, `uuid`, `row_id`).
- `target_column`: Target label column in training data.
- `text_column`: Primary unstructured text field (if natural language task).
- `feature_columns`: List of numeric/engineered tabular features.
- Invariant: `target_column` and `id_column` must never be included in `feature_columns` or `text_column`.

### 3. Language & Text Normalization Options

For text tasks involving low-resource languages or multilingual prompts:
- `language_normalize`: Use `NFKC` (recommended default) to standardize accented glyphs and full-width forms.
- `strip_zero_width: true`: Strips hidden joiners and invisible spaces (`\u200B-\u200D`, `\uFEFF`).
- `clean_text: true`: Collapses irregular whitespace.
- `subword_ngrams: true`: Essential for low-resource languages (Bengali, Swahili, Hausa) and code-switched text. Uses character n-grams (`char_wb`, $3-5$ grams) to maintain coverage under high OOV rates.

### 4. Sample Submission Validation

Verify official submission format using `fixtures/sample_submission.csv`:
- `prediction_column`: Exact name required by competition host (e.g. `prediction`, `label`, `target`, `prob`).
- Provide `--sample-submission <path>` to `pipeline_predict.py` to assert exact column ordering, row counts, and ID sequence before creating submission files.

---

## Configuration Templates

### Template A: Text Classification (Binary / Multi-Class)

Create `backend/configs/challenge_text.yaml`:
```yaml
train_csv: /path/to/official_train.csv
predict_csv: /path/to/official_test.csv
text_column: prompt
feature_columns: []
id_column: id
target_column: label
task_type: binary_classification
split_strategy: stratified
validation_fraction: 0.20
seed: 42
metric: accuracy
prediction_column: prediction
positive_label: "1"
language_normalize: NFKC
strip_zero_width: true
clean_text: true
subword_ngrams: false
sample_submission: /path/to/official_sample_submission.csv
```

### Template B: Multilingual / Low-Resource Challenge

Create `backend/configs/challenge_multilingual.yaml`:
```yaml
train_csv: /path/to/official_train.csv
predict_csv: /path/to/official_test.csv
text_column: text
feature_columns: []
id_column: id
target_column: category
task_type: multiclass_classification
split_strategy: stratified
validation_fraction: 0.20
seed: 42
metric: f1_macro
prediction_column: category
language_normalize: NFKC
strip_zero_width: true
clean_text: true
subword_ngrams: true
sample_submission: /path/to/official_sample_submission.csv
```

### Template C: Continuous / Regression Challenge

Create `backend/configs/challenge_regression.yaml`:
```yaml
train_csv: /path/to/official_train.csv
predict_csv: /path/to/official_test.csv
feature_columns:
  - feature_1
  - feature_2
  - feature_3
id_column: id
target_column: score
task_type: regression
split_strategy: random
validation_fraction: 0.20
seed: 42
metric: rmse
prediction_column: score
sample_submission: /path/to/official_sample_submission.csv
```

---

## Execution Sequence

```bash
# 1. Fit baseline model
python backend/scripts/pipeline_train.py \
  --config backend/configs/challenge_text.yaml \
  --output-dir backend/outputs/baseline_v1

# 2. Evaluate held-out validation metric
python backend/scripts/pipeline_evaluate.py \
  --model-dir backend/outputs/baseline_v1

# 3. Generate and validate official submission
python backend/scripts/pipeline_predict.py \
  --model-dir backend/outputs/baseline_v1 \
  --output backend/submissions/submission_v1.csv \
  --sample-submission /path/to/official_sample_submission.csv
```

## Baseline Artifact Contract

`pipeline_train.py` publishes exactly one model artifact:
`<output-dir>/artifact.json`. The pipeline intentionally does not use Python pickle or
joblib persistence.

| Legacy artifact expectation | Secure pipeline location |
|---|---|
| `model.joblib` | `artifact.json["model"]` and `artifact.json["preprocessing"]` |
| `config.json` | `artifact.json["config"]` |
| `metadata.json` | `artifact.json["run"]` |

Loading an artifact through `app.pipeline.core.load_artifact` validates its schema,
model/config hashes, split indices, and model dimensions before returning it. Use that
loader for integrity checks rather than parsing JSON directly:

```bash
PYTHONPATH=backend python -c "from app.pipeline.core import load_artifact; artifact, _ = load_artifact('backend/outputs/baseline_v1'); print(artifact['run']['run_id'])"
```

Do not create empty or duplicated `model.joblib`, `config.json`, or `metadata.json`
files to satisfy external filename checklists. Update the checklist to this contract.
