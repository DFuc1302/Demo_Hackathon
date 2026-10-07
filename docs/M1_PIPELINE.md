# M1 Configurable Pipeline Guide

## Overview

The M1 pipeline provides an offline, task-independent workflow from CSV inputs through exploratory validation, deterministic splitting, baseline fitting, evaluation, and verified submission generation. It operates completely alongside the English detector without touching the live serving API, Transformer weights, or existing dataset manifests.

## Pipeline Architecture

- **Configuration**: YAML task contracts validated via `app.pipeline.config:load_config`.
- **Data & Splitting**: UTF-8 CSV ingestion, column validation, finite-number enforcement, duplicate-ID rejection, and deterministic cross-split leakage detection via `app.pipeline.data:read_dataset` and `split_dataset`.
- **Baselines & Inference**: TF-IDF text features and standardized numeric features fitted strictly on training rows; Ridge regression for continuous targets and Logistic Regression for binary/multiclass classification via `app.pipeline.baseline:fit_baseline` and `infer`.
- **Safe Persistence**: Pure JSON artifact serialization (`artifact.json`) with cryptographic hashes of config, model weights, and training features; no Python `pickle` dependencies.
- **CLIs**: Lightweight CLI entrypoints under `backend/scripts/` with structured JSON output and clean exit-code semantics.

## Supported Tasks and Metrics

| Task Type (`task_type`) | Supported Metrics (`metric`) | Split Strategy (`split_strategy`) | Notes |
|---|---|---|---|
| `binary_classification` | `accuracy`, `f1`, `f1_macro`, `roc_auc` | `stratified`, `random` | `f1` and `roc_auc` require explicit `positive_label` |
| `multiclass_classification` | `accuracy`, `f1_macro` | `stratified`, `random` | Requires $\ge 3$ distinct classes |
| `regression` | `rmse`, `mae`, `r2` | `random` | Continuous numeric target; `r2` requires $\ge 2$ validation samples |

## Configuration Contract

Configuration files are placed under `backend/configs/` (e.g. `backend/configs/binary_text.yaml`). All file paths resolve relative to the YAML file's directory:

```yaml
train_csv: fixtures/binary_train.csv
predict_csv: fixtures/binary_test.csv
text_column: text
feature_columns: []
id_column: id
target_column: label
task_type: binary_classification
split_strategy: stratified
validation_fraction: 0.25
seed: 42
metric: accuracy
prediction_column: prediction
sample_submission: fixtures/binary_sample_submission.csv
positive_label: attack
```

### Constraints & Invariants

1. **Feature Separation**: `target_column` and `id_column` cannot appear in `feature_columns` or `text_column`.
2. **Leakage Prevention**: Cross-split normalized feature signatures are detected and rejected.
3. **Train-Only Preprocessing**: TF-IDF vocabulary and numeric scaler mean/std are computed strictly on the training partition.
4. **Protected Paths**: Artifacts cannot be written to core data/model directories; outputs must target `backend/outputs/` or `backend/submissions/` (or temporary directories).

## Command Line Usage

### 1. Training

Fits preprocessing and baseline model on the training split, evaluates on the held-out validation split, and writes `artifact.json`:

```bash
python backend/scripts/pipeline_train.py \
  --config backend/configs/binary_text.yaml \
  --output-dir backend/outputs/m1_binary
```

### 2. Evaluation

Evaluates the saved model against held-out validation data or an external labeled test set:

```bash
# Evaluate against original held-out validation rows
python backend/scripts/pipeline_evaluate.py \
  --model-dir backend/outputs/m1_binary

# Evaluate against a separate labeled CSV
python backend/scripts/pipeline_evaluate.py \
  --model-dir backend/outputs/m1_binary \
  --input path/to/labeled_eval.csv
```

### 3. Prediction & Submission Generation

Generates predictions on unlabeled input CSV, preserving input row ordering and ID columns:

```bash
python backend/scripts/pipeline_predict.py \
  --model-dir backend/outputs/m1_binary \
  --output backend/submissions/m1_binary_submission.csv \
  --input backend/configs/fixtures/binary_test.csv \
  --sample-submission backend/configs/fixtures/binary_sample_submission.csv
```

When `--sample-submission` is provided, the CLI strictly verifies matching row counts, column names/order, and ID alignment before creating the output file.

## Offline Example Fixtures

Self-contained synthetic test fixtures are included in `backend/configs/fixtures/`:
- **Binary Classification**: `binary_train.csv` (24 rows), `binary_test.csv` (3 rows), `binary_sample_submission.csv`. Config: `backend/configs/binary_text.yaml`.
- **Regression**: `regression_train.csv` (24 rows, $y = 3x + 2$), `regression_test.csv` (3 rows), `regression_sample_submission.csv`. Config: `backend/configs/regression.yaml`.

## Verification & Testing

Run pipeline test suite:
```bash
pytest backend/tests/test_pipeline.py
```
Run entire backend test suite:
```bash
pytest backend/tests
```
