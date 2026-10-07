# M6 Experiment Tracking and Ensembling Guide

## Overview

The M6 experiment tracking and ensembling module (`backend/app/pipeline/experiments.py`) enables systematic comparison of pipeline baseline runs, side-by-side metric diffing, and multi-model ensembling across classification and regression tasks.

## Core Capabilities

### 1. Experiment Registry & Run Comparison

Class: `ExperimentRegistry(runs_dir)`

- **Artifact Scanning**: Recursively indexes all run directories containing `artifact.json`.
- **Run Summaries**: Extracts model type, task type, metric score, seed, feature columns, subword n-gram configuration, runtime, and cryptographic hashes.
- **Side-by-Side Comparison**: Builds structured comparisons across multiple model directories, ranking models based on metric optimization direction (`higher_is_better`).

### 2. Multi-Model Ensembling

Class: `EnsembleModel(model_dirs, weights=None, method="average")`

- **Supported Methods**:
  - `average`: Unweighted arithmetic average of continuous outputs (regression) or soft probability distributions (classification).
  - `weighted`: User-specified normalized weighting ($\sum w_i = 1.0$) across members.
  - `majority_vote`: Discrete hard voting per sample across classification members.
- **Justified Ensembling Invariant**: Evaluates ensemble performance against all individual member scores. An ensemble is marked `justified_improvement: true` only if its score matches or improves upon the best individual member.

## CLI Usage

### Scan Run Directory
```bash
python backend/scripts/pipeline_experiments.py \
  --scan-dir backend/outputs
```

### Compare Runs Side-by-Side
```bash
python backend/scripts/pipeline_experiments.py \
  --compare backend/outputs/m1_binary backend/outputs/m2_multilingual
```

### Evaluate Ensemble
```bash
python backend/scripts/pipeline_experiments.py \
  --ensemble backend/outputs/m1_binary backend/outputs/m2_multilingual \
  --data backend/configs/fixtures/binary_train.csv \
  --method average
```
