# Hackathon Day-Of Runbook

## Overview

This runbook defines the operational protocol for the RMIT Hackathon 2026. Follow these sequential phases to deliver verified, reproducible submissions and demonstration claims.

---

## Phase 0: Triage & Brief Intake (Minute 0–15)

1. **Read Official Task & Rules**:
   - Determine competition metric (e.g. Accuracy, $F_1$, ROC-AUC, RMSE).
   - Identify submission limit per day (e.g. 5 submissions) and deadline.
   - Confirm allowed external data and library constraints.
2. **Download & Inspect Official Assets**:
   - Save datasets into `backend/data/` (or dedicated local directories).
   - Locate `train.csv`, `test.csv`, and `sample_submission.csv`.
3. **Map Task Parameters**:
   - Consult `docs/CHALLENGE_ADAPTER.md` to map task shape to pipeline settings.

---

## Phase 1: Environment & Data Ingestion (Minute 15–30)

1. **Verify Python Environment**:
   ```bash
   /tmp/demo-v2-venv/bin/python --version
   python -c "import app.pipeline.core, app.pipeline.language; print('Ready')"
   ```
2. **Inspect CSV Schema & Integrity**:
   - Check headers: `head -n 2 train.csv test.csv sample_submission.csv`.
   - Verify non-empty rows and required ID columns.
3. **Author Task Configuration**:
   - Copy the appropriate template from `docs/CHALLENGE_ADAPTER.md` into `backend/configs/challenge.yaml`.

---

## Phase 2: First Verified Submission (Minute 30–45)

**Goal**: Establish an end-to-end working baseline from data to submission before engineering complex models.

1. **Fit Initial Baseline**:
   ```bash
   python backend/scripts/pipeline_train.py \
     --config backend/configs/challenge.yaml \
     --output-dir backend/outputs/baseline_v1
   ```
2. **Verify Baseline Metrics**:
   - Inspect output JSON for validation score and `artifact.json` integrity.
3. **Generate First Valid Submission**:
   ```bash
   python backend/scripts/pipeline_predict.py \
     --model-dir backend/outputs/baseline_v1 \
     --output backend/submissions/submission_baseline.csv \
     --sample-submission path/to/official_sample_submission.csv
   ```
4. **Submit First Artifact**:
   - Upload `submission_baseline.csv` to competition platform to verify format acceptance.

---

## Phase 3: Iteration & Comparison (Minute 45–180)

1. **Single-Variable Hypothesis Testing**:
   - Test one improvement at a time (e.g., text normalization, subword n-grams, feature engineering, alternative seeds).
   - Save each experiment to a unique output directory (`backend/outputs/exp_v2_subwords`, `backend/outputs/exp_v3_features`).
2. **Side-by-Side Performance Comparison**:
   ```bash
   python backend/scripts/pipeline_experiments.py \
     --compare \
     backend/outputs/baseline_v1 \
     backend/outputs/exp_v2_subwords \
     backend/outputs/exp_v3_features
   ```
3. **Error Analysis**:
   - Inspect held-out validation errors using `pipeline_evaluate.py`.

---

## Phase 4: Robustness & Ensembling (Minute 180–240)

1. **Robustness Evaluation**:
   ```bash
   python backend/scripts/pipeline_robustness_eval.py \
     --model-dir backend/outputs/exp_v2_subwords \
     --output-report backend/outputs/robustness_v2.json
   ```
   - Ensure the model does not suffer catastrophic degradation under spelling, Unicode, or spacing perturbations.
2. **Justified Ensembling**:
   - Combine top diverse models (e.g., word-level baseline + subword model):
   ```bash
   python backend/scripts/pipeline_experiments.py \
     --ensemble \
     backend/outputs/baseline_v1 \
     backend/outputs/exp_v2_subwords \
     --data path/to/validation_data.csv \
     --method average
   ```
   - Invariant: Only deploy the ensemble if `justified_improvement: true`.

---

## Phase 5: Final Submission Preparation (Minute 240–270)

1. **Generate Final Submission**:
   ```bash
   python backend/scripts/pipeline_predict.py \
     --model-dir backend/outputs/best_model \
     --output backend/submissions/final_submission.csv \
     --sample-submission path/to/official_sample_submission.csv
   ```
2. **Verify Submission File**:
   - Verify SHA-256 hash: `sha256sum backend/submissions/final_submission.csv`.
   - Check line count: `wc -l backend/submissions/final_submission.csv sample_submission.csv`.
   - Verify no null/NaN values: `grep -E "NaN|null|None|^," backend/submissions/final_submission.csv`.

---

## Phase 6: Demo Alignment & Presentation (Minute 270–300)

1. **Ensure Truthful Alignment**:
   - The presentation and web interface must report the exact held-out validation metrics recorded in `artifact.json`.
   - Do not claim uncalibrated scores as calibrated.
2. **Launch Interactive Demo**:
   ```bash
   # Terminal 1: Backend
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

   # Terminal 2: Frontend
   cd frontend && npm run dev -- --host 0.0.0.0 --strictPort
   ```
3. **Demonstrate Verified Capabilities**:
   - Show baseline pipeline results, multilingual language detection, and security analysis scan.

---

## Troubleshooting & Emergency Protocols

| Incident | Cause | Emergency Fix |
|---|---|---|
| `MemoryError` / CUDA OOM | Batch size or text representation too large | Reduce n-gram range, or run on CPU with `CUDA_VISIBLE_DEVICES=""`. |
| `feature_columns` mismatch | Train and test CSV have different column names | Check headers with `head -n 1`, update `challenge.yaml` column names. |
| Non-finite target in regression | Target column contains `NaN` or `inf` | Clean training data: filter rows where `target` is non-finite before training. |
| Sample submission ID mismatch | Predict output has different ordering | Ensure `--sample-submission` flag is supplied; `pipeline_predict.py` enforces ID alignment. |
