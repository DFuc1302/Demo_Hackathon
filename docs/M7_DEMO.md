# M7 Demo Dashboard & Pipeline API Integration Guide

## Overview

The M7 demo module (`backend/app/pipeline/demo.py`) bridges the M1-M6 pipeline infrastructure into the FastAPI application, exposing pipeline capabilities, unified security/multilingual analysis, experiment tracking, and interactive robustness checks without altering existing detector or red-team contracts.

## API Endpoints

Mounted under `/api/pipeline/`:

### 1. `GET /api/pipeline/capabilities`
Returns the full capabilities registry across all milestones:
- `supported_tasks`: Task types (`binary_classification`, `multiclass_classification`, `regression`) and their supported metrics.
- `target_languages`: Languages (`en`, `sw`, `ha`, `bn`) and scripts.
- `security_categories`: The complete threat taxonomy.
- `risk_thresholds`: Low and high risk thresholds.
- `robustness_transformations`: Perturbation mechanisms.
- `ensembling_methods`: Supported ensembling techniques.

### 2. `POST /api/pipeline/analyze`
Performs end-to-end security and language analysis on prompt input:
- Request: `{"prompt": "...", "task_type": "security", "model_dir": null}`
- Response: Structured `SecurityAnalysisResult` including score, risk level, heuristic signals, language estimate, and component confidence.

### 3. `GET /api/pipeline/runs`
Returns all indexed experiment runs from `backend/outputs/` for display in experiment dashboards.

### 4. `POST /api/pipeline/robustness-check`
Generates transformed prompt variations and analyzes perturbation sensitivity:
- Request: `{"prompt": "...", "transformations": ["spelling_noise", "unicode_variation"]}`
- Response: Transformed variations, language estimate, and original prompt.

## Frontend Client Integration

`frontend/src/api.js` provides client helpers for UI consumption:
- `getPipelineCapabilities(signal)`
- `analyzePipeline(prompt, taskType, modelDir, signal)`
- `getPipelineRuns(signal)`
- `checkPipelineRobustness(prompt, transformations, signal)`

## Safety and Preservation Invariants

1. **Existing Routes Unchanged**: Core serving routes (`/api/analyze`, `/api/health`, `/api/model-info`, `/api/multilingual/*`, `/api/redteam/*`) remain completely unaffected.
2. **Deterministic Fallbacks**: Heuristic and language analysis operate offline with zero external network or LLM dependency.
3. **Frontend Compilation**: Frontend builds cleanly with zero errors (`npm run build`).
