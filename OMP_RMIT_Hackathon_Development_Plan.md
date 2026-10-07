# OMP Handoff — RMIT Hackathon 2026 Demo

## Objective

Adapt the existing demo in `/mnt/d/Demo_Hackathon` (Windows: `D:\Demo_Hackathon`) into a reusable, task-adaptable hackathon project. Preserve useful working functionality and make changes incrementally. The project is managed with GitHub; after each completed milestone, prepare a focused commit and push it to the configured remote.

The official theme described for the event is **Security × Generative AI × Low-Resource Languages**. Exact task briefs are expected on hackathon day, so build reusable infrastructure and avoid hard-coding assumptions about the final tasks. Re-check the official brief and dataset when available; treat this plan as preparation, not as a substitute for the rules.

## Critical instruction: begin with M0 only

**Do not implement, refactor, reorganize, delete, or rewrite code during M0.** First inspect the existing repository and produce the audit documents below. Do not run destructive commands. Preserve current functionality and report any uncertainty.

### M0 — Audit the existing demo (documentation only)

Inspect the repository, entry points, runtime instructions, dependencies, tests, data flow, UI/API, model code, and Git state. Do not expose secrets; do not read or copy credential values into reports.

Create or update:

- `docs/CURRENT_ARCHITECTURE.md` — actual architecture and how to run the demo.
- `docs/GAP_ANALYSIS.md` — existing capabilities versus the preparation goals below.
- `docs/MIGRATION_PLAN.md` — staged, low-risk changes grounded in the actual codebase.

For every existing component, classify it as **KEEP**, **REFACTOR**, **REPLACE**, **REMOVE**, or **NEW**, with a short evidence-based reason. Identify dependencies, risks, and a minimal smoke-test procedure. Clearly distinguish observed facts from assumptions.

At the end of M0, summarize findings and propose the exact scope for M1. **Stop before implementation and wait for my review of the audit.** Do not start M1 in the same run.

## M1–M8 preparation roadmap

These are candidate milestones, not permission to build everything blindly. After M0, use the audit and my review to adjust scope. Keep the original demo working and avoid speculative features that do not fit the real repository or competition rules.

### M1 — Configurable data and Kaggle workflow

Build a small, task-independent path from CSV input through schema checks, exploratory summaries, preprocessing, train/validation, model fitting, evaluation, inference, and submission generation. Configure columns, task type, split strategy, and metric rather than hard-coding them.

Possible interface:

```bash
python -m src.train --config configs/baseline.yaml
python -m src.evaluate --model outputs/model
python -m src.predict --input data/test.csv --output submissions/submission.csv
```

Validate expected columns, row counts, missing values, prediction shape, and sample-submission format. Never assume every task is binary classification or ROC-AUC; derive these from the official task.

### M2 — Multilingual and low-resource text preparation

Create reusable language utilities for language identification (including unknown/uncertain), Unicode normalization, conservative noise cleanup, multilingual representations, and optional augmentation. Keep the raw input and make transformations reproducible/configurable. Avoid translation or augmentation that could change labels without validation.

### M3 — Security analysis module

Prepare modular, measurable support for security-related classification, including prompt injection, adversarial input, and data-quality/poisoning signals where relevant. Return structured outputs such as score, predicted category, language estimate, and component confidence. Treat heuristics as signals, not ground truth; calibrate probabilities against held-out data.

### M4 — Robustness evaluation

Add a controlled test suite for benign transformations such as paraphrase, translation, Unicode variation, spelling/spacing noise, and code-switching, using authorized datasets and clearly labeled synthetic examples. Measure clean and transformed performance by language and relevant subgroup. Keep test generation separate from production inference and document limits.

### M5 — GenAI reliability and grounding components

If the challenge needs generation or retrieval, provide interchangeable interfaces for retrieval, evidence capture, response validation, and confidence reporting. Do not call an external model unless the task, available credentials, network policy, and cost are understood. Keep secrets out of code and logs. Do not build a full RAG system unless it maps to a real challenge or useful demo.

### M6 — Experiment tracking and ensembling

Record configuration, data split/seed, model, features, validation metrics, runtime, and notes for each experiment. Compare simple baselines before expensive models. Add ensembling only when cross-validation supports it and the competition rules permit it. Track compute and memory use.

### M7 — Demo dashboard

Retain and adapt the existing demo UI as a clear view over the pipeline. Prioritize input, selected task/configuration, prediction, confidence or risk, language handling, and evaluation evidence. Add adversarial or grounding visualizations only when implemented and backed by real outputs. Do not let UI work delay a valid training/inference/submission workflow.

### M8 — Challenge-day adapter and runbook

Create `docs/CHALLENGE_ADAPTER.md` and `docs/HACKATHON_RUNBOOK.md`. On challenge day:
1. Read the official task statement and rules.
2. Identify task type, target, metric, split rules, dataset schema, and submission format.
3. Map the task to existing modules; note unsupported requirements.
4. Create a task-specific config and run a minimal baseline.
5. Validate locally with the official metric and format.
6. Inspect errors, improve one change at a time, and record results.
7. Generate and verify the submission artifact.
8. Keep the demo consistent with what the model actually does.

Do not redesign the project just because a task is new. Extend only what the task requires.

## Candidate repository shape

Use this only if the audit supports it; do not reorganize the repository solely to match this diagram.

```text
configs/       task and model configuration
data/          raw, processed, external (large/private data stays out of Git)
src/
  data/        loading and validation
  language/    multilingual preprocessing
  security/    security-related analysis
  models/      baselines and model adapters
  genai/       optional retrieval/generation interfaces
  evaluation/  metrics, robustness, error analysis
  kaggle/      inference and submission checks
frontend/      existing demo UI, adapted as needed
api/           existing or minimal service interface
tests/         data, model, security, multilingual smoke tests
experiments/   small, reproducible experiment records
submissions/   generated outputs (exclude private/large artifacts as appropriate)
docs/          architecture, gaps, migration, adapter, runbook
```

## Working rules for OMP

- Start by checking repository status and reading project instructions (for example, `AGENTS.md`, README, and existing docs).
- For M0, inspect and document only. Do not modify application code or run formatters/migrations.
- Before each later milestone, state its scope, files likely to change, and verification steps. Keep changes within that milestone.
- Preserve existing behavior unless a change is necessary and documented. Prefer small, reversible edits.
- Never delete user data, overwrite datasets, or commit secrets. Keep large datasets, model weights, credentials, and generated caches out of Git unless explicitly required.
- Run relevant existing tests and a focused smoke test after implementation; report commands and results honestly.
- Check `git diff` and repository status before committing. Commit only the completed milestone’s intended files. Push only to the already configured project remote and report the commit/push result; if remote setup or credentials are unavailable, leave the commit ready and explain the blocker.
- Do not claim a capability is implemented until it runs and has been verified.
