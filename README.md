# AI Security Demo

Minimal AI security demonstration inspired by the RMIT Hackathon 2025, designed to evolve toward the RMIT Hackathon 2026 challenge.

## What it demonstrates

- Binary prompt classification: `jailbreak` or `benign`.
- Jailbreak probability, explicit risk level, and explainable text signals.
- A safe red-team harness using three predefined educational cases.
- A configurable LLM endpoint for guardrail pass/fail evaluation.
- A React/Vite/Tailwind web interface backed by FastAPI.

The V1 detector uses TF-IDF features and Logistic Regression. It is a demo signal, not a production security decision.

## Repository layout

- `backend/` — model training, inference, FastAPI API, and tests.
- `frontend/` — React/Vite/Tailwind demo UI.
- `PLAN.md` — milestone scope, acceptance criteria, and Git workflow.
- `docs/demo-guide.md` — presentation and evaluation walkthrough.

## Backend setup

Requires Python 3.11+. From the repository root:

```bash
cd backend
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/train_model.py
.venv/bin/uvicorn app.main:app --reload
```

On Windows PowerShell, use `.venv\Scripts\python.exe` and `.venv\Scripts\uvicorn.exe` instead of the POSIX paths.

The API runs at `http://127.0.0.1:8000`.

## Frontend setup

Requires Node.js and npm. In a second terminal:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

The Vite server normally runs at `http://localhost:5173`. Set `VITE_API_BASE_URL` when the backend uses another URL; the default is `http://127.0.0.1:8000`.

## Red-team endpoint configuration

The red-team dashboard is disabled until an endpoint is configured. The backend reads:

```bash
export REDTEAM_LLM_URL=http://127.0.0.1:9000/generate
export REDTEAM_API_KEY=
```

Use a local fake endpoint for deterministic demos. Never commit API keys or put them in frontend variables. The harness sends JSON containing `prompt`, enforces a timeout, limits response size to 16 KiB, stores only a bounded response excerpt, and reports `pass`, `fail`, or `error`.

## API

- `GET /api/health` — service and model readiness.
- `POST /api/analyze` — analyze `{"prompt": "..."}`.
- `GET /api/redteam/cases` — list safe predefined cases.
- `POST /api/redteam/run` — run selected or all configured cases.

## Verification commands

```bash
cd backend
.venv/bin/python -m pytest -q

cd ../frontend
npm run build
npm audit --omit=dev --audit-level=high
```

The browser smoke path is documented in `docs/demo-guide.md`.

## Limitations and safety

This is a hackathon-style educational demo. The dataset is small and curated; ROC-AUC may not generalize. Keyword signals can miss or over-report attacks. The red-team cases are fixed and contain no real-world exploit tooling. The project does not generate arbitrary attacks, store user prompt history, authenticate users, or claim production readiness.

## Future upgrades

Possible future work includes multilingual and Vietnamese detection, a transformer or DeBERTa model, stronger evaluation data, model calibration, deployment hardening, and authenticated access. These are outside the MVP.
