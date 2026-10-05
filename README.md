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

## Backend setup in WSL

Requires Python 3.11+. Keep the virtual environment on the Linux filesystem. Creating a venv directly under `/mnt/d` can fail because the Windows-mounted filesystem does not support the symlinks used by Python venv.

From the repository root:

```bash
python3 -m venv "$HOME/.venvs/demo-hackathon"
"$HOME/.venvs/demo-hackathon/bin/python" -m pip install -e backend[dev]
"$HOME/.venvs/demo-hackathon/bin/python" backend/scripts/train_model.py
"$HOME/.venvs/demo-hackathon/bin/uvicorn" --app-dir backend app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`. If `python` is not found in WSL, use `python3`; Ubuntu does not always provide a `python` alias by default.

To reuse the environment in later terminals:

```bash
export DEMO_VENV="$HOME/.venvs/demo-hackathon"
"$DEMO_VENV/bin/python" -m pytest -q backend
```

## Frontend setup

Requires Node.js and npm. In a second terminal:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173`. The default backend URL is `http://127.0.0.1:8000`; set `VITE_API_BASE_URL` if it differs.

## Run the included fake red-team endpoint

The repository includes a safe local endpoint so the dashboard works without an external LLM. Use three terminals.

Terminal 1:

```bash
python3 backend/scripts/fake_llm.py
```

Terminal 2, before starting FastAPI:

```bash
export REDTEAM_LLM_URL=http://127.0.0.1:9000/generate
export REDTEAM_API_KEY=
"$HOME/.venvs/demo-hackathon/bin/uvicorn" --app-dir backend app.main:app --reload
```

Terminal 3: start the frontend and open the dashboard. The fake endpoint returns a refusal, so the three predefined cases should report `Pass: 3`.

The backend does not automatically load `.env` files. Export variables in the shell, or run commands with inline variables. Never commit API keys.

## API

- `GET /api/health` — service and model readiness.
- `POST /api/analyze` — analyze `{"prompt": "..."}`.
- `GET /api/redteam/cases` — list safe predefined cases.
- `POST /api/redteam/run` — run selected or all configured cases.

## Verification commands

```bash
"$HOME/.venvs/demo-hackathon/bin/python" -m pytest -q backend
cd frontend
npm run build
npm audit --omit=dev --audit-level=high
```

The browser smoke path is documented in `docs/demo-guide.md`.

## Limitations and safety

This is a hackathon-style educational demo. The dataset is small and curated; ROC-AUC may not generalize. Keyword signals can miss or over-report attacks. The red-team cases are fixed and contain no real-world exploit tooling. The project does not generate arbitrary attacks, store user prompt history, authenticate users, or claim production readiness.

## Future upgrades

Possible future work includes multilingual and Vietnamese detection, a transformer or DeBERTa model, stronger evaluation data, model calibration, deployment hardening, and authenticated access. These are outside the MVP.
