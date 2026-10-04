# AI Security Demo

Minimal AI security demonstration inspired by the RMIT Hackathon 2025, with a design that can evolve toward the RMIT Hackathon 2026 challenge.

## MVP

The demo will provide:

- Binary jailbreak detection for prompt text: `jailbreak` or `benign`.
- Jailbreak probability, risk level, and explainable detected signals.
- ROC-AUC evaluation for the V1 classifier.
- A safe red-team evaluation harness using predefined adversarial test cases.
- A configurable LLM endpoint for educational guardrail evaluation.
- A React web interface backed by a FastAPI service.

The V1 model uses TF-IDF features and Logistic Regression. The project deliberately avoids production-scale infrastructure and harmful exploit tooling.

## Planned stack

- Python and FastAPI
- scikit-learn
- React, Vite, and Tailwind CSS
- JSON or no persistence unless the demo actually requires it

## Repository status

This repository is being built milestone by milestone. See [`PLAN.md`](PLAN.md) for the implementation plan, acceptance criteria, verification steps, and Git workflow.

## Local development

Backend setup:

```bash
cd backend
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/train_model.py
.venv/bin/uvicorn app.main:app --reload
```

The API starts at `http://127.0.0.1:8000`. The health endpoint is
`GET /api/health`, and prompt analysis uses `POST /api/analyze` with a JSON
body such as `{"prompt":"your prompt here"}`.

The frontend setup and commands will be documented when the React application
is added. The expected development flow is:

1. Create and activate a Python environment.
2. Install backend dependencies.
3. Train the baseline model.
4. Start the FastAPI backend.
5. Install frontend dependencies.
6. Start the Vite development server.
7. Open the local web demo.

The backend model artifact is generated locally and is excluded from Git.
No frontend application code is included yet.

## Educational limitations

This is a hackathon-style demonstration, not a production security control. A small synthetic or curated dataset can produce misleading metrics, and keyword-based signals can miss or over-report attacks. Red-team cases remain predefined and educational; the project does not generate or automate harmful real-world exploits.

## Future upgrades

Possible later upgrades include multilingual and Vietnamese detection, a transformer or DeBERTa-based model, improved evaluation data, and deployment hardening. These are out of scope for the MVP.
