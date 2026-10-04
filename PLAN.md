# Implementation Plan: Minimal AI Security Demo

## Repository inspection

- Repository was empty at planning time.
- No tracked files or commits existed.
- Current branch: `master`.
- GitHub remote configured as `origin`.

No implementation, dependency installation, or application files are included in this milestone.

## 1. Project objective

Build a small, explainable AI security demo that:

1. Classifies a prompt as `jailbreak` or `benign`.
2. Returns jailbreak probability, risk level, and detected text signals.
3. Runs predefined educational red-team prompts against a configurable LLM endpoint.
4. Displays both analyses in a React web interface.
5. Uses a simple architecture that can later support multilingual or transformer-based detection.

## 2. MVP scope

### Jailbreak detection

- TF-IDF text features.
- Logistic Regression classifier.
- Binary labels: `jailbreak` and `benign`.
- Probability output.
- ROC-AUC evaluation.
- Explicit risk thresholds.
- Explainable signal extraction using predefined patterns.

### Backend

- FastAPI service.
- `GET /api/health`.
- `POST /api/analyze`.
- Red-team API endpoints.
- Configurable LLM endpoint through environment variables.
- Request validation and bounded input length.
- No database.

### Frontend

- React + Vite.
- Prompt textarea and Analyze button.
- Probability, label, risk level, and detected signals.
- Red-team dashboard.
- Loading, empty, and error states.
- Basic responsive and accessible UI.

### Red-team harness

- Safe, predefined adversarial prompts only.
- No exploit automation.
- Configurable target LLM endpoint.
- Record test case, response status, guardrail pass/fail, bounded response information, and errors.
- JSON output is sufficient for the MVP.

## 3. Non-goals

- Production-grade content moderation.
- Harmful real-world exploit tooling.
- Autonomous agents.
- Prompt injection automation against arbitrary systems.
- User accounts or authentication.
- PostgreSQL, Redis, Celery, Docker, LangChain, vector databases, or microservices.
- Persistent user prompt history.
- Transformer model in V1.
- Multilingual support in the MVP.
- Public deployment requirement.
- Claims that the model is safe for production decisions.

## 4. Proposed architecture

```text
React/Vite frontend
        │ HTTP JSON
        ▼
FastAPI backend
        │
        ├── Inference module
        │       ├── TF-IDF vectorizer
        │       ├── Logistic Regression model
        │       ├── risk mapping
        │       └── signal extraction
        │
        ├── Red-team service
        │       ├── predefined test cases
        │       ├── configurable LLM client
        │       └── pass/fail evaluator
        │
        └── Model/data files
                ├── small labeled dataset
                └── serialized trained model
```

### Backend decisions

- Use FastAPI and Pydantic request/response models.
- Keep business logic independent from HTTP handlers.
- Load the model once at application startup.
- Return stable JSON schemas.
- Use the smallest suitable HTTP client; standard library is preferred where practical.
- Use timeouts and response-size limits for outbound LLM calls.
- Fail clearly when the red-team endpoint is not configured.

### Frontend decisions

- Keep API calls in one small client module.
- Use simple React state; no state-management library.
- Use Tailwind only for styling.
- Render backend-provided risk and signal values rather than duplicating classifier logic in JavaScript.

### Data and model decisions

- Store the small dataset as CSV or JSON.
- Store the trained model artifact under a generated model directory.
- Do not commit large or secret-bearing artifacts.
- Include a reproducible training command and metadata such as ROC-AUC and dataset version.

## 5. Proposed directory structure

```text
Demo_Hackathon/
├── README.md
├── PLAN.md
├── .gitignore
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── inference.py
│   │   ├── signals.py
│   │   └── redteam.py
│   ├── data/
│   │   └── prompts.csv
│   ├── models/
│   │   └── .gitkeep
│   ├── scripts/
│   │   └── train_model.py
│   └── tests/
│       ├── test_training.py
│       ├── test_inference.py
│       ├── test_api.py
│       └── test_redteam.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── components/
│       └── styles.css
└── docs/
    └── demo-guide.md
```

The structure may be simplified if a file proves unnecessary.

## 6. Milestones

### M0 — Repository bootstrap

**Scope:** Create the minimal repository foundation and developer instructions.

**Acceptance criteria:**

- `README.md` describes the project, setup, and intended commands.
- `PLAN.md` contains the milestone plan.
- `.gitignore` excludes Python environments, Node artifacts, generated model files where appropriate, and secrets.
- No application code or dependencies are added.
- Git status is clean after commit and push.

**Verification:** Validate Git state, inspect the diff, check for secrets/generated artifacts, commit only M0 files, push M0, and confirm a clean working tree.

**Suggested commit:**

```text
chore: bootstrap AI security demo repository
```

### M1 — Dataset and baseline classifier

**Scope:** Add a small labeled dataset and reproducible TF-IDF + Logistic Regression training pipeline.

**Acceptance criteria:**

- Dataset contains both `jailbreak` and `benign` examples.
- Training is deterministic and uses a reproducible split.
- Training reports ROC-AUC.
- Invalid labels or insufficient class diversity fail clearly.
- Generated model can be saved and reloaded.
- Dataset contains safe educational examples only.

**Verification:** Test dataset loading and validation, run training, confirm finite ROC-AUC, reload the model artifact, inspect balance, commit and push only M1 files.

**Suggested commit:**

```text
feat: add baseline jailbreak classifier training
```

### M2 — Inference module

**Scope:** Create the reusable inference layer used by the API and tests.

**Acceptance criteria:**

- Accepts prompt text and returns predicted label, jailbreak probability, risk level, and detected signals.
- Handles empty and oversized input predictably.
- Uses the trained artifact without retraining.
- Risk thresholds are explicit and documented.
- Signal extraction is deterministic and explainable.
- Model-load failures are actionable.

**Verification:** Test benign, adversarial-looking, empty, and boundary-length prompts; verify probability bounds and threshold behavior; run a command-line smoke check; commit and push only M2 files.

**Suggested commit:**

```text
feat: add jailbreak inference and risk scoring
```

### M3 — FastAPI backend

**Scope:** Expose inference and health checks through a stable API.

**Acceptance criteria:**

- `GET /api/health` returns service status.
- `POST /api/analyze` validates input and returns the inference response.
- Invalid requests return appropriate 4xx responses.
- CORS supports the local Vite frontend.
- API responses use explicit schemas.
- Backend starts with a documented command.
- Requests do not retrain the model.

**Verification:** Test health, valid analysis, invalid input, and model-load failure; start the server; send real HTTP requests; confirm JSON shape; commit and push only M3 files.

**Suggested commit:**

```text
feat: expose jailbreak analysis through FastAPI
```

### M4 — React demo UI

**Scope:** Build the frontend for prompt analysis.

**Acceptance criteria:**

- User can enter and submit a prompt.
- UI displays probability, label, risk level, and detected signals.
- Loading and error states are visible.
- Empty input is blocked or clearly rejected.
- Layout works at common desktop and mobile widths.
- Basic keyboard and label accessibility is present.

**Verification:** Run Vite, use a real browser to submit a prompt and trigger an error, run the production build, commit and push only M4 files.

**Suggested commit:**

```text
feat: add prompt analysis web interface
```

### M5 — Red-team evaluation harness

**Scope:** Add safe predefined test cases and configurable LLM evaluation.

**Acceptance criteria:**

- Test cases have stable IDs.
- Cases are educational and contain no real-world exploit instructions.
- LLM endpoint and optional API key use environment variables.
- Requests have timeout and response-size limits.
- Results record pass, fail, or error status.
- Results are returned through the backend.
- An unconfigured endpoint fails explicitly without crashing the service.

**Verification:** Test pass, fail, timeout, malformed response, and unconfigured endpoint paths using a local fake endpoint or deterministic test double; confirm valid JSON and no credential logging; commit and push only M5 files.

**Suggested commit:**

```text
feat: add safe red-team evaluation harness
```

### M6 — Integration and testing

**Scope:** Connect the red-team dashboard to the frontend and validate the complete user flow.

**Acceptance criteria:**

- Frontend can start a red-team evaluation.
- Dashboard displays case status and summary counts.
- Analyze and red-team features work against the same backend.
- Backend and frontend contracts are documented.
- Tests cover main API paths and failure states.
- No milestone-specific regressions remain.

**Verification:** Run backend tests and frontend checks/build; start both applications; use the browser to analyze a prompt and run the red-team dashboard; review against this plan; commit and push only M6 files.

**Suggested commit:**

```text
feat: integrate red-team dashboard and end-to-end flow
```

### M7 — Demo polish and documentation

**Scope:** Make the project reproducible and presentation-ready without expanding the architecture.

**Acceptance criteria:**

- README contains setup, training, backend, frontend, and demo instructions.
- Demo guide explains limitations and educational purpose.
- Environment variables are documented without exposing secrets.
- UI terminology and empty/error states are consistent.
- Known limitations and future upgrade paths are documented.
- Final smoke test passes.

**Verification:** Follow setup instructions as far as available, run backend tests and frontend build, perform a final browser smoke test, review against this plan, inspect Git status/diff, commit and push only M7 files, and confirm a clean working tree.

**Suggested commit:**

```text
docs: polish AI security demo and usage guide
```

## 7. Testing strategy

### Backend unit tests

- Dataset validation.
- Model training and reload.
- Probability bounds.
- Risk threshold boundaries.
- Signal extraction.
- Input validation.
- Red-team result classification.
- Timeout and malformed endpoint responses.

### Backend API tests

- Health endpoint.
- Valid analysis request.
- Empty and oversized prompt handling.
- Model initialization failure.
- Red-team configuration errors.
- Stable response schema.

### Frontend verification

- Production build.
- Real-browser prompt analysis.
- Loading and error states.
- Red-team dashboard behavior.
- Responsive layout and basic keyboard accessibility.

### Required smoke path

```text
Start backend
Start frontend
Open browser
Analyze prompt
Inspect probability, risk, and signals
Run red-team evaluation
Inspect pass/fail dashboard
```

## 8. Git/GitHub strategy

For every milestone:

1. Implement only that milestone.
2. Run relevant tests and smoke checks.
3. Review implementation against `PLAN.md`.
4. Check `git status` and `git diff`.
5. Stage only milestone files.
6. Commit using the milestone-specific message.
7. Push to `origin/master`.
8. Verify the working tree is clean.
9. Start the next milestone only after all previous steps succeed.

Do not combine milestone commits. Do not commit secrets, local virtual environments, Node artifacts, or unreviewed generated files.

## 9. Risks

| Risk | Impact | Mitigation |
|---|---:|---|
| Tiny synthetic dataset produces misleading metrics | High | Label as a demo, report ROC-AUC, and document dataset limits |
| Classifier overfits keywords | High | Use varied examples and document heuristic limitations |
| Model artifact paths differ across environments | Medium | Resolve paths relative to the backend project root |
| LLM endpoint is unavailable | Medium | Return explicit configuration/error status and test with a local fake endpoint |
| Outbound responses are too large or slow | Medium | Set timeout and response-size limits |
| Frontend/backend CORS mismatch | Medium | Configure explicit local development origins |
| Users treat output as a security guarantee | High | Add a visible educational disclaimer and documentation |
| Scope expands toward production tooling | High | Keep predefined safe cases and prohibit arbitrary exploit generation |
| GitHub push authentication fails | Medium | Verify remote access before M0 completion and report blockers clearly |

## 10. Dependencies

### Runtime

- Python 3.11 or compatible current Python version.
- Node.js and npm.
- FastAPI.
- Uvicorn.
- scikit-learn.
- pandas only if it materially simplifies dataset handling; otherwise use Python CSV support.
- joblib if needed for model serialization.
- React.
- Vite.
- Tailwind CSS.

### Development

- pytest.
- FastAPI-compatible HTTP test client.
- Browser for manual verification.
- Git and GitHub remote access.

Dependency policy: use the smallest set that supports the required behavior. Avoid adding a package when the standard library or an existing dependency is sufficient.

## 11. Exact first implementation task

### M0 task — Bootstrap the repository

Create only:

```text
README.md
PLAN.md
.gitignore
```

Acceptance criteria:

- `README.md` identifies the project, MVP purpose, planned stack, local development commands to be added later, and educational limitations.
- `PLAN.md` contains this milestone plan.
- `.gitignore` excludes Python virtual environments, `__pycache__`, pytest/cache files, Node dependencies and build output, local `.env` files, and generated model artifacts where appropriate.
- No application code or dependencies are added.
- Validate the files, review the diff, commit only these M0 files, push to GitHub, and verify a clean working tree.

Suggested commit:

```text
chore: bootstrap AI security demo repository
```
