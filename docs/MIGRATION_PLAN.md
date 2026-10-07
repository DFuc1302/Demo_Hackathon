# Migration plan — M0 proposal, not implementation approval

## Decision and invariants

Preserve `backend/app` and `frontend`. Reject a repository-wide move into the candidate top-level `src/` shape: the existing FastAPI package, scripts, fixtures, CI and React client already establish boundaries (`backend/pyproject.toml`, `backend/app/main.py`, `backend/scripts/`, `backend/tests/`, `frontend/src/main.jsx`, `.github/workflows/ci.yml`). A second top-level Python package would duplicate those conventions without official challenge evidence. `OMP_RMIT_Hackathon_Development_Plan.md:83–105` explicitly makes its diagram conditional.

M0 creates only `docs/CURRENT_ARCHITECTURE.md`, `docs/GAP_ANALYSIS.md`, and this document. KEEP/REFACTOR/REPLACE/REMOVE/NEW classifications are in the gap analysis. No working core is replaced, no source is removed, no legacy artifact is deleted, and no dataset/model is overwritten. M1–M8 below are proposals with explicit entry gates; this document is not permission to execute them.

Preserve these boundaries throughout later approved work:

- Existing English `/api/analyze` seven-field schema, model/tokenizer/probabilities and metadata remain unchanged unless separately authorized (`backend/app/inference.py`, `backend/app/schemas.py`). Do not use a new task config to alter release calibration or weaken metadata gates.
- Optional multilingual failures remain fail-closed and independent of English health (`backend/app/main.py`); translated pilot metrics are not native-speaker evidence. Translation-assisted scores stay uncalibrated (`backend/app/multilingual_guardrail.py`).
- Retain raw data and provenance; never overwrite datasets, secrets, model directories or ignored user artifacts. Reuse hashing/split-validation ideas from `backend/app/dataset_contract.py` and safe inventory/error conventions from `backend/app/translation_artifact.py`, without importing detector assumptions into generic training.
- Current worktree has 22 modified and 38 untracked user files, no staged files at baseline; history does not contain all observed V2/multilingual work. M0 publishes only the three audit docs; later commits must isolate reviewed intended files (`docs/CURRENT_ARCHITECTURE.md`).
- No frontend/API coupling until a real challenge adapter needs it. Keep local/CI checks offline using tiny fixtures. No speculative retries, providers, RAG framework or distributed infrastructure.

## M1 — Exact next proposed scope: backend/CLI-only configurable pipeline

**Entry:** user reviews the M0 audit and approves a separate M1 plan. Inspect available official task fields/sample submission; if the brief is still unavailable, use explicit synthetic fixtures and label them preparation, not official challenge support. Finalize config and CLI names/signatures in M1 planning once actual task fields are known. Resolve supported split/metric combinations explicitly rather than guessing competition requirements.

### Exact deliverables

1. **Small task configuration contract under proposed `backend/configs/`.** YAML/schema defines training and prediction input paths; text/feature columns; optional ID column; target column; task type (`binary_classification`, `multiclass_classification`, `regression`); split strategy; seed; metric; prediction column; optional sample-submission path. Clarify path resolution and declared column order. Validate unknown/unsupported task/metric/split combinations early. Treat config as data via safe YAML loading; no secret values or executable config.
2. **Task-independent package under proposed `backend/app/pipeline/`.** Config loading, CSV/schema validation, exploratory summaries, preprocessing, deterministic splitting, simple scikit-learn baselines, metric dispatch, artifact/run metadata, evaluation, prediction and submission-shape validation. Summaries cover row counts/types/nulls and target distribution without printing private row contents. Fit preprocessing on training only; apply the same fitted transform at evaluation/prediction. Preserve raw inputs and prediction row order/optional IDs. Start with conservative baselines appropriate to supported feature/task types; no transformer requirement. Reuse already-declared PyYAML/scikit-learn and hashing/provenance patterns; avoid new dependencies absent a concrete requirement. Keep generic task dispatch separate from DeBERTa/XLM-R/M2M100 and binary `text,label` contracts.
3. **Thin train/evaluate/predict CLI entry points under `backend/scripts/`.** Names and exact flags are finalized during M1 planning after official fields are known; the handoff's `python -m src.train` examples are not current entry points. Keep business logic in the package, not copied between scripts. Train produces a fitted artifact plus run record; evaluate consumes artifact/config and held-out labeled input; predict consumes artifact/config and test CSV and emits a validated submission. Define supported artifact persistence/trust boundary during M1 planning; do not introduce arbitrary pickle loading into public inference. Update setuptools discovery/explicit package listing as needed because `backend/pyproject.toml` currently lists only `app`; do not silently leave the new subpackage out of installation.
4. **Focused tests under existing `backend/tests/`, with tiny temporary CSVs.** Missing/extra-contract columns, null handling, unsupported task/metric combinations, deterministic/leakage-safe splits, train-only preprocessing, prediction shape and row count/order, optional ID preservation, sample-submission column/order matching. Test consumer-visible failures, not implementation wiring. Run entirely offline. Validate one binary fixture and one regression or multiclass fixture end-to-end; compare deterministic outputs/run identities under the same config/seed. Do not assume every metric is ROC-AUC or every output is a probability.
5. **Usage documentation in existing setup docs after smoke proof.** Describe config→summary/validation→preprocess/split→train→evaluate→predict→submission checks, task/metric restrictions, artifact provenance and offline fixture commands. Generated run/model/submission outputs must follow the existing ignore policy without adding private data to Git.

### Explicit exclusions

Do **not** change `backend/app/main.py`, public schemas, existing release model directories, frontend behavior or current training/preparation/evaluation scripts in M1. No `src/` cutover; no retraining/recalibration of release detectors; no multilingual gate changes; no language detection/augmentation, robustness generator, RAG, experiment dashboard, ensemble or official adapter invented before its milestone gate. The generic path runs alongside the verified demo.

### Acceptance and review boundary

- One binary-classification fixture and one regression **or** multiclass fixture complete config → validation/exploratory summary → preprocessing → deterministic split → train → evaluate → predict → submission checks **offline**.
- Same input/config/seed yields reproducible split membership and prediction outputs; artifact/run record includes config, input hashes, split/seed, preprocessing/model identity, metric/result and software versions. Output row count/order/IDs and sample-submission column/order are validated; unsupported metrics fail clearly.
- Relevant new tests and existing backend suite pass; frontend production build remains valid. Smoke the real CLIs, not only mocks. Install/import the new package from a fresh editable environment to prove packaging includes it.
- Compare the same English HTTP smoke prompt against the unchanged release artifact before/after in a stable environment: seven response fields and probabilities unchanged. Document CUDA/CPU numerical environment when comparing; do not mask a difference by changing expectations.
- Review only intended M1 diff/staged paths. Focused commit/push only after approval and verification; preserve pre-existing dirty work. **Stop for user review before M2.**

## M2 — Reusable multilingual/low-resource preparation

**Entry gate:** reviewed M1 plus an official language/task need and lawful labeled data/quality checks. Existing machine-translated sw/ha/bn pilot alone does not justify broader performance claims (`backend/data/multilingual_manifest.json`, `backend/app/multilingual_guardrail.py`).

**Deliverables:** extend the generic pipeline with configurable language identification including unknown/uncertain, raw-preserving Unicode normalization/conservative noise cleanup, multilingual representation adapters and optional augmentation only after label-preservation validation. Record transformation parameters/model revisions/seeds and lineage. Reuse `backend/app/inference.py:normalize_input_text` behavior only where appropriate to the task, not as unconditional destructive preprocessing. Keep generation/translation provenance and raw originals separate.

**Acceptance:** deterministic Unicode/noise/code-switch fixtures, uncertain-language outcome, train/validation/test leakage checks and transformation logs; official per-language held-out evidence for any quality claim. Validate translation/augmentation labels before enabling them. Existing English/multilingual contracts stay intact. **Stop/review before M3.**

## M3 — Modular measurable security analysis

**Entry gate:** authorized labeled security target with agreed task taxonomy, threat scope and relevant subgroups; reviewed prior milestone. Do not invent poisoning labels from regex heuristics.

**Deliverables:** task adapter over M1/M2 for relevant prompt injection/adversarial input/data-quality or poisoning signals; structured score, predicted category, language estimate and component confidence with clear provenance. Reuse English detector/completion/heuristic components where relevant (`backend/app/inference.py`, `backend/app/signals.py`, `backend/app/schemas.py`), without changing their existing public contracts. Keep signals distinct from predictions and calibrated confidence distinct from uncalibrated scores.

**Acceptance:** held-out per-class/language/subgroup metrics, false-positive/error analysis and calibration for claimed probabilities, schema/error tests and real CLI smoke. Any new API integration requires separate contract approval. **Stop/review before M4.**

## M4 — Controlled robustness evaluation

**Entry gate:** approved authorized transformation set and datasets, clear label-preservation/synthetic labeling policy, reviewed baseline. No uncontrolled attack generation or transformation secretly applied to production input.

**Deliverables:** separate evaluation fixtures/runner over M1 predictions for benign paraphrase, translation, Unicode variation, spelling/spacing noise and code-switching where applicable. Store source lineage, transformation/version/seed and explicit synthetic labels; report clean/transformed performance by language and relevant subgroup. Reuse normalization/window regression evidence in `backend/tests/test_inference.py` without claiming it is a general robustness benchmark.

**Acceptance:** reproducible paired clean/transformed evaluation, documented exclusions/label uncertainty, subgroup deltas and artifact hashes, isolated fixture tests and real evaluation-run smoke. Production inference is unchanged by the test generator. **Stop/review before M5.**

## M5 — Conditional GenAI reliability and grounding

**Entry gate:** official generation **or** retrieval requirement, approved provider/network/credential/cost policy and evidence sources. If absent, record this milestone as not applicable after review rather than building speculative RAG.

**Deliverables:** smallest interchangeable interfaces for required retrieval/generation, evidence capture, response validation and justified confidence reporting; adapt the existing bounded transport/scanner patterns (`backend/app/redteam.py`, `backend/app/main.py`) only where they fit. Keep secrets out of code/logs. Build no full RAG system unless the actual task/useful demo requires it; no external model call before policy is understood.

**Acceptance:** real permitted provider/local-model smoke, reproducible evidence trace, supported/unsupported-answer checks, no fabricated citations/confidence and clear failure paths; costs/network limits documented. Fixture tests may cover failures but cannot stand in for real integration proof. **Stop/review before M6.**

## M6 — Experiment records/comparison and conditional ensembles

**Entry gate:** multiple validated pipeline runs with comparable split/metric definitions; official ensemble rules known before ensemble work. M1 run records are the prerequisite, not a second logging framework.

**Deliverables:** extend task-independent records/comparison with config, input/split hashes/seed, model/features, validation metrics, runtime, notes and compute/memory use. Compare simple baselines before expensive models; use the provenance concepts in `backend/app/training.py` without replacing schema-2/3 metadata. Add ensembling only if cross-validation supports improvement and rules permit it.

**Acceptance:** replay two comparable runs; report compute/memory and metric differences; held-out/CV-supported selection avoids test-set tuning. Optional ensemble predictions/submissions pass M1 shape/order checks and show justified improvement. **Stop/review before M7.**

## M7 — Existing demo dashboard over a stable pipeline API

**Entry gate:** approved stable pipeline API/output contracts and working CLI/submission path; reviewed prior outcomes. UI must not delay valid training/inference/submission.

**Deliverables:** reuse `frontend/src/App.jsx`, `frontend/src/components/Analyzer.jsx`, `frontend/src/components/ModelInfo.jsx` and `frontend/src/api.js`. Add only necessary task/config selection, prediction, truthful confidence/risk, language handling and evaluation evidence. Extract capability-focused orchestration boundaries only when needed. Coordinate deployment endpoint/origin configuration with actual deployment. Add adversarial/grounding views only for implemented, verified real outputs.

**Acceptance:** real browser verifies input→prediction, task/config selection, loading/errors, latest-request handling, unavailable capability states, responsive layout/accessibility and truthful calibration/evidence; build and API tests pass. Existing detector/output-scanner/red-team flows remain usable. **Stop/review before M8.**

## M8 — Official challenge adapter and runbook

**Entry gate:** official brief, rules, authorized dataset and sample-submission contract actually available. Re-check license, task type, target/features, metric, split rules, language needs and network/provider constraints. Do not infer these from the preparation theme.

**Deliverables:** proposed `docs/CHALLENGE_ADAPTER.md` and `docs/HACKATHON_RUNBOOK.md`, plus a task config/adapter extending established modules only where required. Document supported/unsupported requirements, concrete commands, environment/artifact provisioning, compute/network constraints, submission validation and truthful demo alignment.

**Acceptance:** follow the full day-of sequence from the handoff: (1) read official statement/rules; (2) identify task/target/metric/splits/schema/submission; (3) map modules and gaps; (4) run minimal baseline; (5) verify official metric/format locally; (6) inspect errors and improve one change at a time with recorded results; (7) generate/hash/verify final submission; (8) keep demo consistent with actual model behavior. Reproduce from documented environment/config and preserve raw data. No repository redesign just because the task is new. **Stop/review before further challenge-specific changes.**

## M0 verification/publication procedure

Run the existing backend suite and frontend build as audit evidence, not permission to fix unrelated failures. Record outcomes in `docs/CURRENT_ARCHITECTURE.md`; preserve failing evidence if present. Exercise the agent-owned HTTP service on 8001 because 8000 was already occupied; verify English health/seven fields and actual optional multilingual state. Docker/browser verification limits remain explicit.

Before publishing, verify existing-path citations and proposed-path labels, all component classifications and every M1–M8 outcome, documentation whitespace and baseline source hashes/status. Review the three-document diff and stage exactly:

```text
docs/CURRENT_ARCHITECTURE.md
docs/GAP_ANALYSIS.md
docs/MIGRATION_PLAN.md
```

Cached paths must contain only those files. Commit message: `docs: audit current hackathon architecture`. Push current `master` to its existing `origin` only using already configured authentication; never modify Git configuration/credentials or stage the handoff/application files. If commit fails, unstage these documents and report the blocker; if push fails after a successful commit, preserve that local focused commit and report the remote error. A docs-only commit intentionally leaves user changes dirty and may reference local/untracked files absent from a clean remote checkout.

## Assumptions / unverified and exact stop boundary

Official task fields, languages, labels, metric, split rules, submission format, available credentials/network/budget, dataset availability and ensemble/augmentation permission are unverified. M1 synthetic fixture support is not an official challenge adapter. No proposal above changes the current multilingual gate failure or establishes native-speaker performance. Proposed directories/entry points are not present and are not created during M0.

**M0 conclusion: propose M1 exactly as the additive backend/CLI-only configuration-driven CSV/baseline/evaluation/prediction/submission path above, then stop and wait for review. Do not create M1 directories or code; do not begin M1 in this run.**
