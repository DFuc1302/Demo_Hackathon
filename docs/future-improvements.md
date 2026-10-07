# Future Improvements & Optimizations

This document outlines high-priority engineering improvements and optimizations identified following the V2 upgrade and production fine-tuning.

---

## 1. Inference Performance & Noise Reduction (Completed)
- **FP16 Half-Precision on CUDA**:
  `backend/app/inference.py` converts the model to FP16 after moving it to CUDA; CPU inference remains FP32. FP16 halves parameter storage relative to FP32; total VRAM and latency improvements depend on workload and have not been benchmarked.
- **Preserve Trained DeBERTa Tokenization**:
  Inference and training load fast tokenizers with `fix_mistral_regex=False`. Transformers 4.57.6 can misidentify this large-vocabulary local DeBERTa artifact as Mistral; forcing the patch replaces its trained Metaspace pre-tokenizer with a Split regex and causes a severe quality regression. Explicitly disabling the patch preserves trained tokenization and avoids the false-positive regex warning. The unrelated upstream `torch.jit.script` deprecation warning remains.
- **Verification**:
  A saved-tokenization regression failed before the correction and passed afterward. Inference tests passed all 37 cases, including CUDA FP16 and CPU FP32 coverage; the full backend suite passed all 78 tests. Fresh single-prompt CUDA FP16 evaluation covered every validation and test row (1,000 per split) with unchanged release temperature and thresholds: validation F1 0.992979 / Brier 0.006180; test F1 0.993021 / recall 0.996 / Brier 0.005185 / ECE 0.003211. All release gates passed. Dataset and model artifact hashes were unchanged; no retraining or recalibration was performed.

---

## 2. Red-Team Harness Concurrency (Completed)
- **Concurrent Guardrail Evaluation**:
  In `backend/app/redteam.py::run_evaluation`, evaluation runs concurrently across cases using Python's `concurrent.futures.ThreadPoolExecutor(max_workers=3)`.
- **Verification**:
  `test_evaluation_runs_concurrently_in_requested_order` verifies concurrent execution using a `ThreadingHTTPServer` and `threading.Barrier(3)` rendezvous while asserting output preservation and case ordering.

---

## 3. Documentation Alignment (Completed)
- **Updated Legacy References**:
  `docs/demo-guide.md` and `README.md` reflect the active, verified, clean dataset (`dfuc1302/demo_hackathon`, revision `d17a4cf856bacd17d799e8a5f2955b0082f75e75`) and record the fresh 2,000-row held-out parity evaluation results ($F_1 = 0.993$, Recall $= 0.996$, Brier $= 0.0052$).

## 4. Repository Cleanliness (Completed)
- **Ignore Root Lockfile**:
  `/package-lock.json` is added to `.gitignore` so accidental root-level `npm` invocations do not stage an untracked lockfile at the repo root.
