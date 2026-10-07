from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.rate_limiter import TokenBucketRateLimiter

from app.inference import InferenceEngine
from app.redteam import RED_TEAM_CASES, configured_api_key, configured_endpoint, result_summary, run_evaluation
from app.schemas import (AnalyzeRequest, AnalyzeResponse, HealthResponse, ModelInfoResponse, RedTeamCaseResponse,
                         RedTeamResultResponse, RedTeamRunRequest, RedTeamRunResponse, RedTeamSummary,
                         ValidateCompletionRequest, ValidateCompletionResponse)
from app.signals import detect_secrets
from app.inference import MultilingualInferenceEngine
from app.translation import TranslationEngine
from app.multilingual_guardrail import MultilingualGuardrailService
from app.schemas import (
    MultilingualAnalyzeRequest,
    MultilingualAnalyzeResponse,
    MultilingualModelInfoResponse,
    TranslatedAssessment,
)

DEFAULT_MODEL_PATH = Path(__file__).parents[1] / "models" / "jailbreak_transformer"
DEFAULT_MULTILINGUAL_MODEL_PATH = Path(__file__).parents[1] / "models" / "multilingual_jailbreak_transformer"
DEFAULT_TRANSLATION_MODEL_PATH = Path(__file__).parents[1] / "models" / "translator" / "m2m100_418M"
LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
if os.getenv("FRONTEND_URL"):
    LOCAL_FRONTEND_ORIGINS.append(os.getenv("FRONTEND_URL"))

MAX_REQUEST_BODY_SIZE = 65_536  # 64 KB


def _apply_security_headers(response: Response) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"

def _public_model_info(engine: InferenceEngine) -> ModelInfoResponse:
    metadata = engine.metadata
    return ModelInfoResponse(model_version=metadata["model_version"], base_model=metadata["model_id"], base_model_revision=metadata["model_revision"], dataset=metadata["dataset_id"], dataset_revision=metadata["dataset_revision"], calibrated=True, classification_threshold=metadata["classification_threshold"], risk_thresholds=metadata["risk_thresholds"], temperature=metadata["calibration_temperature"], metrics=metadata["metrics"])


def create_app(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    multilingual_model_path: str | Path = DEFAULT_MULTILINGUAL_MODEL_PATH,
    translation_model_path: str | Path = DEFAULT_TRANSLATION_MODEL_PATH,
) -> FastAPI:
    queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
    worker_task: asyncio.Task | None = None
    multilingual_semaphore = asyncio.Semaphore(2)
    async def batch_worker(engine: InferenceEngine) -> None:
        while True:
            try:
                prompt, future = await queue.get()
                batch = [(prompt, future)]
                start_time = asyncio.get_event_loop().time()
                while len(batch) < 8:
                    remaining = 0.05 - (asyncio.get_event_loop().time() - start_time)
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                prompts = [p for p, _ in batch]
                futures = [f for _, f in batch]
                try:
                    results = engine.analyze_batch(prompts)
                    for f, res in zip(futures, results):
                        if not f.done():
                            f.set_result(res)
                except Exception as exc:
                    for f in futures:
                        if not f.done():
                            f.set_exception(exc)
                finally:
                    for _ in batch:
                        queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.01)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal worker_task
        try:
            app.state.inference_engine = InferenceEngine(model_path)
            app.state.batch_queue = queue
            worker_task = asyncio.create_task(batch_worker(app.state.inference_engine))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"could not initialize inference model: {exc}") from exc

        # Initialize MultilingualGuardrailService if artifacts exist and validate
        app.state.multilingual_service = None
        app.state.multilingual_available = False
        try:
            m_path = Path(multilingual_model_path)
            t_path = Path(translation_model_path)
            if m_path.exists() and t_path.exists():
                multi_eng = MultilingualInferenceEngine(m_path)
                trans_eng = TranslationEngine(t_path)
                app.state.multilingual_service = MultilingualGuardrailService(
                    multilingual_engine=multi_eng,
                    translation_engine=trans_eng,
                    english_engine=app.state.inference_engine,
                )
                app.state.multilingual_available = True
        except Exception:
            app.state.multilingual_service = None
            app.state.multilingual_available = False

        try:
            yield
        finally:
            if worker_task:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
    app = FastAPI(title="AI Security Demo API", version="0.2.0", lifespan=lifespan)
    app.state.rate_limiter = TokenBucketRateLimiter()

    @app.middleware("http")
    async def security_and_limits_middleware(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_REQUEST_BODY_SIZE:
                    resp = JSONResponse(
                        status_code=413,
                        content={"detail": "Payload too large: request body exceeds 64KB limit."},
                    )
                    _apply_security_headers(resp)
                    return resp
            except ValueError:
                resp = JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
                _apply_security_headers(resp)
                return resp
        elif request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if len(body) > MAX_REQUEST_BODY_SIZE:
                resp = JSONResponse(
                    status_code=413,
                    content={"detail": "Payload too large: request body exceeds 64KB limit."},
                )
                _apply_security_headers(resp)
                return resp

        if request.method == "POST" and request.url.path in ("/api/analyze", "/api/analyze-multilingual"):
            client_ip = request.client.host if request.client else "127.0.0.1"
            limiter: TokenBucketRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
            if limiter is not None:
                allowed, retry_after = limiter.check_rate_limit(client_ip)
                if not allowed:
                    resp = JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded. Too many requests."},
                        headers={"Retry-After": str(retry_after)},
                    )
                    _apply_security_headers(resp)
                    return resp

        response = await call_next(request)
        _apply_security_headers(response)
        return response

    app.add_middleware(CORSMiddleware, allow_origins=LOCAL_FRONTEND_ORIGINS, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        engine = request.app.state.inference_engine
        return HealthResponse(status="ok", model_loaded=True, model_version=engine.metadata["model_version"])

    @app.get("/api/model-info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        return _public_model_info(request.app.state.inference_engine)

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(request: AnalyzeRequest, http_request: Request) -> AnalyzeResponse:
        queue = getattr(http_request.app.state, "batch_queue", None)
        if queue is not None:
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            await queue.put((request.prompt, future))
            try:
                result = await future
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return AnalyzeResponse(**result.__dict__)
        try:
            result = http_request.app.state.inference_engine.analyze(request.prompt)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AnalyzeResponse(**result.__dict__)
    @app.get("/api/multilingual/model-info", response_model=MultilingualModelInfoResponse)
    async def multilingual_model_info(request: Request) -> MultilingualModelInfoResponse:
        service: MultilingualGuardrailService | None = getattr(request.app.state, "multilingual_service", None)
        if not request.app.state.multilingual_available or service is None:
            return MultilingualModelInfoResponse(available=False)

        meta = service.multilingual_engine.metadata
        return MultilingualModelInfoResponse(
            available=True,
            supported_languages=["sw", "ha", "bn"],
            classifier_model=meta["model_id"],
            classifier_revision=meta["model_revision"],
            translation_model="facebook/m2m100_418M",
            translation_revision="55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636",
            benchmark_kind="machine_translated",
            metrics=meta.get("metrics", {}),
        )

    @app.post("/api/analyze-multilingual", response_model=MultilingualAnalyzeResponse)
    async def analyze_multilingual(
        request: MultilingualAnalyzeRequest, http_request: Request
    ) -> MultilingualAnalyzeResponse:
        service: MultilingualGuardrailService | None = getattr(http_request.app.state, "multilingual_service", None)
        if not http_request.app.state.multilingual_available or service is None:
            raise HTTPException(
                status_code=503,
                detail="Multilingual model and translation artifacts are not installed.",
            )

        try:
            async with multilingual_semaphore:
                # Offload compute outside event-loop thread
                res = await asyncio.to_thread(
                    service.analyze,
                    request.prompt,
                    language=request.language,
                    mode=request.mode,
                )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        multi_ass = None
        if res.multilingual_assessment is not None:
            multi_ass = AnalyzeResponse(**res.multilingual_assessment.__dict__)

        trans_ass = None
        if res.translation_assessment is not None:
            trans_ass = TranslatedAssessment(
                analysis=AnalyzeResponse(**res.translation_assessment.analysis.__dict__),
                translation_model=res.translation_assessment.translation_model,
                translation_input_truncated=res.translation_assessment.translation_input_truncated,
            )

        return MultilingualAnalyzeResponse(
            language=res.language,  # type: ignore
            mode=res.mode,          # type: ignore
            decision=res.decision,  # type: ignore
            risk_level=res.risk_level,  # type: ignore
            warnings=res.warnings,
            multilingual_assessment=multi_ass,
            translation_assessment=trans_ass,
        )

    @app.get("/api/redteam/cases", response_model=list[RedTeamCaseResponse])
    async def redteam_cases() -> list[RedTeamCaseResponse]:
        return [RedTeamCaseResponse(**case.__dict__) for case in RED_TEAM_CASES]

    @app.post("/api/redteam/run", response_model=RedTeamRunResponse)
    def redteam_run(request: RedTeamRunRequest) -> RedTeamRunResponse:
        endpoint = configured_endpoint()
        if not endpoint:
            raise HTTPException(status_code=503, detail="REDTEAM_LLM_URL is not configured")
        try:
            results = run_evaluation(endpoint, request.case_ids, api_key=configured_api_key())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        summary = result_summary(results)
        return RedTeamRunResponse(results=[RedTeamResultResponse(**result.__dict__) for result in results], summary=RedTeamSummary(pass_count=summary["pass"], fail_count=summary["fail"], error_count=summary["error"]))

    @app.post("/api/guardrail/validate-completion", response_model=ValidateCompletionResponse)
    def validate_completion(request: ValidateCompletionRequest, http_request: Request) -> ValidateCompletionResponse:
        engine: InferenceEngine = http_request.app.state.inference_engine
        detected_secrets = detect_secrets(request.completion)

        system_prompt_leaks = []
        if request.system_prompt_snippets:
            completion_lower = request.completion.lower()
            for snippet in request.system_prompt_snippets:
                cleaned_snippet = snippet.strip()
                if cleaned_snippet and cleaned_snippet.lower() in completion_lower:
                    system_prompt_leaks.append(snippet)

        try:
            analysis = engine.analyze(request.completion)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        adversarial_signals = analysis.heuristic_signals
        is_safe = (
            len(detected_secrets) == 0
            and len(system_prompt_leaks) == 0
            and len(adversarial_signals) == 0
            and analysis.risk_level != "high"
        )

        return ValidateCompletionResponse(
            safe=is_safe,
            detected_secrets=detected_secrets,
            system_prompt_leaks=system_prompt_leaks,
            adversarial_signals=analysis.heuristic_signals,
            jailbreak_probability=analysis.jailbreak_probability,
            risk_level=analysis.risk_level,
        )

    return app


app = create_app()
