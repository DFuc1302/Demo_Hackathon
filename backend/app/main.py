from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.inference import InferenceEngine
from app.redteam import RED_TEAM_CASES, configured_api_key, configured_endpoint, result_summary, run_evaluation
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    RedTeamCaseResponse,
    RedTeamResultResponse,
    RedTeamRunRequest,
    RedTeamRunResponse,
    RedTeamSummary,
)

DEFAULT_MODEL_PATH = Path(__file__).parents[1] / "models" / "jailbreak_classifier.joblib"
LOCAL_FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app(model_path: str | Path = DEFAULT_MODEL_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.inference_engine = InferenceEngine(model_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"could not initialize inference model: {exc}") from exc
        yield

    app = FastAPI(
        title="AI Security Demo API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=LOCAL_FRONTEND_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=hasattr(request.app.state, "inference_engine"),
        )

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(request: AnalyzeRequest, http_request: Request) -> AnalyzeResponse:
        try:
            result = http_request.app.state.inference_engine.analyze(request.prompt)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AnalyzeResponse(**result.__dict__)

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
        return RedTeamRunResponse(
            results=[RedTeamResultResponse(**result.__dict__) for result in results],
            summary=RedTeamSummary(
                pass_count=summary["pass"],
                fail_count=summary["fail"],
                error_count=summary["error"],
            ),
        )

    return app


app = create_app()
