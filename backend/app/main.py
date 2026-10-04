from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.inference import InferenceEngine
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

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

    return app


app = create_app()
