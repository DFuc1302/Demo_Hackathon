from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.pipeline.config import METRICS
from app.pipeline.experiments import ExperimentRegistry
from app.pipeline.language import identify_language
from app.pipeline.robustness import SUPPORTED_TRANSFORMATIONS, TRANSFORMATION_REGISTRY
from app.pipeline.security import (
    DEFAULT_RISK_THRESHOLDS,
    SECURITY_CATEGORIES,
    analyze_security_prompt,
)

def create_pipeline_app() -> Any:
    """Create dedicated pipeline FastAPI service application."""
    from fastapi import FastAPI
    app = FastAPI(title="Pipeline Demo API", version="1.0.0")
    app.include_router(pipeline_router, prefix="/api/pipeline")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "pipeline"}

    return app


pipeline_router = APIRouter()


class PipelineAnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    task_type: str = Field(default="security")
    model_dir: str | None = Field(default=None)


class RobustnessCheckRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    transformations: list[str] | None = Field(default=None)


@pipeline_router.get("/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """Expose full pipeline capability registry across M1-M6."""
    return {
        "status": "ready",
        "pipeline_version": "v1.0.0",
        "supported_tasks": {
            task: sorted(metrics) for task, metrics in METRICS.items()
        },
        "target_languages": ["en", "sw", "ha", "bn"],
        "language_scripts": ["Latin", "Bengali", "Arabic"],
        "normalization_forms": ["NFKC", "NFC", "NFD", "NFKD"],
        "security_categories": list(SECURITY_CATEGORIES),
        "risk_thresholds": DEFAULT_RISK_THRESHOLDS,
        "robustness_transformations": list(SUPPORTED_TRANSFORMATIONS),
        "ensembling_methods": ["average", "weighted", "majority_vote"],
    }


@pipeline_router.post("/analyze")
async def pipeline_analyze(request: PipelineAnalyzeRequest) -> dict[str, Any]:
    """Analyze prompt through security and language pipeline."""
    try:
        model_path = Path(request.model_dir).resolve() if request.model_dir else None
        res = analyze_security_prompt(request.prompt, model_dir=model_path)
        return res.to_dict()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@pipeline_router.get("/runs")
async def list_pipeline_runs(request: Request) -> list[dict[str, Any]]:
    """List indexed experiment runs from outputs directory."""
    backend_root = Path(__file__).resolve().parents[2]
    outputs_dir = backend_root / "outputs"
    registry = ExperimentRegistry(outputs_dir)
    runs = registry.scan_runs()
    return [r.to_dict() for r in runs]


@pipeline_router.post("/robustness-check")
async def robustness_check(request: RobustnessCheckRequest) -> dict[str, Any]:
    """Generate transformed variations of a prompt for interactive robustness inspection."""
    transforms = request.transformations or list(SUPPORTED_TRANSFORMATIONS)
    variations: dict[str, str] = {}
    for t_name in transforms:
        if t_name in TRANSFORMATION_REGISTRY:
            fn = TRANSFORMATION_REGISTRY[t_name]
            variations[t_name] = fn(request.prompt, seed=42)

    lang_res = identify_language(request.prompt)
    return {
        "original_prompt": request.prompt,
        "language_estimate": {
            "language": lang_res.language,
            "confidence": lang_res.confidence,
            "script": lang_res.script,
            "is_uncertain": lang_res.is_uncertain,
        },
        "transformed_variations": variations,
    }
