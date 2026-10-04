from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.inference import MAX_PROMPT_LENGTH


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., max_length=MAX_PROMPT_LENGTH)


class AnalyzeResponse(BaseModel):
    label: Literal["benign", "jailbreak"]
    jailbreak_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    detected_signals: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
