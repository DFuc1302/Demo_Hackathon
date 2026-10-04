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
class RedTeamCaseResponse(BaseModel):
    case_id: str
    title: str
    prompt: str


class RedTeamRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_ids: list[str] | None = Field(default=None, max_length=3)


class RedTeamResultResponse(BaseModel):
    case_id: str
    title: str
    status: Literal["pass", "fail", "error"]
    http_status: int | None
    response_excerpt: str | None
    error: str | None


class RedTeamSummary(BaseModel):
    pass_count: int
    fail_count: int
    error_count: int


class RedTeamRunResponse(BaseModel):
    results: list[RedTeamResultResponse]
    summary: RedTeamSummary
