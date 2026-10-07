from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.inference import MAX_PROMPT_LENGTH


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., max_length=MAX_PROMPT_LENGTH)
    messages: list[dict[str, str]] | None = Field(default=None)

class AnalyzeResponse(BaseModel):
    label: Literal["benign", "jailbreak"]
    jailbreak_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    heuristic_signals: list[str]
    input_truncated: bool
    model_version: str
    calibrated: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class RiskThresholds(BaseModel):
    low: float
    high: float


class MetricScores(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    expected_calibration_error: float
    confusion_matrix: list[list[int]]


class ModelInfoResponse(BaseModel):
    model_version: str
    base_model: str
    base_model_revision: str
    dataset: str
    dataset_revision: str
    calibrated: bool
    classification_threshold: float
    risk_thresholds: RiskThresholds
    temperature: float
    metrics: dict[Literal["validation", "test"], MetricScores]


class RedTeamCaseResponse(BaseModel):
    case_id: str
    title: str
    prompt: str
    messages: list[dict[str, str]] | None = None
    category: str | None = None

class RedTeamRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_ids: list[str] | None = Field(default=None, max_length=10)

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


class ValidateCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completion: str = Field(..., max_length=MAX_PROMPT_LENGTH)
    system_prompt_snippets: list[str] | None = Field(default=None)


class ValidateCompletionResponse(BaseModel):
    safe: bool
    detected_secrets: list[str]
    system_prompt_leaks: list[str]
    adversarial_signals: list[str]
    jailbreak_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
class MultilingualAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., max_length=MAX_PROMPT_LENGTH)
    language: Literal["sw", "ha", "bn"]
    mode: Literal["multilingual", "translation", "compare"] = "compare"


class TranslatedAssessment(BaseModel):
    analysis: AnalyzeResponse
    translation_model: str
    translation_input_truncated: bool


class MultilingualAnalyzeResponse(BaseModel):
    language: Literal["sw", "ha", "bn"]
    mode: Literal["multilingual", "translation", "compare"]
    decision: Literal["benign", "jailbreak"]
    risk_level: Literal["low", "medium", "high"]
    warnings: list[str]
    multilingual_assessment: AnalyzeResponse | None = None
    translation_assessment: TranslatedAssessment | None = None


class MultilingualModelInfoResponse(BaseModel):
    available: bool
    supported_languages: list[str] = Field(default_factory=lambda: ["sw", "ha", "bn"])
    classifier_model: str | None = None
    classifier_revision: str | None = None
    translation_model: str | None = None
    translation_revision: str | None = None
    benchmark_kind: Literal["machine_translated"] = "machine_translated"
    metrics: dict[str, Any] = Field(default_factory=dict)
