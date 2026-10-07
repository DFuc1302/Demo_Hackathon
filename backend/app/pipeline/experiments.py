from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.pipeline.baseline import infer, metric_score
from app.pipeline.config import TaskConfig
from app.pipeline.core import load_artifact
from app.pipeline.data import Dataset, read_dataset


@dataclass(frozen=True)
class ExperimentSummary:
    run_id: str
    model_dir: str
    task_type: str
    metric: str
    score: float
    higher_is_better: bool
    model_kind: str
    seed: int
    train_rows: int
    validation_rows: int
    runtime_seconds: float
    text_column: str | None
    feature_columns: list[str]
    subword_ngrams: bool
    config_sha256: str
    model_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """Scan, index, and compare pipeline baseline run records."""

    def __init__(self, runs_dir: str | Path | None = None) -> None:
        self.runs_dir = Path(runs_dir).resolve() if runs_dir else None

    def scan_runs(self, base_dir: str | Path | None = None) -> list[ExperimentSummary]:
        target = Path(base_dir).resolve() if base_dir else self.runs_dir
        if not target or not target.exists():
            return []

        summaries: list[ExperimentSummary] = []
        # Find all artifact.json files under target directory
        for artifact_path in sorted(target.rglob("artifact.json")):
            try:
                artifact, config = load_artifact(artifact_path.parent)
                run = artifact["run"]
                summary = ExperimentSummary(
                    run_id=run["run_id"],
                    model_dir=str(artifact_path.parent),
                    task_type=config.task_type,
                    metric=config.metric,
                    score=float(run["score"]),
                    higher_is_better=run["higher_is_better"],
                    model_kind=artifact["model"]["kind"],
                    seed=config.seed,
                    train_rows=int(run["train_rows"]),
                    validation_rows=int(run["validation_rows"]),
                    runtime_seconds=round(float(run["runtime_seconds"]), 4),
                    text_column=config.text_column,
                    feature_columns=list(config.feature_columns),
                    subword_ngrams=config.subword_ngrams,
                    config_sha256=run["config_sha256"],
                    model_sha256=run["model_sha256"],
                )
                summaries.append(summary)
            except Exception:
                continue

        return summaries

    @staticmethod
    def compare_runs(model_dirs: list[str | Path]) -> dict[str, Any]:
        if not model_dirs:
            raise ValueError("model_dirs list cannot be empty")

        records: list[dict[str, Any]] = []
        for d in model_dirs:
            artifact, config = load_artifact(Path(d))
            run = artifact["run"]
            records.append({
                "model_dir": str(d),
                "run_id": run["run_id"],
                "model_kind": artifact["model"]["kind"],
                "task_type": config.task_type,
                "metric": config.metric,
                "score": round(float(run["score"]), 4),
                "higher_is_better": run["higher_is_better"],
                "runtime_seconds": round(float(run["runtime_seconds"]), 4),
                "seed": config.seed,
                "subword_ngrams": config.subword_ngrams,
                "feature_columns": list(config.feature_columns),
            })

        # Rank models
        higher_is_better = records[0]["higher_is_better"]
        ranked = sorted(records, key=lambda x: x["score"], reverse=higher_is_better)
        best = ranked[0]

        return {
            "runs": records,
            "best_run_id": best["run_id"],
            "best_model_dir": best["model_dir"],
            "best_score": best["score"],
            "metric": best["metric"],
            "ranking": [r["model_dir"] for r in ranked],
        }


class EnsembleModel:
    """Ensemble predictor and evaluator over multiple trained pipeline models."""

    def __init__(
        self,
        model_dirs: list[str | Path],
        weights: list[float] | None = None,
        method: Literal["average", "weighted", "majority_vote"] = "average",
    ) -> None:
        if len(model_dirs) < 2:
            raise ValueError("ensembling requires at least two trained member models")

        self.model_dirs = [Path(d).resolve() for d in model_dirs]
        self.method = method

        # Load and validate compatibility of members
        self.members: list[tuple[dict[str, Any], TaskConfig]] = []
        for d in self.model_dirs:
            art, cfg = load_artifact(d)
            self.members.append((art, cfg))

        base_task = self.members[0][1].task_type
        base_classes = self.members[0][0]["model"]["classes"]
        for _, cfg in self.members[1:]:
            if cfg.task_type != base_task:
                raise ValueError("all ensemble members must share the same task_type")
            if cfg.task_type != "regression" and cfg.positive_label != self.members[0][1].positive_label:
                raise ValueError("classification ensemble members must share positive_label")

        self.task_type = base_task
        self.classes = base_classes

        # Normalize weights
        if weights is not None:
            if len(weights) != len(self.model_dirs):
                raise ValueError("weights length must match number of model directories")
            total = sum(weights)
            if total <= 0:
                raise ValueError("sum of weights must be positive")
            self.weights = [w / total for w in weights]
        else:
            n = len(self.model_dirs)
            self.weights = [1.0 / n] * n

    def predict(self, dataset: Dataset) -> tuple[np.ndarray, np.ndarray | None]:
        member_preds: list[np.ndarray] = []
        member_probs: list[np.ndarray | None] = []

        for (art, cfg) in self.members:
            preds, probs = infer(dataset, cfg, art["preprocessing"], art["model"])
            member_preds.append(preds)
            member_probs.append(probs)

        n_samples = len(dataset.rows)

        if self.task_type == "regression":
            # Weighted average of continuous outputs
            weighted_scores = np.zeros(n_samples, dtype=float)
            for preds, w in zip(member_preds, self.weights):
                weighted_scores += preds * w
            return weighted_scores, None

        # Classification
        if self.method == "majority_vote":
            # Discrete vote per row
            final_labels = []
            for i in range(n_samples):
                row_votes = [preds[i] for preds in member_preds]
                winner = Counter(row_votes).most_common(1)[0][0]
                final_labels.append(winner)
            return np.asarray(final_labels), None

        # Soft probability pooling (average or weighted)
        n_classes = len(self.classes)
        pooled_probs = np.zeros((n_samples, n_classes), dtype=float)

        for probs, w in zip(member_probs, self.weights):
            if probs is not None:
                pooled_probs += probs * w

        final_labels = np.asarray(self.classes)[np.argmax(pooled_probs, axis=1)]
        return final_labels, pooled_probs

    def evaluate(self, dataset_path: str | Path) -> dict[str, Any]:
        """Evaluate ensemble and determine if it offers justified improvement over single members."""
        first_cfg = self.members[0][1]
        dataset = read_dataset(dataset_path, first_cfg, labeled=True)

        member_scores: list[float] = []
        for (art, cfg) in self.members:
            preds, probs = infer(dataset, cfg, art["preprocessing"], art["model"])
            score = metric_score(dataset, cfg, art["model"], preds, probs)
            member_scores.append(round(score, 4))

        ens_preds, ens_probs = self.predict(dataset)
        ens_score = metric_score(dataset, first_cfg, self.members[0][0]["model"], ens_preds, ens_probs)
        ens_score = round(ens_score, 4)

        higher_is_better = first_cfg.metric not in ("rmse", "mae")
        best_member = max(member_scores) if higher_is_better else min(member_scores)

        # Justified ensembling invariant: ensemble must match or exceed best individual member
        is_justified = (ens_score >= best_member) if higher_is_better else (ens_score <= best_member)

        return {
            "ensemble_metric": first_cfg.metric,
            "ensemble_score": ens_score,
            "member_scores": member_scores,
            "best_member_score": best_member,
            "justified_improvement": is_justified,
            "method": self.method,
            "member_count": len(self.members),
        }
