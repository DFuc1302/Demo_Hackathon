from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from app.dataset_contract import (
    MULTILINGUAL_LANGUAGES,
    check_multilingual_cross_split_leakage,
    sha256_bytes,
    validate_multilingual_split_rows,
)
from app.model_metadata import validate_model_metadata
from app.multilingual_manifest import (
    CLASSIFIER_MODEL_ID,
    CLASSIFIER_MODEL_REVISION,
    SPLITS,
    validate_multilingual_manifest,
)
from app.translation_artifact import validate_translation_artifact

SCHEMA_VERSION = 3
PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "multilingual"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "multilingual_manifest.json"
DEFAULT_RELEASE_MODEL_DIR = PROJECT_ROOT / "models" / "multilingual_jailbreak_transformer"
DEFAULT_TRANSLATION_MODEL_PATH = PROJECT_ROOT / "models" / "translator" / "m2m100_418M"

MAX_TOKEN_LENGTH = 256
CLASSIFICATION_THRESHOLD = 0.50
RISK_THRESHOLDS = {"low": 0.20, "high": 0.80}


@dataclass(frozen=True)
class MultilingualTrainingResult:
    model_dir: Path
    sample_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    metrics: dict[str, Any]
    temperatures: dict[str, float]
    model_version: str


def _set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (
            (probabilities >= edges[index])
            & (
                (probabilities < edges[index + 1])
                if index < bins - 1
                else (probabilities <= edges[index + 1])
            )
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return result


def _calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | list[list[int]]]:
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": _ece(labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=100, line_search_fn="strong_wolfe"
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_function(
            (values[:, 1] - values[:, 0]) / torch.exp(log_temperature), targets
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp_min(1e-3).item())


def load_multilingual_splits(
    data_dir: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, Any], DatasetDict]:
    data_dir = Path(data_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = {}
    validated_splits = {}
    for split_name in SPLITS:
        path = data_dir / f"{split_name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"split file not found: {path}")
        content = path.read_bytes()
        expected = manifest["splits"][split_name]
        if sha256_bytes(content) != expected["prepared_sha256"]:
            raise ValueError(f"split hash mismatch for {split_name}")
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        validated_rows = validate_multilingual_split_rows(rows, split_name, expected)
        validated_splits[split_name] = validated_rows
        files[split_name] = str(path)

    check_multilingual_cross_split_leakage(validated_splits)
    raw_dataset = load_dataset("json", data_files=files)
    return manifest, raw_dataset


def train_multilingual_model(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    model_dir: str | Path = DEFAULT_RELEASE_MODEL_DIR,
    translation_model_path: str | Path = DEFAULT_TRANSLATION_MODEL_PATH,
    smoke: bool = False,
    seed: int = 42,
) -> MultilingualTrainingResult:
    target_dir = Path(model_dir)
    if smoke and target_dir.resolve() == DEFAULT_RELEASE_MODEL_DIR.resolve():
        raise ValueError("smoke training cannot output directly to the release model directory")

    _set_determinism(seed)
    manifest, dataset = load_multilingual_splits(data_dir, manifest_path)

    # Use pinned revision
    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_MODEL_ID,
        revision=CLASSIFIER_MODEL_REVISION,
        use_fast=True,
    )

    if smoke:
        # Balanced slice per language
        train_rows = dataset["train"].to_list()
        smoke_train = []
        for lang in MULTILINGUAL_LANGUAGES:
            l_rows = [r for r in train_rows if r["language"] == lang]
            for lbl in (0, 1):
                smoke_train.extend([r for r in l_rows if r["label"] == lbl][:2])
        train_ds = Dataset.from_list(smoke_train)
        val_ds = train_ds
        test_ds = train_ds
        epochs = 1
    else:
        train_ds = dataset["train"]
        val_ds = dataset["validation"]
        test_ds = dataset["test"]
        epochs = 2

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            max_length=MAX_TOKEN_LENGTH,
            truncation=True,
            padding=False,
        )

    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "source_id"])
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["text", "source_id"])
    tokenized_test = test_ds.map(tokenize_fn, batched=True, remove_columns=["text", "source_id"])

    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_ID,
        revision=CLASSIFIER_MODEL_REVISION,
        num_labels=2,
        id2label={0: "benign", 1: "jailbreak"},
        label2id={"benign": 0, "jailbreak": 1},
    )

    work_dir = target_dir if not target_dir.exists() else target_dir.parent / f".tmp_{target_dir.name}"
    training_args = TrainingArguments(
        output_dir=str(work_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16 if torch.cuda.is_available() else 4,
        per_device_eval_batch_size=16 if torch.cuda.is_available() else 4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    # Per-language calibration temperature fitting on validation split
    temperatures: dict[str, float] = {}
    val_raw = val_ds.to_list()
    val_preds = trainer.predict(tokenized_val)
    val_logits = np.asarray(val_preds.predictions)
    val_labels = np.asarray(val_ds["label"])

    for lang in MULTILINGUAL_LANGUAGES:
        indices = [i for i, r in enumerate(val_raw) if r["language"] == lang]
        l_logits = val_logits[indices]
        l_labels = val_labels[indices]
        temperatures[lang] = _fit_temperature(l_logits, l_labels)

    # Compute validation and test metrics per language
    metrics: dict[str, Any] = {"multilingual": {"validation": {}, "test": {}}}
    test_raw = test_ds.to_list()
    test_preds = trainer.predict(tokenized_test)
    test_logits = np.asarray(test_preds.predictions)
    test_labels = np.asarray(test_ds["label"])

    for lang in MULTILINGUAL_LANGUAGES:
        # validation metrics
        val_idx = [i for i, r in enumerate(val_raw) if r["language"] == lang]
        t = temperatures[lang]
        val_l_probs = torch.softmax(torch.tensor(val_logits[val_idx]) / t, dim=-1)[:, 1].numpy()
        metrics["multilingual"]["validation"][lang] = _calculate_metrics(val_labels[val_idx], val_l_probs)

        # test metrics
        test_idx = [i for i, r in enumerate(test_raw) if r["language"] == lang]
        test_l_probs = torch.softmax(torch.tensor(test_logits[test_idx]) / t, dim=-1)[:, 1].numpy()
        metrics["multilingual"]["test"][lang] = _calculate_metrics(test_labels[test_idx], test_l_probs)

    # Enforce release gate on non-smoke runs: test F1 >= 0.90, recall >= 0.90, Brier <= 0.15 for EVERY language
    if not smoke:
        for lang in MULTILINGUAL_LANGUAGES:
            lm = metrics["multilingual"]["test"][lang]
            if lm["f1"] < 0.90 or lm["recall"] < 0.90 or lm["brier_score"] > 0.15:
                raise ValueError(
                    f"release gate failed for {lang}: F1={lm['f1']}, recall={lm['recall']}, brier={lm['brier_score']}"
                )

    # Save artifact
    model_version = f"v3-{CLASSIFIER_MODEL_ID.rsplit('/', 1)[-1]}-{CLASSIFIER_MODEL_REVISION[:12]}"
    sample_counts = {s: len(dataset[s]) for s in SPLITS}
    class_counts = {
        s: {
            str(lbl): sum(r["label"] == lbl for r in dataset[s])
            for lbl in (0, 1)
        }
        for s in SPLITS
    }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "model_id": CLASSIFIER_MODEL_ID,
        "model_revision": CLASSIFIER_MODEL_REVISION,
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "dataset_split_hashes": {s: manifest["splits"][s]["prepared_sha256"] for s in SPLITS},
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
        },
        "label_mapping": {"0": "benign", "1": "jailbreak"},
        "max_token_length": MAX_TOKEN_LENGTH,
        "languages": list(MULTILINGUAL_LANGUAGES),
        "calibration_temperatures": temperatures,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "risk_thresholds": RISK_THRESHOLDS,
        "seed": seed,
        "sample_counts": sample_counts,
        "class_counts": class_counts,
        "translation_artifact": manifest["translation_artifact"],
        "metrics": metrics,
    }

    staging_dir = work_dir
    staging_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging_dir, safe_serialization=True)
    tokenizer.save_pretrained(staging_dir)
    (staging_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Validate saved artifact before cutover
    validate_model_metadata(metadata, staging_dir)
    from app.inference import MultilingualInferenceEngine
    _ = MultilingualInferenceEngine(staging_dir)

    if staging_dir != target_dir:
        if target_dir.exists():
            backup_dir = target_dir.parent / f".prev_{target_dir.name}"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            target_dir.rename(backup_dir)
        staging_dir.rename(target_dir)

    return MultilingualTrainingResult(
        model_dir=target_dir,
        sample_counts=sample_counts,
        class_counts=class_counts,
        metrics=metrics,
        temperatures=temperatures,
        model_version=model_version,
    )
