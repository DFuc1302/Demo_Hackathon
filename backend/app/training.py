from __future__ import annotations

import json
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from app.dataset_contract import (check_cross_split_leakage, sha256_bytes,
                                  validate_split_rows)
from app.inference import InferenceEngine
import datasets
from datasets import DatasetDict, load_dataset
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
import transformers
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).parents[1]
RELEASE_MODEL_DIR = PROJECT_ROOT / "models" / "jailbreak_transformer"
MODEL_ID = "microsoft/deberta-v3-small"
MAX_TOKEN_LENGTH = 256
CLASSIFICATION_THRESHOLD = 0.50
RISK_THRESHOLDS = {"low": 0.20, "high": 0.80}

@dataclass(frozen=True)
class TrainingResult:
    model_dir: Path
    sample_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    metrics: dict[str, dict[str, float | list[list[int]]]]
    temperature: float
    model_version: str


def _manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("model_id") != MODEL_ID:
        raise ValueError("dataset manifest must use schema 2 and the pinned DeBERTa model")
    return manifest

def load_dataset_splits(dataset_dir: str | Path, manifest_path: str | Path) -> DatasetDict:
    dataset_dir = Path(dataset_dir)
    manifest = _manifest(manifest_path)
    splits: dict[str, list[dict[str, int | str]]] = {}
    files: dict[str, str] = {}
    for split, expected in manifest["splits"].items():
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"prepared dataset split not found: {path}; run prepare_v2_dataset.py")
        content = path.read_bytes()
        if sha256_bytes(content) != expected.get("prepared_sha256"):
            raise ValueError(f"prepared {split} hash does not match manifest: {path}")
        try:
            rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"prepared {split} contains invalid JSON: {path}") from exc
        validated = validate_split_rows(rows, split, expected)
        splits[split] = validated
        files[split] = str(path)
    check_cross_split_leakage(splits)
    dataset = load_dataset("json", data_files=files)
    if set(dataset) != {"train", "validation", "test"} or set(dataset["train"].column_names) != {"text", "label"}:
        raise ValueError("prepared dataset must expose exactly train, validation, and test text/label splits")
    return dataset

def _set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _balanced_subset(dataset, count: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = dataset.to_list()
    selected = []
    for label in (0, 1):
        candidates = [row for row in rows if row["label"] == label]
        if len(candidates) < count // 2:
            raise ValueError("smoke training requires enough examples in each class")
        selected.extend(candidates[index] for index in rng.choice(len(candidates), count // 2, replace=False))
    rng.shuffle(selected)
    from datasets import Dataset
    return Dataset.from_list(selected)


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & ((probabilities < edges[index + 1]) if index < bins - 1 else (probabilities <= edges[index + 1]))
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | list[list[int]]]:
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    return {"accuracy": float(accuracy_score(labels, predictions)), "precision": float(precision_score(labels, predictions, zero_division=0)), "recall": float(recall_score(labels, predictions, zero_division=0)), "f1": float(f1_score(labels, predictions, zero_division=0)), "roc_auc": float(roc_auc_score(labels, probabilities)), "pr_auc": float(average_precision_score(labels, probabilities)), "brier_score": float(brier_score_loss(labels, probabilities)), "expected_calibration_error": _ece(labels, probabilities), "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100, line_search_fn="strong_wolfe")
    loss_function = torch.nn.BCEWithLogitsLoss()
    def closure():
        optimizer.zero_grad()
        loss = loss_function((values[:, 1] - values[:, 0]) / torch.exp(log_temperature), targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp_min(1e-3).item())


def _predict_logits(trainer: Trainer, dataset) -> tuple[np.ndarray, np.ndarray]:
    output = trainer.predict(dataset)
    return np.asarray(output.predictions), np.asarray(dataset["label"], dtype=np.int64)


def _metadata(manifest: dict, temperature: float, metrics: dict, sample_counts: dict, class_counts: dict, seed: int) -> dict:
    revision = manifest["model_revision"]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": f"v2-{MODEL_ID.rsplit('/', 1)[-1]}-{revision[:12]}",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "dataset_split_hashes": {split: value["source_sha256"] for split, value in manifest["splits"].items()},
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
        },
        "label_mapping": {"0": "benign", "1": "jailbreak"},
        "max_token_length": MAX_TOKEN_LENGTH,
        "calibration_temperature": temperature,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "risk_thresholds": RISK_THRESHOLDS,
        "seed": seed,
        "sample_counts": sample_counts,
        "class_counts": class_counts,
        "metrics": metrics,
    }

def train_transformer(dataset_dir: str | Path, output_dir: str | Path, *, manifest_path: str | Path, seed: int = 42, smoke: bool = False) -> TrainingResult:
    destination = Path(output_dir)
    if smoke and destination.resolve() == RELEASE_MODEL_DIR.resolve():
        raise ValueError("smoke training cannot write to the release model directory")
    _set_determinism(seed)
    manifest = _manifest(manifest_path)
    dataset = load_dataset_splits(dataset_dir, manifest_path)
    if smoke:
        dataset = DatasetDict({"train": _balanced_subset(dataset["train"], 64, seed), "validation": _balanced_subset(dataset["validation"], 32, seed), "test": dataset["test"]})
    sample_counts = {split: len(dataset[split]) for split in dataset}
    class_counts = {split: {str(label): sum(int(value) == label for value in dataset[split]["label"]) for label in (0, 1)} for split in dataset}
    revision = manifest["model_revision"]
    # DeBERTa uses its original pre-tokenizer, not Mistral's regex patch.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, use_fast=True, fix_mistral_regex=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=revision, num_labels=2, id2label={0: "benign", 1: "jailbreak"}, label2id={"benign": 0, "jailbreak": 1})
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_TOKEN_LENGTH)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    args = TrainingArguments(output_dir=str(Path(output_dir).with_name(Path(output_dir).name + "_training")), seed=seed, num_train_epochs=1 if smoke else 3, learning_rate=2e-5, per_device_train_batch_size=8, per_device_eval_batch_size=16, gradient_accumulation_steps=2, weight_decay=0.01, warmup_ratio=0.10, eval_strategy="epoch", save_strategy="epoch", metric_for_best_model="f1", load_best_model_at_end=True, save_total_limit=1, fp16=torch.cuda.is_available(), report_to=[])
    def compute(eval_pred):
        logits, labels = eval_pred
        probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return {"f1": float(f1_score(labels, probabilities >= CLASSIFICATION_THRESHOLD, zero_division=0))}
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute)
    trainer.train()
    validation_logits, validation_labels = _predict_logits(trainer, tokenized["validation"])
    temperature = _fit_temperature(validation_logits, validation_labels)
    test_logits, test_labels = _predict_logits(trainer, tokenized["test"])
    validation_probabilities = torch.softmax(torch.tensor(validation_logits / temperature), dim=-1)[:, 1].numpy()
    test_probabilities = torch.softmax(torch.tensor(test_logits / temperature), dim=-1)[:, 1].numpy()
    metrics = {"validation": _metrics(validation_labels, validation_probabilities), "test": _metrics(test_labels, test_probabilities)}
    if not smoke and (metrics["test"]["f1"] < 0.90 or metrics["test"]["recall"] < 0.90 or metrics["test"]["brier_score"] > 0.15):
        print(json.dumps(metrics, indent=2), file=sys.stderr)
        raise RuntimeError("release quality gates failed; existing release artifact was left untouched")
    destination = Path(output_dir)
    if smoke:
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        trainer.save_model(destination)
        tokenizer.save_pretrained(destination)
        metadata = _metadata(manifest, temperature, metrics, sample_counts, class_counts, seed)
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        InferenceEngine(destination)
        return TrainingResult(destination, sample_counts, class_counts, metrics, temperature, metadata["model_version"])

    staging = destination.with_name(".staging_jailbreak_transformer")
    previous = destination.with_name(".previous_jailbreak_transformer")
    if previous.exists():
        raise RuntimeError(f"recovery conflict exists at {previous}; resolve manually before promoting release")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    trainer.save_model(staging)
    tokenizer.save_pretrained(staging)
    metadata = _metadata(manifest, temperature, metrics, sample_counts, class_counts, seed)
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    InferenceEngine(staging)

    has_prior = destination.exists()
    if has_prior:
        destination.rename(previous)
    try:
        staging.rename(destination)
        InferenceEngine(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if has_prior and previous.exists():
            previous.rename(destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return TrainingResult(destination, sample_counts, class_counts, metrics, temperature, metadata["model_version"])
