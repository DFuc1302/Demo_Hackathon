from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast


def _metadata() -> dict:
    metric = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9, "roc_auc": 0.95, "pr_auc": 0.95, "brier_score": 0.1, "expected_calibration_error": 0.05, "confusion_matrix": [[9, 1], [1, 9]]}
    return {
        "schema_version": 2,
        "model_version": "test-v2",
        "model_id": "local-test-bert",
        "model_revision": "local",
        "dataset_id": "local-test",
        "dataset_revision": "local",
        "dataset_split_hashes": {
            "train": "0" * 64,
            "validation": "1" * 64,
            "test": "2" * 64,
        },
        "software_versions": {
            "python": "3.12.3",
            "torch": torch.__version__,
            "transformers": "4.49.0",
            "datasets": "3.3.2",
        },
        "label_mapping": {"0": "benign", "1": "jailbreak"},
        "max_token_length": 256,
        "calibration_temperature": 1.0,
        "classification_threshold": 0.5,
        "risk_thresholds": {"low": 0.2, "high": 0.8},
        "seed": 42,
        "sample_counts": {"train": 20, "validation": 20, "test": 20},
        "class_counts": {
            "train": {"0": 10, "1": 10},
            "validation": {"0": 10, "1": 10},
            "test": {"0": 10, "1": 10},
        },
        "metrics": {"validation": metric, "test": metric},
    }

@pytest.fixture(scope="session")
def tiny_model_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_dir = tmp_path_factory.mktemp("tiny-model-template")
    (model_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nplease\nsummarize\nignore\nall\nprevious\ninstructions\nreveal\nhidden\nsystem\nprompt\nx\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)
    config = BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=2, id2label={0: "benign", 1: "jailbreak"}, label2id={"benign": 0, "jailbreak": 1})
    torch.manual_seed(42)
    BertForSequenceClassification(config).save_pretrained(model_dir, safe_serialization=True)
    (model_dir / "metadata.json").write_text(json.dumps(_metadata(), indent=2), encoding="utf-8")
    return model_dir


@pytest.fixture
def tiny_model_dir(tmp_path: Path, tiny_model_template: Path) -> Path:
    model_dir = tmp_path / "tiny-model"
    shutil.copytree(tiny_model_template, model_dir)
    return model_dir


@pytest.fixture(autouse=True)
def allow_local_redteam_default(monkeypatch):
    monkeypatch.setenv("ALLOW_LOCAL_REDTEAM", "1")
