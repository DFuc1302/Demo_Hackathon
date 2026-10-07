from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import io
import urllib.request

import pytest

from app.dataset_contract import (check_cross_split_leakage, normalize_text_key,
                                  serialize_jsonl, sha256_bytes, validate_split_rows)
from app.training import load_dataset_splits
from scripts.prepare_v2_dataset import _read_split, prepare_dataset


def _csv(rows):
    output = []
    for row in rows:
        output.append(f'{row[0]},{row[1]}')
    return ('text,label\n' + '\n'.join(output) + '\n').encode()


def _expected(raw, rows, split="train"):
    prep = [{"text": r[0].strip(), "label": r[1]} for r in rows if r[0].strip()]
    serialized = serialize_jsonl(prep)
    return {
        "source_file": f"data/{split}.csv",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_sha256": hashlib.sha256(serialized).hexdigest(),
        "rows": len(rows),
        "class_counts": {"0": sum(row[1] == 0 for row in rows), "1": sum(row[1] == 1 for row in rows)},
    }
def test_split_validation_rejects_empty_text_and_invalid_labels():
    rows = [("", 0), ("safe", 1)]
    raw = _csv(rows)
    with pytest.raises(ValueError, match="empty text"):
        _read_split(raw, "train", _expected(raw, rows))
    rows = [("safe", 0), ("attack", 2)]
    raw = _csv(rows)
    with pytest.raises(ValueError, match="outside"):
        _read_split(raw, "train", _expected(raw, rows))


def test_split_validation_rejects_normalized_duplicates():
    rows = [("same prompt", 0), ("Same   prompt", 1)]
    raw = _csv(rows)
    with pytest.raises(ValueError, match="duplicate"):
        _read_split(raw, "train", _expected(raw, rows))

def test_split_validation_rejects_manifest_hash_mismatch():
    raw = _csv([("safe", 0), ("attack", 1)])
    with pytest.raises(ValueError, match="hash"):
        _read_split(raw, "train", {"source_sha256": "wrong", "rows": 2, "class_counts": {"0": 1, "1": 1}})

def _create_clean_dataset(tmp_path: Path):
    splits_data = {
        "train": [("train benign", 0), ("train jailbreak", 1)],
        "validation": [("val benign", 0), ("val jailbreak", 1)],
        "test": [("test benign", 0), ("test jailbreak", 1)],
    }
    manifest = {
        "schema_version": 2,
        "dataset_id": "clean-org/clean-dataset",
        "dataset_revision": "a" * 40,
        "model_id": "microsoft/deberta-v3-small",
        "model_revision": "b" * 40,
        "license": "mit",
        "language": "en",
        "columns": ["text", "label"],
        "splits": {},
    }
    dataset_dir = tmp_path / "v2"
    dataset_dir.mkdir(parents=True)
    raw_csvs = {}
    for split, pairs in splits_data.items():
        raw = _csv(pairs)
        raw_csvs[split] = raw
        exp = _expected(raw, pairs, split=split)
        manifest["splits"][split] = exp
        rows = [{"text": p[0], "label": p[1]} for p in pairs]
        (dataset_dir / f"{split}.jsonl").write_bytes(serialize_jsonl(rows))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, dataset_dir, manifest, raw_csvs


def test_validate_split_rows_rejects_whitespace_only_text():
    rows = [{"text": "   ", "label": 0}, {"text": "benign", "label": 1}]
    with pytest.raises(ValueError, match="empty text"):
        validate_split_rows(rows, "train")


def test_validate_split_rows_rejects_boolean_labels():
    rows = [{"text": "safe", "label": True}]
    with pytest.raises(ValueError, match="invalid label"):
        validate_split_rows(rows, "train")


def test_offline_preparation_rejects_missing_cached_split(tmp_path: Path):
    manifest_path, dataset_dir, manifest, _ = _create_clean_dataset(tmp_path)
    (dataset_dir / "test.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        prepare_dataset(manifest_path, dataset_dir, offline=True)


def test_offline_preparation_rejects_prepared_hash_mismatch(tmp_path: Path):
    manifest_path, dataset_dir, manifest, _ = _create_clean_dataset(tmp_path)
    test_file = dataset_dir / "test.jsonl"
    test_file.write_bytes(test_file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="prepared test hash does not match"):
        prepare_dataset(manifest_path, dataset_dir, offline=True)


def test_offline_preparation_rejects_same_count_content_tampering(tmp_path: Path):
    manifest_path, dataset_dir, manifest, _ = _create_clean_dataset(tmp_path)
    tampered_rows = [{"text": "tampered benign", "label": 0}, {"text": "tampered jailbreak", "label": 1}]
    (dataset_dir / "test.jsonl").write_bytes(serialize_jsonl(tampered_rows))
    with pytest.raises(ValueError, match="prepared test hash does not match"):
        prepare_dataset(manifest_path, dataset_dir, offline=True)


@pytest.mark.parametrize(
    "left,right",
    [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ],
)
def test_prepare_dataset_rejects_each_split_pair_leakage(tmp_path: Path, left: str, right: str):
    manifest_path, dataset_dir, manifest, _ = _create_clean_dataset(tmp_path)
    rows_left = [{"text": "leaked text", "label": 0}, {"text": f"{left} other", "label": 1}]
    rows_right = [{"text": "Leaked   Text", "label": 0}, {"text": f"{right} other", "label": 1}]
    (dataset_dir / f"{left}.jsonl").write_bytes(serialize_jsonl(rows_left))
    (dataset_dir / f"{right}.jsonl").write_bytes(serialize_jsonl(rows_right))
    manifest["splits"][left]["prepared_sha256"] = sha256_bytes(serialize_jsonl(rows_left))
    manifest["splits"][right]["prepared_sha256"] = sha256_bytes(serialize_jsonl(rows_right))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        prepare_dataset(manifest_path, dataset_dir, offline=True)


def test_online_preparation_revision_acceptance_and_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path, dataset_dir, manifest, raw_csvs = _create_clean_dataset(tmp_path)
    target_dir = tmp_path / "prepared_online"

    class FakeResponse:
        def __init__(self, data: bytes):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def fake_urlopen(url, *args, **kwargs):
        if "/api/datasets/" in url:
            return FakeResponse(json.dumps({"sha": manifest["dataset_revision"]}).encode())
        if "README.md" in url:
            return FakeResponse(b"---\nlicense: mit\nlanguage:\n- en\n---\n# Clean\n")
        for split, raw in raw_csvs.items():
            if split in url:
                return FakeResponse(raw)
        raise ValueError(f"unexpected url: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    results = prepare_dataset(manifest_path, target_dir, offline=False)
    assert len(results) == 3
    assert (target_dir / "train.jsonl").exists()

    # Test revision mismatch
    def fake_urlopen_mismatch(url, *args, **kwargs):
        if "/api/datasets/" in url:
            return FakeResponse(json.dumps({"sha": "different_revision_hash"}).encode())
        return fake_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_mismatch)
    with pytest.raises(RuntimeError, match="revision"):
        prepare_dataset(manifest_path, tmp_path / "other", offline=False)


def test_online_preparation_rejects_missing_mit_or_english(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path, dataset_dir, manifest, raw_csvs = _create_clean_dataset(tmp_path)

    class FakeResponse:
        def __init__(self, data: bytes):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    # Non-mit license
    def fake_urlopen_bad_license(url, *args, **kwargs):
        if "/api/datasets/" in url:
            return FakeResponse(json.dumps({"sha": manifest["dataset_revision"]}).encode())
        if "README.md" in url:
            return FakeResponse(b"---\nlicense: apache-2.0\nlanguage: en\n---\n")
        return FakeResponse(b"")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_bad_license)
    with pytest.raises(RuntimeError, match="MIT"):
        prepare_dataset(manifest_path, tmp_path / "out1", offline=False)

    # Missing english
    def fake_urlopen_bad_lang(url, *args, **kwargs):
        if "/api/datasets/" in url:
            return FakeResponse(json.dumps({"sha": manifest["dataset_revision"]}).encode())
        if "README.md" in url:
            return FakeResponse(b"---\nlicense: mit\nlanguage: fr\n---\n")
        return FakeResponse(b"")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_bad_lang)
    with pytest.raises(RuntimeError, match="English"):
        prepare_dataset(manifest_path, tmp_path / "out2", offline=False)


def test_load_dataset_splits_verifies_prepared_hashes_and_leakage(tmp_path: Path):
    manifest_path, dataset_dir, manifest, _ = _create_clean_dataset(tmp_path)
    ds = load_dataset_splits(dataset_dir, manifest_path)
    assert set(ds.keys()) == {"train", "validation", "test"}

    # Tamper with test.jsonl
    test_file = dataset_dir / "test.jsonl"
    test_file.write_bytes(test_file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash does not match"):
        load_dataset_splits(dataset_dir, manifest_path)
def test_smoke_training_direct_call_rejects_release_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.training as training_mod
    release_dir = tmp_path / "models" / "jailbreak_transformer"
    release_dir.mkdir(parents=True)
    sentinel = release_dir / "sentinel.txt"
    sentinel.write_text("protected", encoding="utf-8")
    monkeypatch.setattr(training_mod, "RELEASE_MODEL_DIR", release_dir)
    with pytest.raises(ValueError, match="smoke training cannot write to the release model directory"):
        training_mod.train_transformer(tmp_path / "data", release_dir, manifest_path=tmp_path / "manifest.json", smoke=True)
    assert sentinel.read_text(encoding="utf-8") == "protected"
    assert list(release_dir.iterdir()) == [sentinel]


def test_smoke_training_cli_rejects_release_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import scripts.train_model as train_script
    release_dir = tmp_path / "models" / "jailbreak_transformer"
    release_dir.mkdir(parents=True)
    sentinel = release_dir / "sentinel.txt"
    sentinel.write_text("protected", encoding="utf-8")
    monkeypatch.setattr(train_script, "DEFAULT_MODEL", release_dir)
    monkeypatch.setattr("sys.argv", ["train_model.py", "--smoke", "--model-dir", str(release_dir)])
    with pytest.raises(SystemExit, match="smoke training cannot write to the release model directory"):
        train_script.main()
    assert sentinel.read_text(encoding="utf-8") == "protected"
    assert list(release_dir.iterdir()) == [sentinel]


def test_smoke_training_cli_defaults_to_tmp_smoke_directory(monkeypatch: pytest.MonkeyPatch):
    import scripts.train_model as train_script
    from app.training import TrainingResult
    captured = {}
    def fake_train(dataset_dir, output_dir, *, manifest_path, seed=42, smoke=False):
        captured["dataset_dir"] = dataset_dir
        captured["output_dir"] = output_dir
        captured["manifest_path"] = manifest_path
        captured["smoke"] = smoke
        return TrainingResult(
            model_dir=output_dir,
            sample_counts={"train": 64, "validation": 32, "test": 32},
            class_counts={"train": {"0": 32, "1": 32}, "validation": {"0": 16, "1": 16}, "test": {"0": 16, "1": 16}},
            metrics={"validation": {"f1": 0.95}, "test": {"f1": 0.95}},
            temperature=1.0,
            model_version="test-model-v2",
        )
    monkeypatch.setattr(train_script, "train_transformer", fake_train)
    monkeypatch.setattr("sys.argv", ["train_model.py", "--smoke"])
    train_script.main()
    assert captured["smoke"] is True
    assert captured["output_dir"] == Path("/tmp/demo-v2-smoke")


def test_balanced_subset_determinism_and_class_balance():
    from datasets import Dataset
    from app.training import _balanced_subset
    data = [{"text": f"benign {i}", "label": 0} for i in range(10)] + [
        {"text": f"attack {i}", "label": 1} for i in range(10)
    ]
    ds = Dataset.from_list(data)
    subset1 = _balanced_subset(ds, 6, seed=42)
    subset2 = _balanced_subset(ds, 6, seed=42)
    assert len(subset1) == 6
    assert [r["text"] for r in subset1] == [r["text"] for r in subset2]
    c0 = sum(r["label"] == 0 for r in subset1)
    c1 = sum(r["label"] == 1 for r in subset1)
    assert c0 == 3 and c1 == 3

    imbalanced_data = [{"text": "benign 0", "label": 0}] + [
        {"text": f"attack {i}", "label": 1} for i in range(10)
    ]
    imbalanced_ds = Dataset.from_list(imbalanced_data)
    with pytest.raises(ValueError, match="enough examples"):
        _balanced_subset(imbalanced_ds, 6, seed=42)


def test_fit_temperature_and_calibrated_nll():
    import numpy as np
    import torch
    from app.training import _fit_temperature
    np.random.seed(42)
    logits = np.array([[-2.0, 2.0], [3.0, -3.0], [-1.0, 1.0], [2.0, -2.0]], dtype=np.float32)
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    temp = _fit_temperature(logits, labels)
    assert np.isfinite(temp) and temp > 0.0

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.int64)
    uncalibrated_nll = torch.nn.functional.cross_entropy(logits_t, labels_t).item()
    calibrated_nll = torch.nn.functional.cross_entropy(logits_t / temp, labels_t).item()
    assert calibrated_nll <= uncalibrated_nll + 1e-5


def test_metrics_and_ece():
    import numpy as np
    from app.training import _metrics, _ece
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    probs = np.array([0.1, 0.9, 0.50, 0.8], dtype=np.float32)
    m = _metrics(labels, probs)
    required_metric_keys = {
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score",
        "expected_calibration_error", "confusion_matrix"
    }
    assert set(m.keys()) == required_metric_keys
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "expected_calibration_error"):
        assert 0.0 <= m[k] <= 1.0
    cm = m["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2
    assert cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] == 4

    perfect_labels = np.array([0, 1])
    perfect_probs = np.array([0.0, 1.0])
    assert _ece(perfect_labels, perfect_probs) == 0.0


def test_metadata_generation_satisfies_schema_validator():
    from app.training import _metadata
    from app.model_metadata import validate_model_metadata
    metric = {
        "accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1": 0.95,
        "roc_auc": 0.98, "pr_auc": 0.98, "brier_score": 0.05,
        "expected_calibration_error": 0.02, "confusion_matrix": [[10, 0], [0, 10]]
    }
    manifest = {
        "model_revision": "a" * 40,
        "dataset_id": "clean/data",
        "dataset_revision": "b" * 40,
        "splits": {
            "train": {"source_sha256": "c" * 64},
            "validation": {"source_sha256": "d" * 64},
            "test": {"source_sha256": "e" * 64},
        },
    }
    sample_counts = {"train": 40, "validation": 20, "test": 20}
    class_counts = {
        "train": {"0": 20, "1": 20},
        "validation": {"0": 10, "1": 10},
        "test": {"0": 10, "1": 10},
    }
    meta = _metadata(manifest, 1.25, {"validation": metric, "test": metric}, sample_counts, class_counts, 42)
    validated = validate_model_metadata(meta, "test_output")
    assert validated["calibration_temperature"] == 1.25
    assert validated["seed"] == 42
    assert "transformers" in validated["software_versions"]
    assert "datasets" in validated["software_versions"]


def _setup_training_mocks(monkeypatch, tmp_path):
    import app.training as training_mod
    manifest_path, dataset_dir, manifest, _ = _create_clean_dataset(tmp_path)
    destination = tmp_path / "models" / "jailbreak_transformer"

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [0, 1]}
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "vocab.txt").write_text("vocab")

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def train(self):
            pass
        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_text("weights")
        def predict(self, ds):
            import numpy as np
            n = len(ds)
            # High confidence benign / jailbreak predictions to easily pass quality gates
            preds = np.zeros((n, 2), dtype=np.float32)
            for i, row in enumerate(ds):
                if row["label"] == 1:
                    preds[i, 1] = 5.0
                else:
                    preds[i, 0] = 5.0
            return type("Output", (), {"predictions": preds})()

    monkeypatch.setattr(training_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(training_mod.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(training_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(training_mod, "DataCollatorWithPadding", lambda *a: None)
    return manifest_path, dataset_dir, destination


def test_promotion_rejects_existing_recovery_backup_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.training as training_mod
    manifest_path, dataset_dir, destination = _setup_training_mocks(monkeypatch, tmp_path)
    destination.mkdir(parents=True)
    previous = destination.with_name(".previous_jailbreak_transformer")
    previous.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="recovery conflict exists"):
        training_mod.train_transformer(dataset_dir, destination, manifest_path=manifest_path, smoke=False)


def test_promotion_staging_validation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.training as training_mod
    manifest_path, dataset_dir, destination = _setup_training_mocks(monkeypatch, tmp_path)
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("active_release")

    def fake_engine(path):
        if ".staging" in str(path):
            raise ValueError("corrupt staging metadata")
        return object()

    monkeypatch.setattr(training_mod, "InferenceEngine", fake_engine)
    with pytest.raises(ValueError, match="corrupt staging metadata"):
        training_mod.train_transformer(dataset_dir, destination, manifest_path=manifest_path, smoke=False)
    assert sentinel.read_text() == "active_release"


def test_promotion_promoted_reload_failure_with_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.training as training_mod
    manifest_path, dataset_dir, destination = _setup_training_mocks(monkeypatch, tmp_path)
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("active_release")

    def fake_engine(path):
        if Path(path) == destination:
            raise RuntimeError("promoted reload failed")
        return object()

    monkeypatch.setattr(training_mod, "InferenceEngine", fake_engine)
    with pytest.raises(RuntimeError, match="promoted reload failed"):
        training_mod.train_transformer(dataset_dir, destination, manifest_path=manifest_path, smoke=False)
    assert sentinel.read_text() == "active_release"
    previous = destination.with_name(".previous_jailbreak_transformer")
    assert not previous.exists()


def test_first_release_failure_leaves_no_release_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.training as training_mod
    manifest_path, dataset_dir, destination = _setup_training_mocks(monkeypatch, tmp_path)
    assert not destination.exists()

    def fake_engine(path):
        if Path(path) == destination:
            raise RuntimeError("first release reload failed")
        return object()

    monkeypatch.setattr(training_mod, "InferenceEngine", fake_engine)
    with pytest.raises(RuntimeError, match="first release reload failed"):
        training_mod.train_transformer(dataset_dir, destination, manifest_path=manifest_path, smoke=False)
    assert not destination.exists()


def test_successful_promotion_cleans_up_previous_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.training as training_mod
    manifest_path, dataset_dir, destination = _setup_training_mocks(monkeypatch, tmp_path)
    destination.mkdir(parents=True)
    old_file = destination / "old.txt"
    old_file.write_text("old_version")

    monkeypatch.setattr(training_mod, "InferenceEngine", lambda path: object())
    result = training_mod.train_transformer(dataset_dir, destination, manifest_path=manifest_path, smoke=False)
    assert result.model_dir == destination
    assert not old_file.exists()
    assert (destination / "model.safetensors").exists()
    previous = destination.with_name(".previous_jailbreak_transformer")
    assert not previous.exists()
