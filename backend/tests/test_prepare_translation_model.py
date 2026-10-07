from __future__ import annotations

import io
import json

import pytest

from app.translation_artifact import MODEL_ID, MODEL_REVISION
from scripts import prepare_translation_model as preparation


@pytest.mark.parametrize("field,value", [("id", "different/model"), ("sha", "0" * 40)])
def test_wrong_pinned_identity_fails_before_download(tmp_path, monkeypatch, field, value):
    info = {"id": MODEL_ID, "sha": MODEL_REVISION}
    info[field] = value
    monkeypatch.setattr(preparation.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(info).encode()))
    with pytest.raises(ValueError, match="identity or exact revision"):
        preparation.prepare(tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize("license", ["apache-2.0", "MIT", "", "mit OR apache-2.0"])
def test_nonexact_model_card_license_fails_before_weights(tmp_path, monkeypatch, license):
    responses = iter([json.dumps({"id": MODEL_ID, "sha": MODEL_REVISION}).encode(),
                      f"---\nlicense: '{license}'\n---\nModel card".encode()])
    monkeypatch.setattr(preparation.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(next(responses)))
    with pytest.raises(ValueError, match="exact MIT"):
        preparation.prepare(tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def test_missing_isolation_binary_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation.shutil, "which", lambda name: None)
    def forbidden(*args, **kwargs):
        pytest.fail("weights must not download without isolation")
    monkeypatch.setattr(preparation.urllib.request, "urlopen", forbidden)
    with pytest.raises(RuntimeError, match="refusing unsafe fallback"):
        preparation.prepare(tmp_path / "artifact")
