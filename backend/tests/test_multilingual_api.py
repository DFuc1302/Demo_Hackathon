from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import MultilingualAnalyzeResponse


def test_multilingual_api_unavailable_returns_503(tmp_path):
    # App created with nonexistent multilingual model paths
    app = create_app(
        multilingual_model_path=tmp_path / "nonexistent_multi",
        translation_model_path=tmp_path / "nonexistent_trans",
    )
    with TestClient(app) as client:
        # GET model info returns available=False
        info_resp = client.get("/api/multilingual/model-info")
        assert info_resp.status_code == 200
        assert info_resp.json()["available"] is False

        # POST analyze-multilingual returns 503
        resp = client.post(
            "/api/analyze-multilingual",
            json={"prompt": "habari", "language": "sw", "mode": "compare"},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Multilingual model and translation artifacts are not installed."

        # English API remains functional
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"



def test_multilingual_api_installed_release_is_unavailable_until_uniform_gates_pass():
    with TestClient(create_app()) as client:
        info = client.get("/api/multilingual/model-info")
        assert info.status_code == 200
        assert info.json()["available"] is False
        response = client.post("/api/analyze-multilingual", json={"prompt": "habari", "language": "sw", "mode": "compare"})
        assert response.status_code == 503
        assert response.json()["detail"] == "Multilingual model and translation artifacts are not installed."
