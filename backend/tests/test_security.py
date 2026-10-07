from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rate_limiter import TokenBucketRateLimiter


@pytest.fixture
def client(tiny_model_dir: Path, tmp_path: Path):
    app = create_app(tiny_model_dir, multilingual_model_path=tmp_path / 'missing-multilingual', translation_model_path=tmp_path / 'missing-translator')
    with TestClient(app) as test_client:
        yield test_client


def test_security_headers_present_on_responses(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert response.headers["content-security-policy"] == "default-src 'self'; frame-ancestors 'none';"


def test_oversized_request_body_rejected_with_413(client):
    # Payload exceeding 65_536 bytes
    large_payload = "a" * 66_000
    response = client.post(
        "/api/analyze",
        content=large_payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(large_payload))},
    )
    assert response.status_code == 413
    assert "Payload too large" in response.json()["detail"]
    # Verify security headers are also present on 413 response
    assert response.headers["x-content-type-options"] == "nosniff"


def test_rate_limiting_triggers_429_after_quota(client):
    # Freeze rate limiter simulated time without touching global asyncio event loop time
    fixed_time = 1000.0
    client.app.state.rate_limiter._time_func = lambda: fixed_time

    # With default capacity 60, sending 70 rapid requests should allow exactly 60 and return 429 on subsequent
    results = []
    for i in range(70):
        resp = client.post("/api/analyze", json={"prompt": f"test prompt {i}"})
        results.append(resp.status_code)

    success_count = sum(1 for code in results if code == 200)
    rate_limited_count = sum(1 for code in results if code == 429)

    assert success_count == 60, f"Expected 60 successful requests, got {success_count}"
    assert rate_limited_count == 10, f"Expected 10 rate-limited requests, got {rate_limited_count}"

    # Verify 429 response structure
    exceeded_resp = client.post("/api/analyze", json={"prompt": "overflow request"})
    assert exceeded_resp.status_code == 429
    assert "Retry-After" in exceeded_resp.headers
    assert int(exceeded_resp.headers["Retry-After"]) >= 1
    assert "Rate limit exceeded" in exceeded_resp.json()["detail"]
    assert exceeded_resp.headers["x-content-type-options"] == "nosniff"

def test_rate_limiter_unit_behavior():
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=5)
    ip = "192.168.1.100"

    # Consume all 5 burst tokens
    for _ in range(5):
        allowed, retry_after = limiter.check_rate_limit(ip)
        assert allowed is True
        assert retry_after == 0

    # 6th should be rejected
    allowed, retry_after = limiter.check_rate_limit(ip)
    assert allowed is False
    assert retry_after >= 1

    # Reset clears buckets
    limiter.reset()
    allowed, retry_after = limiter.check_rate_limit(ip)
    assert allowed is True


def test_ssrf_blocks_cloud_metadata_unconditionally(monkeypatch):
    from app.redteam import _validate_endpoint

    # Blocked even when local redteam is enabled
    monkeypatch.setenv("ALLOW_LOCAL_REDTEAM", "1")
    with pytest.raises(ValueError, match="forbidden cloud metadata or link-local address"):
        _validate_endpoint("http://169.254.169.254/latest/meta-data")

    # Also blocked when local redteam is disabled
    monkeypatch.setenv("ALLOW_LOCAL_REDTEAM", "0")
    with pytest.raises(ValueError, match="forbidden cloud metadata or link-local address"):
        _validate_endpoint("http://169.254.169.254/latest/meta-data")


def test_ssrf_blocks_private_and_loopback_ips_when_not_allowed(monkeypatch):
    from app.redteam import _validate_endpoint

    monkeypatch.setenv("ALLOW_LOCAL_REDTEAM", "0")

    with pytest.raises(ValueError, match="forbidden loopback address"):
        _validate_endpoint("http://127.0.0.1:8000/generate")

    with pytest.raises(ValueError, match="forbidden private network address"):
        _validate_endpoint("http://10.0.0.1:8000/generate")

    with pytest.raises(ValueError, match="forbidden private network address"):
        _validate_endpoint("http://192.168.1.50:8000/generate")


def test_ssrf_allows_loopback_when_explicitly_enabled(monkeypatch):
    from app.redteam import _validate_endpoint

    monkeypatch.setenv("ALLOW_LOCAL_REDTEAM", "1")
    # Should not raise
    _validate_endpoint("http://127.0.0.1:9000/generate")


def test_ssrf_blocks_unusual_ports(monkeypatch):
    from app.redteam import _validate_endpoint

    monkeypatch.setenv("ALLOW_LOCAL_REDTEAM", "0")
    # Port 22 (SSH), 6379 (Redis) should be rejected
    with pytest.raises(ValueError, match="not an allowed red-team endpoint port"):
        _validate_endpoint("http://8.8.8.8:22/generate")

    with pytest.raises(ValueError, match="not an allowed red-team endpoint port"):
        _validate_endpoint("http://8.8.8.8:6379/generate")


def test_ssrf_blocks_non_http_schemes():
    from app.redteam import _validate_endpoint

    with pytest.raises(ValueError, match="http or https URL"):
        _validate_endpoint("gopher://127.0.0.1:9000/test")

    with pytest.raises(ValueError, match="http or https URL"):
        _validate_endpoint("file:///etc/passwd")


def test_detect_secrets_identifies_credentials():
    from app.signals import detect_secrets

    # Clean text
    assert detect_secrets("Hello! Here is the summary of your article.") == []

    # OpenAI key
    res = detect_secrets("My key is sk-proj-1234567890abcdefghij123456")
    assert "openai_api_key" in res

    # GitHub token
    res = detect_secrets("Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz12")
    assert "github_token" in res

    # JWT token
    res = detect_secrets("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozG48E9Qabcdef123456789")
    assert "jwt_token" in res

    # AWS Access Key
    res = detect_secrets("AWS key: AKIAIOSFODNN7EXAMPLE")
    assert "aws_access_key" in res

    # Private Key
    res = detect_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
    assert "private_key" in res


def test_validate_completion_endpoint_flags_secrets_and_leaks(client):
    # 1. Clean completion
    clean_resp = client.post(
        "/api/guardrail/validate-completion",
        json={
            "completion": "Paris is the capital of France.",
            "system_prompt_snippets": ["You are a confidential assistant for Project Titan."],
        },
    )
    assert clean_resp.status_code == 200
    clean_data = clean_resp.json()
    assert clean_data["safe"] is True
    assert clean_data["detected_secrets"] == []
    assert clean_data["system_prompt_leaks"] == []

    # 2. Leaked system prompt snippet
    leaked_resp = client.post(
        "/api/guardrail/validate-completion",
        json={
            "completion": "As instructed, you are a confidential assistant for Project Titan.",
            "system_prompt_snippets": ["You are a confidential assistant for Project Titan."],
        },
    )
    assert leaked_resp.status_code == 200
    leaked_data = leaked_resp.json()
    assert leaked_data["safe"] is False
    assert len(leaked_data["system_prompt_leaks"]) == 1

    # 3. Secret leak
    secret_resp = client.post(
        "/api/guardrail/validate-completion",
        json={
            "completion": "Here is the API key: sk-proj-1234567890abcdefghij9999",
            "system_prompt_snippets": None,
        },
    )
    assert secret_resp.status_code == 200
    secret_data = secret_resp.json()
    assert secret_data["safe"] is False
    assert "openai_api_key" in secret_data["detected_secrets"]


def test_dockerfile_and_nginx_security_configurations():
    root = Path(__file__).parents[2]

    backend_dockerfile = (root / "backend" / "Dockerfile").read_text(encoding="utf-8")
    assert "useradd -u 10001" in backend_dockerfile
    assert "USER appuser" in backend_dockerfile
    assert "chown -R appuser:appuser" in backend_dockerfile

    frontend_dockerfile = (root / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY frontend/nginx.conf /etc/nginx/nginx.conf" in frontend_dockerfile

    nginx_conf = (root / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    assert "server_tokens off;" in nginx_conf
    assert "user nginx;" in nginx_conf
    assert "X-Content-Type-Options" in nginx_conf
    assert "X-Frame-Options" in nginx_conf
    assert "Content-Security-Policy" in nginx_conf
