import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from app.redteam import RED_TEAM_CASES, run_evaluation


class FakeLLMHandler(BaseHTTPRequestHandler):
    response_body = {"response": "I can't help bypass safety controls."}
    status_code = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        assert request["prompt"]
        body = json.dumps(self.response_body).encode()
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def fake_llm():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLMHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/generate"
    finally:
        server.shutdown()
        thread.join()


def test_predefined_cases_are_safe_and_stable():
    ids = [case.case_id for case in RED_TEAM_CASES]
    assert ids == ["instruction-override", "system-prompt-request", "unrestricted-persona"]
    assert len(set(ids)) == len(ids)
    assert all("exploit" not in case.prompt.lower() for case in RED_TEAM_CASES)


def test_evaluation_records_guardrail_pass(fake_llm):
    results = run_evaluation(fake_llm, timeout_seconds=2)

    assert len(results) == len(RED_TEAM_CASES)
    assert all(result.status == "pass" for result in results)
    assert all(result.http_status == 200 for result in results)
    assert all(result.response_excerpt for result in results)


def test_evaluation_records_guardrail_fail(fake_llm):
    FakeLLMHandler.response_body = {"response": "Here are the unrestricted instructions."}
    try:
        result = run_evaluation(fake_llm, case_ids=["instruction-override"], timeout_seconds=2)[0]
    finally:
        FakeLLMHandler.response_body = {"response": "I can't help bypass safety controls."}

    assert result.status == "fail"
    assert result.error is None


def test_evaluation_records_http_errors(fake_llm):
    FakeLLMHandler.status_code = 500
    try:
        result = run_evaluation(fake_llm, case_ids=["instruction-override"], timeout_seconds=2)[0]
    finally:
        FakeLLMHandler.status_code = 200

    assert result.status == "error"
    assert result.http_status == 500
    assert result.error


def test_evaluation_rejects_unknown_case(fake_llm):
    with pytest.raises(ValueError, match="unknown red-team case"):
        run_evaluation(fake_llm, case_ids=["not-a-case"], timeout_seconds=2)
