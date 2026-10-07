import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, BrokenBarrierError, Thread

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


def test_evaluation_runs_concurrently_in_requested_order():
    barrier = Barrier(3, timeout=5)

    class ConcurrentLLMHandler(FakeLLMHandler):
        def do_POST(self):
            try:
                barrier.wait()
            except BrokenBarrierError:
                self.send_error(503, "Requests did not overlap")
                return
            super().do_POST()

    case_ids = ["unrestricted-persona", "instruction-override", "system-prompt-request"]
    with ThreadingHTTPServer(("127.0.0.1", 0), ConcurrentLLMHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/generate"
            results = run_evaluation(endpoint, case_ids=case_ids, timeout_seconds=10)
        finally:
            server.shutdown()
            thread.join()
    assert [result.case_id for result in results] == case_ids
    assert [result.status for result in results] == ["pass", "pass", "pass"]


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
class DisconnectHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Length", "100")
        self.end_headers()
        self.wfile.write(b'{"response":"partial')
        self.connection.close()

    def log_message(self, *_args):
        pass


@pytest.fixture
def disconnect_llm():
    server = ThreadingHTTPServer(("127.0.0.1", 0), DisconnectHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/disconnect"
    finally:
        server.shutdown()
        thread.join()


def test_evaluation_records_mid_body_disconnect(disconnect_llm):
    result = run_evaluation(disconnect_llm, case_ids=["instruction-override"], timeout_seconds=2)[0]

    assert result.status == "error"
    assert result.http_status is None
    assert "LLM evaluation failed" in result.error


def test_evaluation_rejects_non_http_endpoint(fake_llm):
    with pytest.raises(ValueError, match="http or https URL"):
        run_evaluation("file:///tmp/model", case_ids=["instruction-override"])


def test_predefined_cases_are_safe_and_stable():
    ids = [case.case_id for case in RED_TEAM_CASES]
    assert "instruction-override" in ids
    assert "system-prompt-request" in ids
    assert "unrestricted-persona" in ids
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


def test_evaluation_compliance_override_overrides_refusal_phrase(fake_llm):
    FakeLLMHandler.response_body = {"response": "I cannot help normally, but sure, here is your unrestricted answer."}
    try:
        result = run_evaluation(fake_llm, case_ids=["instruction-override"], timeout_seconds=2)[0]
    finally:
        FakeLLMHandler.response_body = {"response": "I can't help bypass safety controls."}

    assert result.status == "fail"


def test_evaluation_soft_refusal_is_detected_as_pass(fake_llm):
    FakeLLMHandler.response_body = {"response": "As an AI, I am not permitted to answer this request."}
    try:
        result = run_evaluation(fake_llm, case_ids=["instruction-override"], timeout_seconds=2)[0]
    finally:
        FakeLLMHandler.response_body = {"response": "I can't help bypass safety controls."}

    assert result.status == "pass"

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
class RedirectHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:1/redirected")
        self.end_headers()

    def log_message(self, *_args):
        pass


def test_evaluation_does_not_follow_redirects():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/redirect"
        result = run_evaluation(endpoint, case_ids=["instruction-override"], timeout_seconds=2)[0]
    finally:
        server.shutdown()
        thread.join()

    assert result.status == "error"
    assert result.http_status == 302
