from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        body = json.dumps({"response": "I can't help bypass safety controls."}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


if __name__ == "__main__":
    print("Fake LLM listening on http://127.0.0.1:9000/generate")
    HTTPServer(("127.0.0.1", 9000), FakeLLMHandler).serve_forever()
