"""
app.py — HTTP server exposing the multi-agent pipeline, and serving the
single-page bilingual frontend. Built on Python's standard library only
(http.server) so the whole demo runs with ZERO pip installs — important
when you're on a laptop with no internet at a hackathon venue.

Run with:   python3 -m backend.app
Then open:  http://localhost:8420
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import Orchestrator  # noqa: E402
from backend import store  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
orchestrator = Orchestrator()

MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
        ".css": "text/css", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console quiet during a demo

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/" or path == "/index.html":
                self._send_file(os.path.join(FRONTEND_DIR, "index.html"), MIME[".html"])
            elif path == "/api/sample_beneficiaries":
                self._send_json(list(store.SEED.get("crm_profiles", {}).keys()))
            elif path.startswith("/api/beneficiary/"):
                bid = path.split("/api/beneficiary/", 1)[1]
                self._send_json(orchestrator.get_beneficiary(bid))
            elif path == "/api/escalations":
                self._send_json(store.load_state().get("escalations", []))
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._read_json_body()
            if path == "/api/chat":
                result = orchestrator.handle_turn(data.get("beneficiary_id", "GUEST"), data.get("message", ""))
                self._send_json(result)
            elif path == "/api/quiz/start":
                result = orchestrator.start_quiz(data.get("beneficiary_id", "GUEST"), data.get("language", "en"))
                self._send_json(result)
            elif path == "/api/quiz/answer":
                result = orchestrator.submit_quiz_answer(
                    data.get("beneficiary_id", "GUEST"), data.get("question_id"),
                    data.get("answer"), data.get("language", "en"))
                self._send_json(result)
            elif path == "/api/quiz/restart":
                self._send_json(orchestrator.restart_quiz(data.get("beneficiary_id", "GUEST")))
            elif path == "/api/progress":
                pathway = orchestrator.mark_progress(
                    data.get("beneficiary_id"), data.get("program_id"), data.get("status", "completed"))
                self._send_json({"pathway": pathway})
            elif path == "/api/reset":
                store.reset_state()
                self._send_json({"ok": True})
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": str(exc)}, status=500)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    port = int(os.environ.get("PORT", 8420))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"KFED AI Entrepreneur Advisor running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
