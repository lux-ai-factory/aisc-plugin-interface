import http.server
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class Stub:
    """A local HTTP server with canned answers per path; records every request."""

    def __init__(self):
        self.routes = {}      # path -> list of (status, body, delay) served in order, last one repeats
        self.seen = []
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                # header names are case-insensitive in HTTP: keep them lower-cased
                stub.seen.append({"method": self.command, "path": self.path,
                                  "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                plan = stub.routes.get(self.path.split("?")[0]) or [(404, {"detail": "no route"}, 0)]
                status, payload, delay = plan.pop(0) if len(plan) > 1 else plan[0]
                if delay:
                    time.sleep(delay)
                if callable(payload):
                    payload = payload(stub.seen[-1])
                if status in (301, 302):
                    self.send_response(status)
                    self.send_header("Location", payload)
                    self.end_headers()
                    return
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PUT = _serve

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.host = f"127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def route(self, path, *responses):
        self.routes[path] = [r if len(r) == 3 else (*r, 0) for r in responses]

    def stop(self):
        self.server.shutdown()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.stop()


class FakePlugin:
    """What EndpointClient needs from a plugin: its inputs and upload_artifact."""

    def __init__(self, inputs=None):
        self.inputs = inputs or {}
        self.artifacts = {}

    def get_input_data(self, name):
        return self.inputs.get(name)

    def upload_artifact(self, name, content):
        self.artifacts[name] = content


@pytest.fixture
def plugin():
    return FakePlugin
