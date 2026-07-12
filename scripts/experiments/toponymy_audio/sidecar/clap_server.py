"""Stand-in for the VTSearch app side of the sidecar architecture.

Serves the active embedder's text branch over localhost HTTP (stdlib only):
POST /encode {"texts": [...]} -> {"vectors": [[...], ...]}. In a real
integration this would be an internal endpoint of the Flask app; here it
runs in the transformers-5 venv to prove cross-venv text encoding works.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

common.setup_env()

from run_toponymy import ClapTextEncoder  # noqa: E402

encoder = ClapTextEncoder(sys.argv[1] if len(sys.argv) > 1 else "clap")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/encode":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        vecs = encoder.encode(body["texts"])
        payload = json.dumps({"vectors": vecs.tolist()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8763
    print(f"clap text server ready on :{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
