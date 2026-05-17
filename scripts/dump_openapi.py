"""Print the flask-smorest OpenAPI 3.0 spec for the HTTP API to stdout.

Importing ``app`` constructs the Flask application and registers every
blueprint without starting the server, so the resulting
``api.spec.to_dict()`` is the same spec served at ``/api/openapi.json``
on the running app. Used by:

- ``frontend/scripts/regenerate-api-client.sh`` to refresh
  ``frontend/openapi.json`` before re-running the TS code generator.
- CI's drift guard, which regenerates the snapshot and fails if it
  differs from the checked-in copy.

The script is invoked as ``python scripts/dump_openapi.py`` from the
repository root — it relies on ``app.py`` being importable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    # ``VTSEARCH_SERVER_INIT=1`` would trigger model loading at import
    # time (the gunicorn entry point). Strip it so this script is a
    # pure spec dump that doesn't download weights or warm embedders.
    os.environ.pop("VTSEARCH_SERVER_INIT", None)

    # ``app.py`` lives at the repo root; this script sits in ``scripts/``.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # ``app.py`` (and a few of its transitive imports) prints status
    # banners to stdout at import time. Redirect that chatter to stderr
    # so this script's stdout stays a clean JSON document.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from app import api  # noqa: E402 — env-var stripping must precede import
    finally:
        sys.stdout = real_stdout

    spec = api.spec.to_dict()
    json.dump(spec, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
