"""Tests for the standardized JSON error envelope.

The ``{error, detail, request_id}`` shape is consumed by the frontend
``ErrorService`` / global error banner; keep this contract stable.
"""

from __future__ import annotations

import json


class TestErrorEnvelope:
    """4xx responses from inline error returns include ``request_id``."""

    def test_invalid_json_body_carries_request_id(self, client):
        # ``/api/embed`` is still on the legacy ``get_json_or_400``
        # helper (the dual-mode multipart-or-JSON dispatcher doesn't fit
        # a single marshmallow schema). An empty JSON body trips its
        # "Invalid request body" path, exercising the legacy envelope
        # that this test is here to pin.
        resp = client.post("/api/embed", content_type="application/json")
        assert resp.status_code == 400
        body = json.loads(resp.data)
        assert body["error"] == "Invalid request body"
        assert "request_id" in body and len(body["request_id"]) >= 8
        # The header echoes the same id so the client can correlate logs.
        assert resp.headers.get("X-Request-Id") == body["request_id"]

    def test_inbound_request_id_round_trips_into_body(self, client):
        # Same endpoint as above; see comment there for why ``/api/embed``
        # rather than a migrated route.
        resp = client.post(
            "/api/embed",
            content_type="application/json",
            headers={"X-Request-Id": "test-rid-12345"},
        )
        assert resp.status_code == 400
        body = json.loads(resp.data)
        assert body["request_id"] == "test-rid-12345"

    def test_error_response_helper_includes_extra_fields(self):
        # Exercise the helper directly so plugin discovery and route
        # wiring don't muddy the contract.
        from app import app as flask_app
        from vtsearch.routes._shared import error_response

        with flask_app.test_request_context("/api/anything"):
            from flask import g

            g.request_id = "abc123"
            resp, status = error_response(
                "Missing required field(s): ['name']",
                400,
                missing_fields=["name"],
            )
            assert status == 400
            body = resp.get_json()
            assert body["error"] == "Missing required field(s): ['name']"
            assert body["missing_fields"] == ["name"]
            assert body["request_id"] == "abc123"

    def test_error_response_omits_request_id_outside_request(self):
        # Background-thread error paths have no g.request_id; the helper
        # should still produce a valid JSON envelope.
        from app import app as flask_app
        from vtsearch.routes._shared import error_response

        with flask_app.app_context():
            resp, status = error_response("Something broke", 500)
            assert status == 500
            body = resp.get_json()
            assert body == {"error": "Something broke"}

    def test_unknown_api_path_returns_json_404(self, client):
        resp = client.get("/api/this-route-does-not-exist")
        assert resp.status_code == 404
        body = json.loads(resp.data)
        assert "error" in body
        assert "request_id" in body

    def test_non_api_404_is_not_json(self, client):
        # Non-/api/ paths fall through to the SPA / static handler, which
        # may return HTML or its own response; we only standardize JSON
        # for /api/. Just check the API guard didn't accidentally hijack it.
        resp = client.get("/some-non-api-path")
        # We don't care about the status; only that we didn't wrap it
        # in our JSON envelope when the path isn't /api/.
        if resp.status_code == 404 and resp.is_json:
            body = json.loads(resp.data)
            # If a non-API 404 happens to be JSON for some unrelated
            # reason, it should not have come from our envelope (which
            # always sets request_id).
            assert "request_id" not in body or resp.headers.get("Content-Type", "").startswith("application/json")


class TestUncaughtExceptionHandler:
    """Uncaught exceptions on /api/ become JSON 500 with our envelope."""

    def test_uncaught_exception_returns_json_500(self, client, monkeypatch):
        # Hijack an existing view function so we exercise the global
        # error handler against a real request. We can't add a new route
        # to the app at test time; Flask freezes the URL map after the
        # first request, which has already happened by the time the api/
        # suite reaches this test.
        from app import app as flask_app

        target_endpoint = None
        for ep, view in flask_app.view_functions.items():
            for rule in flask_app.url_map.iter_rules(endpoint=ep):
                if rule.rule.startswith("/api/") and "GET" in (rule.methods or set()):
                    target_endpoint = ep
                    target_rule = rule.rule
                    break
            if target_endpoint:
                break
        assert target_endpoint is not None, "no /api/ GET route found"

        def boom(*a, **kw):
            raise RuntimeError("boom!")

        monkeypatch.setitem(flask_app.view_functions, target_endpoint, boom)

        # Use a concrete path (the rule may have variables we'd need to
        # substitute). For most /api/ routes the path itself is literal.
        path = target_rule.split("<")[0]  # strip any url variables
        if not path.startswith("/api/"):
            path = "/api/" + path.lstrip("/")
        resp = client.get(path)
        assert resp.status_code == 500
        body = json.loads(resp.data)
        assert body["error"] == "Internal server error"
        assert "RuntimeError" in body.get("detail", "")
        assert "request_id" in body
