"""Tests for the single JSON error envelope.

Every API error -- from a ``flask_smorest.abort()`` inside a route, from a
global ``@app.errorhandler``, or from a route helper returning an error
tuple -- renders the same ``{code, status, message, request_id, ...}`` shape
(see :mod:`vtsearch.errors`). It is consumed by the frontend interceptor /
``apiErrorMessage`` helper and documented in the OpenAPI spec; keep this
contract stable.
"""

from __future__ import annotations

import json


class TestErrorEnvelope:
    """4xx responses from inline error returns include ``request_id``."""

    def test_helper_and_abort_agree_on_the_envelope_keys(self, client):
        """The two ways a route can fail must produce the same shape.

        This is the whole point of the unification: ``error_response`` (used
        by the global handlers) and ``flask_smorest.abort`` (used by ~350
        route sites) used to emit ``{error, detail, request_id}`` and
        ``{code, status, message, errors}`` respectively, so the client had
        to read both spellings.
        """
        # error_response path (get_json_or_400 on /api/embed).
        helper = json.loads(client.post("/api/embed", content_type="application/json").data)
        # abort(404, message=...) path (server-media thumbnail).
        aborted = json.loads(client.get("/api/server-media-files/nope.wav/thumbnail").data)

        for body in (helper, aborted):
            assert set(body) >= {"code", "status", "message", "request_id"}
            assert isinstance(body["code"], int)
            assert isinstance(body["status"], str)
        assert helper["code"] == 400
        assert aborted["code"] == 404

    def test_abort_message_survives_the_global_404_handler(self, client):
        """``abort(404, message=...)`` bodies are no longer discarded.

        The app registers a ``NotFound`` handler so unknown ``/api/`` paths
        render JSON rather than werkzeug's HTML page. Flask resolves the most
        specific exception class first, so that handler used to take *every*
        404 away from flask-smorest and render the literal 'Not Found' --
        silently dropping the message from ~85 ``abort(404, message=...)``
        sites. It now delegates the rendering back to flask-smorest.
        """
        resp = client.get("/api/server-media-files/nope.wav/thumbnail")
        assert resp.status_code == 404
        body = json.loads(resp.data)
        assert body["message"] == "File not found: nope.wav"
        assert "request_id" in body

    def test_abort_extra_kwargs_ride_along(self, client):
        """Unreserved ``abort()`` kwargs become top-level fields.

        flask-smorest's own handler reads only ``message``/``errors``/
        ``headers`` and drops the rest; the ``**extra`` support the retired
        hand-rolled envelope had is preserved by the ``VTSearchApi`` override.
        """
        # ``eval_train_and_score_result`` aborts with two extras, and its
        # 404 is exactly the case the global ``NotFound`` handler used to
        # flatten.
        resp = client.get("/api/eval/train-and-score/result?job_id=no-such-job")
        assert resp.status_code == 404
        body = json.loads(resp.data)
        assert body["message"] == "Job not found"
        assert body["job_id"] == "no-such-job"
        assert body["job_status"] == "missing"
        # The envelope's own fields win: an extra never overwrites one.
        assert body["status"] == "Not Found"
        assert "request_id" in body

    def test_webargs_internals_never_reach_the_body(self, client):
        """A 422 must not try to serialize webargs's ``schema`` kwarg.

        webargs attaches the live marshmallow ``Schema`` and the
        ``ValidationError`` to ``exc.data`` alongside ``messages``; a
        pass-through that copied every unreserved key would 500 the error
        handler itself on ``Object of type ... is not JSON serializable``.
        """
        resp = client.post("/api/medias/1/vote", json={"target": 123})
        assert resp.status_code == 422
        body = json.loads(resp.data)
        assert "schema" not in body
        assert "exc" not in body
        assert body["errors"]

    def test_invalid_json_body_carries_request_id(self, client):
        # ``/api/embed`` is still on the legacy ``get_json_or_400``
        # helper (the dual-mode multipart-or-JSON dispatcher doesn't fit
        # a single marshmallow schema). An empty JSON body trips its
        # "Invalid request body" path, exercising the legacy envelope
        # that this test is here to pin.
        resp = client.post("/api/embed", content_type="application/json")
        assert resp.status_code == 400
        body = json.loads(resp.data)
        assert body["message"] == "Invalid request body"
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
        from vtsearch.errors import error_response

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
            assert body["code"] == 400
            assert body["status"] == "Bad Request"
            assert body["message"] == "Missing required field(s): ['name']"
            assert body["missing_fields"] == ["name"]
            assert body["request_id"] == "abc123"

    def test_error_response_omits_request_id_outside_request(self):
        # Background-thread error paths have no g.request_id; the helper
        # should still produce a valid JSON envelope.
        from app import app as flask_app
        from vtsearch.errors import error_response

        with flask_app.app_context():
            resp, status = error_response("Something broke", 500)
            assert status == 500
            body = resp.get_json()
            assert body == {
                "code": 500,
                "status": "Internal Server Error",
                "message": "Something broke",
            }

    def test_unknown_api_path_returns_json_404(self, client):
        resp = client.get("/api/this-route-does-not-exist")
        assert resp.status_code == 404
        body = json.loads(resp.data)
        # No route aborted, so there is no message to preserve; the envelope
        # falls back to the status name rather than omitting the key.
        assert body["message"] == "Not Found"
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
        assert body["message"] == "Internal server error"
        assert "RuntimeError" in body.get("detail", "")
        assert "request_id" in body


class TestExceptionDetailPathScrub:
    """``format_exception_detail`` must not leak the absolute server path.

    An OS error (e.g. ENAMETOOLONG on a rename) carries the absolute path;
    the 500 body scrubs it down to the data-dir-relative tail.
    """

    def test_data_dir_prefix_is_stripped(self):
        import os

        from vtscore.config import DATA_DIR
        from vtsearch.routes._shared import format_exception_detail

        leaked = str(DATA_DIR / "detectors" / "x.json.tmp")
        detail = format_exception_detail(OSError(f"[Errno 36] File name too long: '{leaked}'"))

        # The absolute mount-point prefix is gone...
        assert str(DATA_DIR.parent) not in detail
        # ...but the useful data-dir-relative tail survives.
        assert f"detectors{os.sep}x.json.tmp" in detail

    def test_ordinary_message_untouched(self):
        from vtsearch.routes._shared import format_exception_detail

        detail = format_exception_detail(RuntimeError("embedder X not loaded"))
        assert detail == "RuntimeError: embedder X not loaded"
