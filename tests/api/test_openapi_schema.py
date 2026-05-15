"""Tests for the auto-generated OpenAPI 3.0 schema.

Covers both the generator (`vtsearch.openapi.generate_openapi_spec`) and
the ``/openapi.json`` endpoint that serves it.
"""

from __future__ import annotations

from vtsearch.openapi import _flask_rule_to_openapi_path, generate_openapi_spec


class TestRuleConversion:
    def test_plain_path_passes_through(self):
        path, params = _flask_rule_to_openapi_path("/api/version")
        assert path == "/api/version"
        assert params == []

    def test_int_parameter(self):
        path, params = _flask_rule_to_openapi_path("/api/foo/<int:id>")
        assert path == "/api/foo/{id}"
        assert len(params) == 1
        assert params[0]["name"] == "id"
        assert params[0]["in"] == "path"
        assert params[0]["required"] is True
        assert params[0]["schema"]["type"] == "integer"

    def test_string_default(self):
        path, params = _flask_rule_to_openapi_path("/api/foo/<name>")
        assert path == "/api/foo/{name}"
        assert params[0]["schema"]["type"] == "string"

    def test_multiple_parameters(self):
        path, params = _flask_rule_to_openapi_path("/api/<int:x>/<string:y>")
        assert path == "/api/{x}/{y}"
        assert [p["name"] for p in params] == ["x", "y"]


class TestSpecGeneration:
    def test_openapi_version(self, client):
        spec = generate_openapi_spec(client.application)
        assert spec["openapi"].startswith("3.0")

    def test_info_block(self, client):
        spec = generate_openapi_spec(client.application, title="X", version="9.9")
        assert spec["info"]["title"] == "X"
        assert spec["info"]["version"] == "9.9"

    def test_includes_known_routes(self, client):
        spec = generate_openapi_spec(client.application)
        paths = spec["paths"]
        assert "/api/version" in paths
        # Detector endpoints are namespaced under /api/detectors/...
        assert any(p.startswith("/api/detectors") for p in paths), "detector routes missing"

    def test_skips_static_route(self, client):
        spec = generate_openapi_spec(client.application)
        # Flask's static rule (/static/<path:filename>) is skipped.
        assert "/static/{filename}" not in spec["paths"]

    def test_tags_are_emitted(self, client):
        spec = generate_openapi_spec(client.application)
        assert "tags" in spec
        tag_names = {t["name"] for t in spec["tags"]}
        # At least the datasets tag should be present.
        assert "datasets" in tag_names

    def test_post_routes_have_request_body(self, client):
        spec = generate_openapi_spec(client.application)
        # Find a known POST route and check it carries a requestBody.
        found_post_with_body = False
        for path, methods in spec["paths"].items():
            if "post" in methods and "requestBody" in methods["post"]:
                found_post_with_body = True
                break
        assert found_post_with_body, "no POST operation declared a requestBody"


class TestOpenApiEndpoint:
    def test_endpoint_returns_spec(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["openapi"].startswith("3.0")
        assert "/api/version" in data["paths"]

    def test_endpoint_self_describes(self, client):
        """The /openapi.json route should itself appear in the spec it serves."""
        resp = client.get("/openapi.json")
        data = resp.get_json()
        assert "/openapi.json" in data["paths"]
