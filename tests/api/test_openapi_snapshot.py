"""Guards on the post-processed OpenAPI spec served at ``/api/openapi.json``.

The spec is the source of the checked-in ``frontend/openapi.json`` snapshot
that ``./run-tests.sh`` diffs against. flask-smorest derives the 422 response
component name and description from ``http.HTTPStatus(422)``, which Python 3.13
renamed from ``UNPROCESSABLE_ENTITY`` / "Unprocessable Entity" to the RFC 9110
spelling ``UNPROCESSABLE_CONTENT`` / "Unprocessable Content". Without
normalization the regenerated snapshot drifts purely because of the interpreter
version. These tests pin the canonical form so the drift guard is stable.
"""

from __future__ import annotations

import app as app_module  # noqa: F401  (triggers conftest side effects)
from vtsearch.openapi_postprocess import normalize_unprocessable_response


class TestUnprocessableResponseNormalization:
    """The 422 response component is pinned regardless of Python version."""

    def test_spec_uses_canonical_422_name(self, client):
        spec = client.get("/api/openapi.json").get_json()
        responses = spec["components"]["responses"]
        assert "UNPROCESSABLE_CONTENT" in responses
        assert "UNPROCESSABLE_ENTITY" not in responses
        assert responses["UNPROCESSABLE_CONTENT"]["description"] == "Unprocessable Content"

    def test_no_refs_point_at_legacy_name(self, client):
        spec = client.get("/api/openapi.json").get_json()
        legacy_ref = "#/components/responses/UNPROCESSABLE_ENTITY"
        assert legacy_ref not in _all_refs(spec)

    def test_normalizes_legacy_python311_spec(self):
        """A spec emitted by flask-smorest on Python < 3.13 is rewritten."""
        spec = {
            "components": {
                "responses": {
                    "UNPROCESSABLE_ENTITY": {
                        "description": "Unprocessable Entity",
                        "content": {},
                    }
                }
            },
            "paths": {
                "/api/thing": {"post": {"responses": {"422": {"$ref": "#/components/responses/UNPROCESSABLE_ENTITY"}}}}
            },
        }

        normalize_unprocessable_response(spec)

        responses = spec["components"]["responses"]
        assert set(responses) == {"UNPROCESSABLE_CONTENT"}
        assert responses["UNPROCESSABLE_CONTENT"]["description"] == "Unprocessable Content"
        ref = spec["paths"]["/api/thing"]["post"]["responses"]["422"]["$ref"]
        assert ref == "#/components/responses/UNPROCESSABLE_CONTENT"

    def test_idempotent_on_canonical_spec(self):
        """Running against an already-canonical spec changes nothing."""
        spec = {
            "components": {
                "responses": {
                    "UNPROCESSABLE_CONTENT": {
                        "description": "Unprocessable Content",
                        "content": {},
                    }
                }
            },
        }

        normalize_unprocessable_response(spec)

        assert set(spec["components"]["responses"]) == {"UNPROCESSABLE_CONTENT"}
        assert spec["components"]["responses"]["UNPROCESSABLE_CONTENT"]["description"] == "Unprocessable Content"


def _all_refs(node) -> set[str]:
    refs: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                refs.add(value)
            else:
                refs |= _all_refs(value)
    elif isinstance(node, list):
        for item in node:
            refs |= _all_refs(item)
    return refs
