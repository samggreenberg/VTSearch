"""Tests for server-side auth enforcement (issue #2946).

The ``_enforce_auth`` before_request hook rejects unauthenticated
``/api/*`` requests with 401 when the active provider's ``enforce_auth()``
is true. Verifies:

- DefaultLoginProvider (single-user) is completely unaffected.
- TrivialLoginProvider deliberately opts out (anonymous fallback intact).
- ApiKeyLoginProvider actually gates the API: no/invalid Bearer -> 401,
  valid Bearer -> served as the mapped user.
- The auth-status/login/logout allowlist and non-API paths pass through.
- A custom provider inherits enforcement (secure by default) and fails
  closed when its checks raise.
"""

from __future__ import annotations

import hashlib
import json

from vtsearch.auth import (
    ApiKeyLoginProvider,
    LoginProvider,
    TrivialLoginProvider,
    set_login_provider,
)


def _api_key_provider(tmp_path, key: str = "sekrit-key", username: str = "alice") -> ApiKeyLoginProvider:
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text(json.dumps({hashlib.sha256(key.encode()).hexdigest(): username}))
    return ApiKeyLoginProvider(keys_file=keys_file)


class TestDefaultProviderUnaffected:
    """Single-user deployments (no --login flag) must see zero change."""

    def test_api_served_without_credentials(self, client):
        resp = client.get("/api/medias/ids")
        assert resp.status_code == 200

    def test_auth_status_reports_authenticated(self, client):
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        assert resp.get_json()["authenticated"] is True


class TestTrivialProviderOptsOut:
    """Trivial login is passwordless; its gate is UI-only by design."""

    def test_enforce_auth_is_false(self):
        assert TrivialLoginProvider().enforce_auth() is False

    def test_anonymous_api_access_still_served(self, client):
        set_login_provider(TrivialLoginProvider())
        resp = client.get("/api/medias/ids")
        assert resp.status_code == 200


class TestApiKeyEnforcement:
    """ApiKeyLoginProvider gates every /api/* path outside the allowlist."""

    def test_no_token_is_401(self, client, tmp_path):
        set_login_provider(_api_key_provider(tmp_path))
        resp = client.get("/api/medias/ids")
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["error"] == "Authentication required"
        assert data["code"] == "auth_required"
        assert resp.headers["WWW-Authenticate"] == "Bearer"

    def test_invalid_token_is_401(self, client, tmp_path):
        set_login_provider(_api_key_provider(tmp_path))
        resp = client.get("/api/medias/ids", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_valid_token_is_served(self, client, tmp_path):
        set_login_provider(_api_key_provider(tmp_path, key="k123", username="alice"))
        resp = client.get("/api/medias/ids", headers={"Authorization": "Bearer k123"})
        assert resp.status_code == 200

    def test_auth_status_exempt(self, client, tmp_path):
        set_login_provider(_api_key_provider(tmp_path))
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["provider"] == "api_key"
        assert data["authenticated"] is False

    def test_login_logout_exempt(self, client, tmp_path):
        # api_key doesn't support login/logout, but the endpoints must be
        # reachable (they answer 400, not 401) so the SPA can probe them.
        set_login_provider(_api_key_provider(tmp_path))
        assert client.post("/api/auth/login", json={"username": "x"}).status_code == 400
        assert client.post("/api/auth/logout").status_code == 400

    def test_spa_shell_not_gated(self, client, tmp_path):
        # The guard only covers /api/*. Whether "/" is 200 or 404 here
        # depends on whether the Angular bundle is built (see
        # tests/core/test_frontend.py); what matters is that it isn't 401.
        set_login_provider(_api_key_provider(tmp_path))
        resp = client.get("/")
        assert resp.status_code != 401

    def test_huggingface_endpoints_are_gated(self, client, tmp_path):
        # hf_auth_bp configures a server-wide HuggingFace token; it must sit
        # behind the same gate as the rest of the API (issue #2946).
        set_login_provider(_api_key_provider(tmp_path))
        assert client.get("/api/auth/huggingface/status").status_code == 401

    def test_unknown_api_path_is_401_not_404(self, client, tmp_path):
        # The gate runs before routing, so unauthenticated probes can't map
        # the API surface via 404-vs-405 differences.
        set_login_provider(_api_key_provider(tmp_path))
        assert client.get("/api/definitely/not/a/route").status_code == 401


class TestCustomProviderDefaults:
    """Custom providers are enforced by construction (fail closed)."""

    def test_enforce_auth_defaults_to_true(self):
        class HeaderProvider(LoginProvider):
            name = "header"

            def get_user(self, request):
                return "anonymous"

            def is_authenticated(self, request):
                return False

        assert HeaderProvider().enforce_auth() is True

    def test_unauthenticated_custom_provider_is_401(self, client):
        class HeaderProvider(LoginProvider):
            name = "header"

            def get_user(self, request):
                return request.headers.get("X-User") or "anonymous"

            def is_authenticated(self, request):
                return "X-User" in request.headers

        set_login_provider(HeaderProvider())
        assert client.get("/api/medias/ids").status_code == 401
        assert client.get("/api/medias/ids", headers={"X-User": "bob"}).status_code == 200

    def test_raising_provider_fails_closed(self, client):
        class BrokenProvider(LoginProvider):
            name = "broken"

            def get_user(self, request):
                return "anonymous"

            def is_authenticated(self, request):
                raise RuntimeError("backend down")

        set_login_provider(BrokenProvider())
        assert client.get("/api/medias/ids").status_code == 401
