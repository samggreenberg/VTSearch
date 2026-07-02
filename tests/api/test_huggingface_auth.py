"""HuggingFace OAuth routes (``/api/auth/huggingface/*``).

The external HuggingFace HTTP calls (token exchange + userinfo) are stubbed, so
these exercise our handshake bookkeeping: configuration gating, CSRF-state
validation, credential storage, and the browser redirects.
"""

import requests

from vtsearch.routes import auth_huggingface as hf_mod
from vtscore.security.hf_auth import is_authenticated, set_credential


class TestStatus:
    def test_unconfigured_anonymous(self, client, monkeypatch):
        monkeypatch.delenv("HF_OAUTH_CLIENT_ID", raising=False)
        resp = client.get("/api/auth/huggingface/status")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["configured"] is False
        assert body["authenticated"] is False

    def test_configured_and_signed_in(self, client, monkeypatch):
        monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "cid")
        set_credential("tok", username="alice", scopes="read-repos")
        body = client.get("/api/auth/huggingface/status").get_json()
        assert body["configured"] is True
        assert body["authenticated"] is True
        assert body["username"] == "alice"


class TestLogin:
    def test_unconfigured_returns_configured_false(self, client, monkeypatch):
        monkeypatch.delenv("HF_OAUTH_CLIENT_ID", raising=False)
        body = client.get("/api/auth/huggingface/login").get_json()
        assert body == {"configured": False}

    def test_configured_returns_authorize_url(self, client, monkeypatch):
        monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "my-client-id")
        body = client.get("/api/auth/huggingface/login").get_json()
        assert body["configured"] is True
        url = body["authorize_url"]
        assert url.startswith("https://huggingface.co/oauth/authorize?")
        assert "client_id=my-client-id" in url
        assert "code_challenge_method=S256" in url
        # The CSRF state was stashed in the session for the callback to check.
        with client.session_transaction() as sess:
            assert sess.get(hf_mod._SESS_STATE)
            assert sess.get(hf_mod._SESS_VERIFIER)


class TestCallback:
    def test_provider_error_redirects_to_error(self, client):
        resp = client.get("/api/auth/huggingface/callback?error=access_denied")
        assert resp.status_code == 302
        assert "hf_auth=error" in resp.headers["Location"]

    def test_state_mismatch_redirects_to_error(self, client):
        with client.session_transaction() as sess:
            sess[hf_mod._SESS_STATE] = "expected"
            sess[hf_mod._SESS_VERIFIER] = "v"
        resp = client.get("/api/auth/huggingface/callback?code=c&state=wrong")
        assert resp.status_code == 302
        assert "hf_auth=error" in resp.headers["Location"]
        assert "invalid_state" in resp.headers["Location"]
        assert is_authenticated() is False

    def test_success_stores_credential(self, client, monkeypatch):
        monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "cid")
        monkeypatch.setattr(
            hf_mod,
            "_exchange_code",
            lambda code, redirect_uri, verifier: {
                "access_token": "AT",
                "expires_in": 3600,
                "scope": "read-repos",
            },
        )
        monkeypatch.setattr(hf_mod, "_fetch_username", lambda token: "alice")

        with client.session_transaction() as sess:
            sess[hf_mod._SESS_STATE] = "st"
            sess[hf_mod._SESS_VERIFIER] = "ver"
            sess[hf_mod._SESS_REDIRECT] = "http://localhost/api/auth/huggingface/callback"

        resp = client.get("/api/auth/huggingface/callback?code=the-code&state=st")
        assert resp.status_code == 302
        assert "hf_auth=success" in resp.headers["Location"]
        assert is_authenticated() is True

    def test_exchange_failure_redirects_to_error(self, client, monkeypatch):
        monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "cid")

        def _boom(*a, **k):
            raise requests.RequestException("nope")

        monkeypatch.setattr(hf_mod, "_exchange_code", _boom)

        with client.session_transaction() as sess:
            sess[hf_mod._SESS_STATE] = "st"
            sess[hf_mod._SESS_VERIFIER] = "ver"

        resp = client.get("/api/auth/huggingface/callback?code=c&state=st")
        assert resp.status_code == 302
        assert "hf_auth=error" in resp.headers["Location"]
        assert is_authenticated() is False


class TestLogout:
    def test_logout_clears_credential(self, client):
        set_credential("tok", username="alice")
        assert is_authenticated() is True
        resp = client.post("/api/auth/huggingface/logout")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert is_authenticated() is False
