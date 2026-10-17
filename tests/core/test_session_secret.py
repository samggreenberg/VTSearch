"""The shipped Flask secret key is replaced wherever session cookies are trusted (#5002).

``app.py`` falls back to a constant ``secret_key`` that anyone can read in the
source. :func:`secure_session_secret` swaps it for a random per-process key at
server start when ``--login trivial`` or the HuggingFace OAuth flow is active,
and leaves single-user mode, and any operator-set key, alone.
"""

from __future__ import annotations

import pytest
from flask import Flask
from flask.sessions import SecureCookieSessionInterface

import app as app_module
from vtsearch.auth import ApiKeyLoginProvider, TrivialLoginProvider, set_login_provider
from vtsearch.auth.session_secret import DEFAULT_SECRET_KEY, secure_session_secret

_LOGGER = "vtsearch.auth.session_secret"


@pytest.fixture
def fresh_app(monkeypatch):
    """A Flask app keyed the way ``app.py`` keys one when ``VTSEARCH_SECRET_KEY`` is unset."""
    monkeypatch.delenv("HF_OAUTH_CLIENT_ID", raising=False)
    app = Flask(__name__)
    app.secret_key = DEFAULT_SECRET_KEY
    return app


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


def _forge_session_cookie(payload: dict) -> str:
    """Sign ``payload`` as a session cookie with the public default key."""
    forger = Flask("forger")
    forger.secret_key = DEFAULT_SECRET_KEY
    serializer = SecureCookieSessionInterface().get_signing_serializer(forger)
    assert serializer is not None
    return serializer.dumps(payload)


class TestReplacesTheDefault:
    def test_trivial_login(self, fresh_app, caplog):
        set_login_provider(TrivialLoginProvider())

        assert secure_session_secret(fresh_app) is True

        assert fresh_app.secret_key
        assert fresh_app.secret_key != DEFAULT_SECRET_KEY
        [warning] = _warnings(caplog)
        assert "--login trivial" in warning
        assert "VTSEARCH_SECRET_KEY" in warning
        # A multi-worker deployment has to learn from this line that it needs the variable.
        assert "single worker" in warning

    def test_huggingface_flow_under_the_default_provider(self, fresh_app, monkeypatch, caplog):
        """The HF flow keeps its OAuth ``state`` in the session under any login provider."""
        monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "cid")

        assert secure_session_secret(fresh_app) is True

        assert fresh_app.secret_key != DEFAULT_SECRET_KEY
        [warning] = _warnings(caplog)
        assert "HF_OAUTH_CLIENT_ID" in warning
        assert "--login trivial" not in warning

    def test_both_features_share_one_warning(self, fresh_app, monkeypatch, caplog):
        monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "cid")
        set_login_provider(TrivialLoginProvider())

        secure_session_secret(fresh_app)

        [warning] = _warnings(caplog)
        assert "--login trivial" in warning
        assert "HF_OAUTH_CLIENT_ID" in warning

    def test_second_call_is_a_no_op(self, fresh_app, caplog):
        set_login_provider(TrivialLoginProvider())
        secure_session_secret(fresh_app)
        first = fresh_app.secret_key

        assert secure_session_secret(fresh_app) is False

        assert fresh_app.secret_key == first
        assert len(_warnings(caplog)) == 1

    def test_each_process_gets_its_own_key(self, monkeypatch):
        monkeypatch.delenv("HF_OAUTH_CLIENT_ID", raising=False)
        set_login_provider(TrivialLoginProvider())
        keys = set()
        for _ in range(2):
            app = Flask(__name__)
            app.secret_key = DEFAULT_SECRET_KEY
            secure_session_secret(app)
            keys.add(app.secret_key)
        assert len(keys) == 2


class TestLeavesTheKeyAlone:
    def test_operator_set_key_is_used_unchanged(self, fresh_app, monkeypatch, caplog):
        fresh_app.secret_key = "operator-chosen-key"
        monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "cid")
        set_login_provider(TrivialLoginProvider())

        assert secure_session_secret(fresh_app) is False

        assert fresh_app.secret_key == "operator-chosen-key"
        assert _warnings(caplog) == []

    def test_default_provider_with_nothing_configured(self, fresh_app, caplog):
        """Single-user mode without the HF flow trusts no cookie: unchanged from before."""
        assert secure_session_secret(fresh_app) is False

        assert fresh_app.secret_key == DEFAULT_SECRET_KEY
        assert _warnings(caplog) == []

    def test_api_key_login_trusts_no_cookie(self, fresh_app, tmp_path, caplog):
        set_login_provider(ApiKeyLoginProvider(keys_file=tmp_path / "api_keys.json"))

        assert secure_session_secret(fresh_app) is False

        assert fresh_app.secret_key == DEFAULT_SECRET_KEY
        assert _warnings(caplog) == []


class TestForgedCookies:
    """The property the check exists for, end to end through the real app."""

    @pytest.fixture
    def trivial_client(self, client, monkeypatch):
        monkeypatch.delenv("HF_OAUTH_CLIENT_ID", raising=False)
        # monkeypatch restores whatever key the module-level app carried.
        monkeypatch.setattr(app_module.app, "secret_key", DEFAULT_SECRET_KEY)
        set_login_provider(TrivialLoginProvider())
        return client

    def _status(self, client, cookie: str) -> dict:
        # Through the cookie jar: the test client replaces an explicit Cookie header with the jar's.
        client.set_cookie(app_module.app.config["SESSION_COOKIE_NAME"], cookie)
        return client.get("/api/auth/status").get_json()

    def test_default_key_lets_a_forged_cookie_name_any_user(self, trivial_client):
        """The control: without the check, a cookie minted from the source is believed."""
        forged = _forge_session_cookie({TrivialLoginProvider._COOKIE_KEY: "alice"})
        assert self._status(trivial_client, forged)["user"] == "alice"

    def test_replaced_key_rejects_a_forged_cookie(self, trivial_client):
        secure_session_secret(app_module.app)
        forged = _forge_session_cookie({TrivialLoginProvider._COOKIE_KEY: "alice"})

        status = self._status(trivial_client, forged)

        assert status["user"] == "anonymous"
        assert status["authenticated"] is False


def test_initialize_server_secures_the_key_before_anything_else(monkeypatch):
    """Both ``python app.py`` and gunicorn's ``VTSEARCH_SERVER_INIT=1`` go through ``initialize_server``."""

    class _Stop(Exception):
        pass

    seen = []

    def _record(app):
        seen.append(app)
        raise _Stop

    monkeypatch.setattr(app_module, "secure_session_secret", _record)
    with pytest.raises(_Stop):
        app_module.initialize_server()
    assert seen == [app_module.app]
