"""HuggingFace OAuth ("Sign in with HuggingFace") routes.

These authenticate VTSearch's *outbound* requests to the HuggingFace Hub so
gated demo datasets and gated model weights download successfully.  This is
separate from VTSearch's own user auth (:mod:`vtsearch.auth`): signing in here
hands the *server* a Hub token, it does not change who the VTSearch user is.

The obtained access token is held in the process-scoped, in-memory store in
:mod:`vtscore.security.hf_auth`; the download and embedder code read it from
there.  Nothing is written to disk.

Setup (one-time, by whoever runs the server): register an OAuth app at
https://huggingface.co/settings/applications/new with the redirect URI
``<base-url>/api/auth/huggingface/callback`` and the ``read-repos`` scope, then
expose the credentials via env vars:

* ``HF_OAUTH_CLIENT_ID``     - required to enable the flow.
* ``HF_OAUTH_CLIENT_SECRET`` - for a confidential client (omit for a public
  PKCE-only client).
* ``HF_OAUTH_REDIRECT_URI``  - optional override when the public URL differs
  from what Flask sees (e.g. behind a TLS-terminating proxy).
* ``HF_OAUTH_SCOPES``        - optional, defaults to ``openid profile read-repos``.

This is a plain Flask blueprint (not flask-smorest): the callback is a browser
redirect target that returns a 302, not a typed JSON endpoint, so keeping the
whole group off the OpenAPI surface avoids schema noise for no benefit.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from urllib.parse import urlencode, urljoin

import requests
from flask import Blueprint, current_app, jsonify, redirect, request, session

from vtscore.security.hf_auth import clear_credential, get_status, set_credential

hf_auth_bp = Blueprint("hf_auth", __name__)

_AUTHORIZE_URL = "https://huggingface.co/oauth/authorize"
_TOKEN_URL = "https://huggingface.co/oauth/token"
_USERINFO_URL = "https://huggingface.co/oauth/userinfo"
_DEFAULT_SCOPES = "openid profile read-repos"

# Flask-session keys holding the in-flight OAuth handshake.
_SESS_STATE = "hf_oauth_state"
_SESS_VERIFIER = "hf_oauth_verifier"
_SESS_REDIRECT = "hf_oauth_redirect_uri"


def _client_id() -> str:
    return (os.environ.get("HF_OAUTH_CLIENT_ID") or "").strip()


def _client_secret() -> str:
    return (os.environ.get("HF_OAUTH_CLIENT_SECRET") or "").strip()


def _scopes() -> str:
    return (os.environ.get("HF_OAUTH_SCOPES") or _DEFAULT_SCOPES).strip()


def _is_configured() -> bool:
    """The flow is available iff an OAuth client id has been provided."""
    return bool(_client_id())


def _redirect_uri() -> str:
    """The callback URL HuggingFace redirects back to after authorization.

    Prefers the explicit ``HF_OAUTH_REDIRECT_URI`` override (needed behind a
    proxy that rewrites scheme/host); otherwise derives it from the incoming
    request so local runs work with zero extra config.  Must match a redirect
    URI registered on the HuggingFace OAuth app exactly.
    """
    override = (os.environ.get("HF_OAUTH_REDIRECT_URI") or "").strip()
    if override:
        return override
    return urljoin(request.url_root, "api/auth/huggingface/callback")


def _pkce_pair() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for a PKCE S256 handshake."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


@hf_auth_bp.route("/api/auth/huggingface/status", methods=["GET"])
def hf_status():
    """Report whether the flow is configured and whether we're signed in."""
    payload = {"configured": _is_configured(), **get_status()}
    return jsonify(payload)


@hf_auth_bp.route("/api/auth/huggingface/login", methods=["GET"])
def hf_login():
    """Begin the OAuth handshake; return the HuggingFace authorize URL.

    The frontend navigates the browser to ``authorize_url``.  When the flow
    isn't configured we return ``{configured: false}`` (200) so the UI can show
    setup guidance instead of a dead button.
    """
    if not _is_configured():
        return jsonify({"configured": False})

    state = secrets.token_urlsafe(32)
    verifier, challenge = _pkce_pair()
    redirect_uri = _redirect_uri()

    session[_SESS_STATE] = state
    session[_SESS_VERIFIER] = verifier
    session[_SESS_REDIRECT] = redirect_uri

    params = {
        "client_id": _client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _scopes(),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return jsonify({"configured": True, "authorize_url": f"{_AUTHORIZE_URL}?{urlencode(params)}"})


def _finish_redirect(status: str, reason: str = "") -> object:
    """Clear handshake state and bounce the browser back to the app root."""
    for key in (_SESS_STATE, _SESS_VERIFIER, _SESS_REDIRECT):
        session.pop(key, None)
    params = {"hf_auth": status}
    if reason:
        params["hf_auth_reason"] = reason
    return redirect(f"/?{urlencode(params)}")


@hf_auth_bp.route("/api/auth/huggingface/callback", methods=["GET"])
def hf_callback():
    """OAuth redirect target: exchange the code for a token, then store it."""
    if request.args.get("error"):
        return _finish_redirect("error", request.args.get("error_description") or request.args.get("error") or "denied")

    code = request.args.get("code", "")
    state = request.args.get("state", "")
    expected_state = session.get(_SESS_STATE)
    verifier = session.get(_SESS_VERIFIER)
    redirect_uri = session.get(_SESS_REDIRECT) or _redirect_uri()

    if not code or not state or not expected_state or not secrets.compare_digest(state, expected_state):
        return _finish_redirect("error", "invalid_state")

    try:
        token_data = _exchange_code(code, redirect_uri, verifier or "")
        access_token = token_data.get("access_token")
        if not access_token:
            return _finish_redirect("error", "no_token")
        username = _fetch_username(access_token)
        expires_in = token_data.get("expires_in")
        expires_at = time.time() + float(expires_in) if expires_in else None
        set_credential(
            access_token,
            username=username,
            expires_at=expires_at,
            scopes=token_data.get("scope", "") or "",
        )
    except requests.RequestException:
        current_app.logger.exception("HuggingFace OAuth token exchange failed")
        return _finish_redirect("error", "exchange_failed")

    return _finish_redirect("success")


@hf_auth_bp.route("/api/auth/huggingface/logout", methods=["POST"])
def hf_logout():
    """Forget the stored HuggingFace credential (sign out)."""
    clear_credential()
    return jsonify({"ok": True})


def _exchange_code(code: str, redirect_uri: str, verifier: str) -> dict:
    """POST the authorization code to HuggingFace and return the token JSON."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": _client_id(),
        "code_verifier": verifier,
    }
    secret = _client_secret()
    if secret:
        data["client_secret"] = secret
    resp = requests.post(
        _TOKEN_URL,
        data=data,
        headers={"Accept": "application/json"},
        timeout=(10, 30),
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_username(access_token: str) -> str:
    """Best-effort lookup of the signed-in HuggingFace username.

    Failures here are non-fatal: the token is still valid for downloads, we
    just fall back to an empty display name.
    """
    try:
        resp = requests.get(
            _USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout=(10, 30),
        )
        resp.raise_for_status()
        info = resp.json()
    except (requests.RequestException, ValueError):
        return ""
    return info.get("preferred_username") or info.get("name") or ""
