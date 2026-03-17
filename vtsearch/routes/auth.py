"""Authentication status and login/logout routes."""

from __future__ import annotations

import re

from flask import Blueprint, jsonify, request, session

from vtsearch.auth import TrivialLoginProvider, get_login_provider

auth_bp = Blueprint("auth", __name__)

# Username constraints for the trivial provider.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@auth_bp.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Return the current authentication state.

    Response shape::

        {
            "provider": "default",
            "user": "default",
            "authenticated": true,
            "login_required": false
        }
    """
    provider = get_login_provider()
    return jsonify(provider.status_dict(request))


@auth_bp.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Set the session username (trivial provider only).

    Expects JSON ``{"username": "alice"}``.  Returns the updated auth
    status dict on success.
    """
    provider = get_login_provider()
    if not isinstance(provider, TrivialLoginProvider):
        return jsonify({"error": "Login not supported by the active provider"}), 400

    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    if not username or not _USERNAME_RE.match(username):
        return jsonify({"error": "Username must be 1-64 alphanumeric/dash/underscore characters"}), 400

    session[TrivialLoginProvider._COOKIE_KEY] = username
    return jsonify(provider.status_dict(request))


@auth_bp.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """Clear the session username (trivial provider only)."""
    provider = get_login_provider()
    if not isinstance(provider, TrivialLoginProvider):
        return jsonify({"error": "Logout not supported by the active provider"}), 400

    session.pop(TrivialLoginProvider._COOKIE_KEY, None)
    return jsonify(provider.status_dict(request))
