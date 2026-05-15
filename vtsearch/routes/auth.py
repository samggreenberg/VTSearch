"""Authentication status and login/logout routes.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

from flask import request, session
from flask_smorest import Blueprint, abort

from vtsearch.auth import TrivialLoginProvider, get_login_provider
from vtsearch.schemas.auth import AuthStatusSchema, LoginRequestSchema

auth_bp = Blueprint(
    "auth",
    __name__,
    description="Authentication status, login, and logout.",
)


def _trivial_or_400() -> TrivialLoginProvider:
    """Return the active provider if it's the trivial one, else 400."""
    provider = get_login_provider()
    if not isinstance(provider, TrivialLoginProvider):
        abort(400, message="Login/logout not supported by the active provider")
    return provider


@auth_bp.route("/api/auth/status", methods=["GET"])
@auth_bp.response(200, AuthStatusSchema)
def auth_status():
    """Return the current authentication state."""
    provider = get_login_provider()
    return provider.status_dict(request)


@auth_bp.route("/api/auth/login", methods=["POST"])
@auth_bp.arguments(LoginRequestSchema)
@auth_bp.response(200, AuthStatusSchema)
@auth_bp.alt_response(400, description="Active login provider does not support login.")
def auth_login(body: dict):
    """Set the session username (trivial provider only)."""
    provider = _trivial_or_400()
    session[TrivialLoginProvider._COOKIE_KEY] = body["username"]
    return provider.status_dict(request)


@auth_bp.route("/api/auth/logout", methods=["POST"])
@auth_bp.response(200, AuthStatusSchema)
@auth_bp.alt_response(400, description="Active login provider does not support logout.")
def auth_logout():
    """Clear the session username (trivial provider only)."""
    provider = _trivial_or_400()
    session.pop(TrivialLoginProvider._COOKIE_KEY, None)
    return provider.status_dict(request)
