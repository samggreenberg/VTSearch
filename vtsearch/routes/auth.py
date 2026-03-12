"""Authentication status routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_login_provider

auth_bp = Blueprint("auth", __name__)


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
