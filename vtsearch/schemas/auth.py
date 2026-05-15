"""Schemas for the auth API (``/api/auth/*``).

Models the response shape of ``LoginProvider.status_dict()`` and the
request body accepted by ``POST /api/auth/login``. Provider
implementations may add extra fields to ``status_dict``, so the
response schema is open (``unknown = "include"``).
"""

from __future__ import annotations

from marshmallow import Schema, fields, pre_load, validate


class AuthStatusSchema(Schema):
    """Response for ``GET /api/auth/status`` and ``POST /api/auth/{login,logout}``.

    The four fields below are the contract every ``LoginProvider`` must
    fulfil; custom providers may add extra keys and they pass through
    untouched (``unknown = "include"``).
    """

    provider = fields.String(required=True, metadata={"description": "Active login provider name."})
    user = fields.String(
        required=True,
        metadata={"description": "Current user, or 'anonymous' when not logged in."},
    )
    authenticated = fields.Boolean(required=True)
    login_required = fields.Boolean(required=True)

    class Meta:
        unknown = "include"


class LoginRequestSchema(Schema):
    """Body for ``POST /api/auth/login`` (trivial provider only).

    Mirrors the historical ``^[A-Za-z0-9_-]{1,64}$`` validation from
    the legacy hand-rolled route. Surrounding whitespace is stripped
    before the regex runs, matching the legacy ``.strip()`` behavior.
    """

    username = fields.String(
        required=True,
        validate=validate.Regexp(
            r"^[A-Za-z0-9_-]{1,64}$",
            error="Username must be 1-64 alphanumeric/dash/underscore characters",
        ),
    )

    @pre_load
    def _strip_username(self, data, **_kwargs):
        if isinstance(data, dict):
            raw = data.get("username")
            if isinstance(raw, str):
                data = {**data, "username": raw.strip()}
        return data


__all__ = ["AuthStatusSchema", "LoginRequestSchema"]
