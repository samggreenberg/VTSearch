"""Shared helpers for Flask route handlers."""

from __future__ import annotations

from flask import jsonify, request


def get_json_or_400():
    """Parse the request body as JSON, returning a 400 response on failure.

    Returns:
        The parsed JSON data (usually a dict) on success, or a
        ``(response, 400)`` tuple that can be returned directly from a
        Flask view when the body is missing or unparseable.

    Usage in a route::

        data = get_json_or_400()
        if not isinstance(data, dict):
            return data  # it's already a (jsonify(...), 400) tuple
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid request body"}), 400
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400
    return data
