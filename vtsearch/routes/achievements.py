"""Flask routes for the Achievements API.

Endpoints
---------
GET  /api/achievements
    Return current achievement state for the UI.

POST /api/achievements/<category_id>/acknowledge
    Mark a tier as announced.  Body: ``{"tier_idx": <int>}``.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch import achievements

achievements_bp = Blueprint("achievements", __name__)


@achievements_bp.route("/api/achievements", methods=["GET"])
def get_achievements():
    """Return the full achievement state."""
    return jsonify(achievements.get_full_state())


@achievements_bp.route("/api/achievements/<category_id>/acknowledge", methods=["POST"])
def acknowledge_achievement(category_id: str):
    """Mark *category_id*'s tier as announced."""
    body = request.get_json(force=True, silent=True) or {}
    tier_idx = body.get("tier_idx")
    if not isinstance(tier_idx, int):
        return jsonify({"error": "tier_idx (int) is required"}), 400
    changed = achievements.acknowledge(category_id, tier_idx)
    return jsonify({"ok": True, "changed": changed})
