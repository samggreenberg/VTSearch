"""Flask routes for the Achievements API.

Endpoints
---------
GET  /api/achievements
    Return current achievement state for the UI.

POST /api/achievements/<category_id>/acknowledge
    Mark a tier as announced.  Body: ``{"tier_idx": <int>}``.

POST /api/achievements/check-phrase
    Match a user-submitted phrase against the Readme Reader doc set.
    Body: ``{"phrase": "<string>"}``.

GET  /api/achievements/docs/<doc_id>/raw
    Stream the raw markdown of a Readme Reader doc, so the frontend can link
    to it directly without requiring internet access to GitHub.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from vtsearch import achievements

achievements_bp = Blueprint("achievements", __name__)

#: Repo root resolved at import time.  ``vtsearch/routes/achievements.py``
#: lives two levels under the repo root.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


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


@achievements_bp.route("/api/achievements/check-phrase", methods=["POST"])
def check_phrase():
    """Check a user-submitted phrase against the Readme Reader docs."""
    body = request.get_json(force=True, silent=True) or {}
    phrase = body.get("phrase")
    if not isinstance(phrase, str):
        return jsonify({"error": "phrase (string) is required"}), 400
    result = achievements.record_doc_phrase(phrase)
    return jsonify(result)


@achievements_bp.route("/api/achievements/docs/<doc_id>/raw", methods=["GET"])
def get_doc_raw(doc_id: str):
    """Return the raw markdown for a Readme Reader doc."""
    doc = next((d for d in achievements.DOCS if d["id"] == doc_id), None)
    if doc is None:
        return jsonify({"error": "unknown doc id"}), 404
    abs_path = (_REPO_ROOT / doc["path"]).resolve()
    if _REPO_ROOT not in abs_path.parents and abs_path != _REPO_ROOT:
        return jsonify({"error": "doc resolved outside repo"}), 500
    if not abs_path.is_file():
        return jsonify({"error": "doc file missing"}), 500
    return Response(abs_path.read_text(encoding="utf-8"), mimetype="text/plain; charset=utf-8")
