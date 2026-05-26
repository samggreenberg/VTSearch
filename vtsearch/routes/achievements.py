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

Migrated to ``flask_smorest`` so the JSON routes are described in
``/api/openapi.json``. The raw-markdown route stays undecorated (it
serves ``text/plain``, not JSON); it's still attached to the same
``Blueprint`` and Flask routes it normally, just absent from the spec.
See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

from pathlib import Path

from flask import Response
from flask_smorest import Blueprint, abort

from vtsearch import achievements
from vtsearch.schemas.achievements import (
    AchievementStateSchema,
    AcknowledgeRequestSchema,
    AcknowledgeResponseSchema,
    CheckPhraseRequestSchema,
    CheckPhraseResponseSchema,
)

achievements_bp = Blueprint(
    "achievements",
    __name__,
    description="Counters, tiers, and the Readme Reader doc set.",
)

#: Repo root resolved at import time.  ``vtsearch/routes/achievements.py``
#: lives two levels under the repo root.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@achievements_bp.route("/api/achievements", methods=["GET"])
@achievements_bp.response(200, AchievementStateSchema)
def get_achievements():
    """Return the full achievement state."""
    return achievements.get_full_state()


@achievements_bp.route("/api/achievements/<category_id>/acknowledge", methods=["POST"])
@achievements_bp.arguments(AcknowledgeRequestSchema)
@achievements_bp.response(200, AcknowledgeResponseSchema)
def acknowledge_achievement(body: dict, category_id: str):
    """Mark *category_id*'s tier as announced."""
    changed = achievements.acknowledge(category_id, body["tier_idx"])
    return {"ok": True, "changed": changed}


@achievements_bp.route("/api/achievements/check-phrase", methods=["POST"])
@achievements_bp.arguments(CheckPhraseRequestSchema)
@achievements_bp.response(200, CheckPhraseResponseSchema)
def check_phrase(body: dict):
    """Check a user-submitted phrase against the Readme Reader docs."""
    return achievements.record_doc_phrase(body["phrase"])


@achievements_bp.route("/api/achievements/docs/<doc_id>/raw", methods=["GET"])
def get_doc_raw(doc_id: str):
    """Return the raw markdown for a Readme Reader doc.

    Plain-text response, intentionally undecorated so it stays out of
    the OpenAPI spec (the spec is for JSON APIs).
    """
    doc = next((d for d in achievements.DOCS if d["id"] == doc_id), None)
    if doc is None:
        abort(404, message="unknown doc id")
    abs_path = (_REPO_ROOT / doc["path"]).resolve()
    if _REPO_ROOT not in abs_path.parents and abs_path != _REPO_ROOT:
        abort(500, message="doc resolved outside repo")
    if not abs_path.is_file():
        abort(500, message="doc file missing")
    return Response(abs_path.read_text(encoding="utf-8"), mimetype="text/plain; charset=utf-8")
