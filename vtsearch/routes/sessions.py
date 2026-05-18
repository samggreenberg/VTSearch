"""Flask routes for the Recent Sessions API.

The frontend's burger-menu submenu calls ``POST /api/sessions/recent``
whenever the user enters ``/label/:ds/:det`` or ``/find/:ds/:det`` to
bump that pair's ``last_activity`` timestamp, and reads
``GET /api/sessions/recent`` to render the list.

Storage lives in the per-user settings tier under the ``recent_sessions``
key. The list is capped at :data:`MAX_RECENT_SESSIONS` entries and is
always sorted most-recent-first.

Entries whose ids no longer resolve in the dataset or detector registry
are filtered out on read so the menu never offers a dead link.
"""

from __future__ import annotations

import time
from typing import Any

from flask_smorest import Blueprint

from vtsearch import settings
from vtsearch.datasets.registry import list_datasets
from vtsearch.detectors.registry import list_detectors
from vtsearch.schemas.sessions import (
    BumpRecentSessionRequestSchema,
    RecentSessionsResponseSchema,
)

#: Maximum number of recent sessions kept per user.
MAX_RECENT_SESSIONS: int = 10

sessions_bp = Blueprint(
    "sessions",
    __name__,
    description="Recent (dataset, detector) labelling sessions per user.",
)


def _hydrate(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve registry names and drop entries pointing at deleted ids."""
    dataset_map = {d["id"]: d.get("name") or d["id"] for d in list_datasets()}
    detector_map = {d["id"]: d.get("name") or d["id"] for d in list_detectors()}

    hydrated: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ds_id = entry.get("dataset_id")
        det_id = entry.get("detector_id")
        if not isinstance(ds_id, str) or not isinstance(det_id, str):
            continue
        if ds_id not in dataset_map or det_id not in detector_map:
            continue
        try:
            last_activity = float(entry.get("last_activity") or 0.0)
        except (TypeError, ValueError):
            last_activity = 0.0
        hydrated.append(
            {
                "dataset_id": ds_id,
                "detector_id": det_id,
                "dataset_name": dataset_map[ds_id],
                "detector_name": detector_map[det_id],
                "last_activity": last_activity,
            }
        )
    return hydrated


@sessions_bp.route("/api/sessions/recent", methods=["GET"])
@sessions_bp.response(200, RecentSessionsResponseSchema)
def list_recent_sessions():
    """Return the current user's recent (dataset, detector) sessions.

    Entries whose dataset or detector id is no longer registered are
    filtered out. Most-recent first.
    """
    raw = settings.get_recent_sessions()
    return {"sessions": _hydrate(raw)}


@sessions_bp.route("/api/sessions/recent", methods=["POST"])
@sessions_bp.arguments(BumpRecentSessionRequestSchema)
@sessions_bp.response(200, RecentSessionsResponseSchema)
def bump_recent_session(body: dict):
    """Record activity on a ``(dataset_id, detector_id)`` pair.

    Upserts the pair, moves it to the front of the list, and trims
    the list to :data:`MAX_RECENT_SESSIONS` entries. The response is
    the same hydrated list as :func:`list_recent_sessions` so callers
    can update local UI in one round-trip.
    """
    dataset_id = body["dataset_id"]
    detector_id = body["detector_id"]

    raw = settings.get_recent_sessions()
    filtered = [
        e
        for e in raw
        if isinstance(e, dict) and not (e.get("dataset_id") == dataset_id and e.get("detector_id") == detector_id)
    ]
    updated: list[dict[str, Any]] = [
        {
            "dataset_id": dataset_id,
            "detector_id": detector_id,
            "last_activity": time.time(),
        },
        *filtered,
    ][:MAX_RECENT_SESSIONS]
    settings.set_recent_sessions(updated)

    return {"sessions": _hydrate(updated)}
