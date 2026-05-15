"""Achievement system: persistent per-user milestone tracking.

State lives in the current user's per-user settings file under the
``achievement_state`` key, so milestones are isolated per user and
round-trip through that user's settings sync source like every other
per-user setting.

Categories and their tier thresholds are declared statically in
:data:`ACHIEVEMENTS`.  Counters are incremented from hook points across the
codebase (vote toggle, dataset load completion, label import, find).  The UI
polls :func:`get_full_state` to render the Achievements tab and to discover
newly-unlocked tiers (``pending_announcements``); the client ACKs each
announcement via :func:`acknowledge` so the popup doesn't fire on refresh.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

from vtsearch.auth import get_current_user
from vtsearch.settings import _ensure_user_loaded, _save_user, _settings_lock


def _load_state_dict() -> dict[str, Any]:
    """Return the current user's settings cache dict (lock held by caller)."""
    return _ensure_user_loaded(get_current_user())


def _persist_state() -> None:
    """Persist the current user's settings cache (lock held by caller)."""
    _save_user(get_current_user())


logger = logging.getLogger(__name__)

#: Tier display names, indexed 0..3.
TIER_NAMES: tuple[str, ...] = ("Bronze", "Silver", "Gold", "Platinum")

#: Achievement category definitions.  Each tier list MUST be ascending and
#: aligned with :data:`TIER_NAMES`.
ACHIEVEMENTS: list[dict[str, Any]] = [
    {
        "id": "datasets_loaded",
        "name": "Datasets Loaded",
        "description": "Load datasets you imported. Demos and synthetic don't count.",
        "icon": "cubes",
        "tiers": [1, 10, 100, 1000],
    },
    {
        "id": "votes_cast",
        "name": "Votes Cast",
        "description": "Every Good or Bad you cast on a media item.",
        "icon": "checkbox-checked",
        "tiers": [100, 1000, 10000, 100000],
    },
    {
        "id": "detectors_trained",
        "name": "Detectors Trained",
        "description": "Each detector you trained — counted on its first vote.",
        "icon": "graduation",
        "tiers": [2, 20, 200, 2000],
    },
    {
        "id": "detectors_imported",
        "name": "Detectors Imported",
        "description": "Detectors built up from imported labels.",
        "icon": "robot",
        "tiers": [1, 10, 100, 1000],
    },
    {
        "id": "find_media",
        "name": "Find Media",
        "description": "Media items scored by Find (GUI and CLI combined).",
        "icon": "search",
        "tiers": [200, 2000, 20000, 200000],
    },
    {
        "id": "days_active",
        "name": "Days Active",
        "description": "Distinct UTC calendar days on which you cast at least one vote.",
        "icon": "lightning",
        "tiers": [2, 20, 200, 2000],
    },
    {
        "id": "media_types_touched",
        "name": "Media Types Touched",
        "description": "Distinct media types you've voted on (audio, image, text, video, document).",
        "icon": "palette",
        "tiers": [2, 3, 4, 5],
    },
    {
        "id": "vote_streak",
        "name": "Marathoner",
        "description": "Longest run of consecutive votes with at most a 10-minute gap between any two.",
        "icon": "flask",
        "tiers": [200, 400, 600, 1000],
    },
    {
        "id": "hours_voted",
        "name": "Around the Clock",
        "description": "Distinct hours of the day (UTC, 0-23) in which you've cast at least one vote.",
        "icon": "steering-wheel",
        "tiers": [6, 12, 20, 24],
    },
    {
        "id": "docs_read",
        "name": "Readme Reader",
        "description": (
            "Find the code phrase at the bottom of each documentation page and "
            "paste it in to claim credit. Four docs hide a phrase; Platinum is "
            "all four."
        ),
        "icon": "file-text",
        "tiers": [1, 2, 3, 4],
    },
]

_ACH_BY_ID: dict[str, dict[str, Any]] = {a["id"]: a for a in ACHIEVEMENTS}

#: Importer names whose successful dataset loads do NOT count toward
#: ``datasets_loaded``.  These are the demo/synthetic data paths users
#: don't have ownership over.
EXCLUDED_DATASET_IMPORTERS: frozenset[str] = frozenset({"demo", "synthetic"})

#: Maximum gap (in seconds) between two consecutive votes that still keeps the
#: Marathoner streak alive.  Anything strictly greater than this resets the
#: streak counter to 1 on the next vote.
STREAK_GAP_SECONDS: float = 10 * 60

#: Readme Reader docs.  Each entry pairs a doc with the code phrase printed at
#: the bottom of it.  The phrase is matched server-side: the user pastes their
#: guess and we compare it to ``_DOC_HASHES`` (SHA-256 of the normalised
#: phrase).  Phrases are kept in source so the test suite can verify each doc's
#: footer is in sync, but :func:`get_full_state` never returns them to the
#: client — only the per-doc read state.
#:
#: ``path`` is repo-relative so the docs route can stream the raw markdown.
_DOCS_RAW: list[dict[str, str]] = [
    {
        "id": "readme",
        "name": "README",
        "path": "README.md",
        "phrase": "all aboard the embedding express",
    },
    {
        "id": "user_guide",
        "name": "User Guide",
        "path": "docs/USER_GUIDE.md",
        "phrase": "label like nobody's watching",
    },
    {
        "id": "cli",
        "name": "CLI",
        "path": "docs/CLI.md",
        "phrase": "command palette unlocked",
    },
    {
        "id": "api",
        "name": "HTTP API",
        "path": "docs/API.md",
        "phrase": "json all the way down",
    },
]


def _normalize_phrase(phrase: str) -> str:
    """Lower-case + collapse internal whitespace + strip; the canonical form
    used for hashing and matching.  Tolerant to copy-paste artefacts (extra
    spaces, trailing newline, accidental capitalisation)."""
    return " ".join(phrase.lower().split())


def _hash_phrase(phrase: str) -> str:
    return hashlib.sha256(_normalize_phrase(phrase).encode("utf-8")).hexdigest()


#: ``doc_id → sha256(normalised phrase)``.  Precomputed at import time.
_DOC_HASHES: dict[str, str] = {d["id"]: _hash_phrase(d["phrase"]) for d in _DOCS_RAW}

#: Public, phrase-free copy of the doc list for serialisation.
DOCS: list[dict[str, str]] = [{"id": d["id"], "name": d["name"], "path": d["path"]} for d in _DOCS_RAW]
_DOC_BY_ID: dict[str, dict[str, str]] = {d["id"]: d for d in DOCS}


def _ensure_state(settings: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable ``achievement_state`` sub-dict inside *settings*.

    Initializes missing keys in place so callers don't have to.  Must be
    called while holding ``_settings_lock``.
    """
    state = settings.get("achievement_state")
    if not isinstance(state, dict):
        state = {}
        settings["achievement_state"] = state
    counters = state.setdefault("counters", {})
    announced = state.setdefault("announced", {})
    state.setdefault("trained_detector_ids", [])
    state.setdefault("imported_detector_ids", [])
    state.setdefault("days_seen", [])
    state.setdefault("media_types_seen", [])
    state.setdefault("hours_seen", [])
    state.setdefault("docs_read_ids", [])
    if not isinstance(state.get("last_vote_ts"), (int, float)):
        state["last_vote_ts"] = 0.0
    if not isinstance(state.get("current_streak"), int):
        state["current_streak"] = 0
    for a in ACHIEVEMENTS:
        cid = a["id"]
        if cid not in counters or not isinstance(counters[cid], int):
            counters[cid] = 0
        if cid not in announced or not isinstance(announced[cid], int):
            announced[cid] = -1
    return state


def _current_tier_idx(category_id: str, counter: int) -> int:
    """Highest tier index (0..3) reached by *counter*, or -1 for none."""
    tiers = _ACH_BY_ID[category_id]["tiers"]
    idx = -1
    for i, threshold in enumerate(tiers):
        if counter >= threshold:
            idx = i
    return idx


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------


def record_vote(
    detector_id: str = "",
    media_type: str = "",
    *,
    now: float | None = None,
) -> None:
    """Record one user vote and credit every vote-driven achievement.

    Credits ``votes_cast`` always, ``detectors_trained`` on a detector's first
    vote, ``media_types_touched`` on a media type's first vote, ``days_active``
    on the first vote of each UTC calendar day, ``hours_voted`` on the first
    vote within each UTC hour-of-day bucket, and updates the ``vote_streak``
    watermark (longest run of consecutive votes with gaps ≤ 10 minutes).

    Args:
        detector_id: ID of the detector the vote belongs to.  Empty string
            (no active detector) skips the training credit.
        media_type: Canonical media type id (``"audio"``/``"image"``/etc.)
            for the item being voted on.  Empty string skips the
            ``media_types_touched`` credit.
        now: Override the current unix timestamp (seconds since epoch); only
            used by tests.  Default uses :func:`time.time`.
    """
    ts = time.time() if now is None else float(now)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    hour = dt.hour

    with _settings_lock:
        s = _load_state_dict()
        state = _ensure_state(s)
        counters = state["counters"]

        counters["votes_cast"] += 1

        if detector_id:
            trained = state["trained_detector_ids"]
            if detector_id not in trained:
                trained.append(detector_id)
                counters["detectors_trained"] += 1

        if media_type:
            seen_types = state["media_types_seen"]
            if media_type not in seen_types:
                seen_types.append(media_type)
                counters["media_types_touched"] += 1

        days_seen = state["days_seen"]
        if date_str not in days_seen:
            days_seen.append(date_str)
            counters["days_active"] += 1

        hours_seen = state["hours_seen"]
        if hour not in hours_seen:
            hours_seen.append(hour)
            counters["hours_voted"] += 1

        last_ts = float(state.get("last_vote_ts") or 0.0)
        if last_ts <= 0.0 or (ts - last_ts) > STREAK_GAP_SECONDS:
            current = 1
        else:
            current = int(state.get("current_streak") or 0) + 1
        state["current_streak"] = current
        state["last_vote_ts"] = ts
        if current > counters["vote_streak"]:
            counters["vote_streak"] = current

        _persist_state()


def record_dataset_load(importer_name: str) -> None:
    """Record one dataset load (skipping demos/synthetic)."""
    if importer_name in EXCLUDED_DATASET_IMPORTERS:
        return
    with _settings_lock:
        s = _load_state_dict()
        state = _ensure_state(s)
        state["counters"]["datasets_loaded"] += 1
        _persist_state()


def record_detector_import(detector_id: str) -> None:
    """Record one detector receiving imported labels.  Dedupes by detector_id."""
    if not detector_id:
        return
    with _settings_lock:
        s = _load_state_dict()
        state = _ensure_state(s)
        imported = state["imported_detector_ids"]
        if detector_id in imported:
            return
        imported.append(detector_id)
        state["counters"]["detectors_imported"] += 1
        _persist_state()


def record_find(n_scored: int) -> None:
    """Record *n_scored* media items processed by a Find operation."""
    if n_scored <= 0:
        return
    with _settings_lock:
        s = _load_state_dict()
        state = _ensure_state(s)
        state["counters"]["find_media"] += int(n_scored)
        _persist_state()


def record_doc_phrase(phrase: str) -> dict[str, Any]:
    """Check a user-submitted phrase against every doc and credit on match.

    The match is case-insensitive and whitespace-normalised — anything that
    survives :func:`_normalize_phrase` and SHA-256-hashes to a registered doc
    counts.  A phrase that matches a doc already in ``docs_read_ids`` is
    reported as ``already_read`` (idempotent, no double-credit).

    Returns a result dict with:

    - ``matched`` (bool): whether the phrase corresponds to a known doc.
    - ``doc_id`` / ``doc_name`` (str | None): identifying the matched doc.
    - ``already_read`` (bool): True when the doc was previously credited.
    """
    h = _hash_phrase(phrase)
    matched_id: str | None = None
    for doc_id, doc_hash in _DOC_HASHES.items():
        if h == doc_hash:
            matched_id = doc_id
            break

    if matched_id is None:
        return {"matched": False, "doc_id": None, "doc_name": None, "already_read": False}

    doc = _DOC_BY_ID[matched_id]
    with _settings_lock:
        s = _load_state_dict()
        state = _ensure_state(s)
        read_ids = state["docs_read_ids"]
        if matched_id in read_ids:
            return {
                "matched": True,
                "doc_id": matched_id,
                "doc_name": doc["name"],
                "already_read": True,
            }
        read_ids.append(matched_id)
        state["counters"]["docs_read"] = len(read_ids)
        _persist_state()

    return {
        "matched": True,
        "doc_id": matched_id,
        "doc_name": doc["name"],
        "already_read": False,
    }


# ---------------------------------------------------------------------------
# Read-only API
# ---------------------------------------------------------------------------


def get_full_state() -> dict[str, Any]:
    """Return the achievement state shaped for the frontend.

    The shape is::

        {
            "tier_names": ["Bronze", "Silver", "Gold", "Platinum"],
            "achievements": [
                {
                    "id": "...",
                    "name": "...",
                    "description": "...",
                    "icon": "...",
                    "tiers": [..., ..., ..., ...],
                    "counter": <int>,
                    "tier_idx": <int, -1 = locked>,
                    "next_threshold": <int | null>,
                },
                ...
            ],
            "pending_announcements": [
                {
                    "id": "...",
                    "name": "...",
                    "icon": "...",
                    "tier_idx": <int>,
                    "tier_name": "Bronze" | ...,
                    "threshold": <int>,
                },
                ...
            ],
        }
    """
    with _settings_lock:
        s = _load_state_dict()
        state = _ensure_state(s)
        achievements: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for a in ACHIEVEMENTS:
            cid = a["id"]
            counter = int(state["counters"].get(cid, 0))
            tier_idx = _current_tier_idx(cid, counter)
            announced_idx = int(state["announced"].get(cid, -1))
            next_threshold: int | None = a["tiers"][tier_idx + 1] if tier_idx + 1 < len(a["tiers"]) else None
            achievements.append(
                {
                    "id": cid,
                    "name": a["name"],
                    "description": a["description"],
                    "icon": a["icon"],
                    "tiers": list(a["tiers"]),
                    "counter": counter,
                    "tier_idx": tier_idx,
                    "next_threshold": next_threshold,
                }
            )
            for i in range(announced_idx + 1, tier_idx + 1):
                pending.append(
                    {
                        "id": cid,
                        "name": a["name"],
                        "icon": a["icon"],
                        "tier_idx": i,
                        "tier_name": TIER_NAMES[i],
                        "threshold": a["tiers"][i],
                    }
                )
        read_ids = set(state.get("docs_read_ids", []))
        docs = [{"id": d["id"], "name": d["name"], "path": d["path"], "read": d["id"] in read_ids} for d in DOCS]
        return {
            "tier_names": list(TIER_NAMES),
            "achievements": achievements,
            "pending_announcements": pending,
            "docs": docs,
        }


def acknowledge(category_id: str, tier_idx: int) -> bool:
    """Mark a tier as announced so it isn't popped again.

    Returns True if the announced index advanced, False otherwise.
    """
    if category_id not in _ACH_BY_ID:
        return False
    if tier_idx < 0 or tier_idx >= len(TIER_NAMES):
        return False
    with _settings_lock:
        s = _load_state_dict()
        state = _ensure_state(s)
        prev = int(state["announced"].get(category_id, -1))
        if tier_idx <= prev:
            return False
        state["announced"][category_id] = tier_idx
        _persist_state()
        return True
