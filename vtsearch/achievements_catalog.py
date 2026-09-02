"""Static achievement catalog: the declarations, none of the state machine.

This module is pure data plus the two tiny pure helpers that derive tables
from it (:func:`_normalize_phrase` / :func:`_hash_phrase`).  It imports
nothing from the rest of the app, so it can be read — and edited, when a new
achievement category or Readme Reader doc is added — without carrying the
counter/watermark logic in :mod:`vtsearch.achievements` in your head.

:mod:`vtsearch.achievements` re-exports every public name here, so callers
keep importing them from there; this split is an internal reorganisation, not
a new import surface.
"""

from __future__ import annotations

import hashlib
from typing import Any


#: Tier display names, indexed 0..3.
TIER_NAMES: tuple[str, ...] = ("Bronze", "Silver", "Gold", "Platinum")

#: Achievement category definitions.  Each tier list MUST be ascending and
#: aligned with :data:`TIER_NAMES`.
ACHIEVEMENTS: list[dict[str, Any]] = [
    {
        "id": "datasets_loaded",
        "name": "Data: Set",
        "description": "Load datasets you imported. Demos and synthetic don't count.",
        "icon": "cubes",
        "tiers": [1, 10, 100, 1000],
    },
    {
        "id": "votes_cast",
        "name": "Get Out the Vote",
        "description": "Every Good or Bad you cast on a media item.",
        "icon": "checkbox-checked",
        "tiers": [100, 1000, 10000, 100000],
    },
    {
        "id": "detectors_trained",
        "name": "Teacher's Pet",
        "description": "Each detector you trained, counted on its first vote.",
        "icon": "graduation",
        "tiers": [2, 20, 200, 2000],
    },
    {
        "id": "detectors_imported",
        "name": "Detector Collector",
        "description": "Detectors built up from imported labels.",
        "icon": "robot",
        "tiers": [1, 10, 100, 1000],
    },
    {
        "id": "find_media",
        "name": "Finders Keepers",
        "description": "Media items scored by Find (GUI and CLI combined).",
        "icon": "search",
        "tiers": [2000, 20000, 200000, 2000000],
    },
    {
        "id": "days_active",
        "name": "Your Days are Numbered",
        "description": "Distinct calendar days on which you cast at least one vote.",
        "icon": "calendar",
        "tiers": [2, 20, 200, 2000],
    },
    {
        "id": "media_types_touched",
        "name": "Multi Media",
        "description": "Distinct media types you've voted on (audio, image, text, video, document).",
        "icon": "palette",
        "tiers": [2, 3, 4, 5],
    },
    {
        "id": "vote_streak",
        "name": "Marathoner",
        "description": "Longest run of consecutive votes with at most a 10-minute gap.",
        "icon": "running",
        "tiers": [200, 400, 600, 1000],
    },
    {
        "id": "hours_voted",
        "name": "Around the Clock",
        "description": "Distinct hours of the day (0-23) in which you've cast at least one vote.",
        "icon": "clock",
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

#: Canonical media types tracked by the "Multi Media" achievement, in display
#: order, paired with their human-readable labels.  The achievement's tiers
#: (max 5) assume exactly these five types; :func:`~vtsearch.achievements.get_full_state` reports a
#: per-type ticked flag so the UI can show which ones the user has voted on.
MEDIA_TYPES: list[dict[str, str]] = [
    {"id": "audio", "name": "Audio"},
    {"id": "image", "name": "Image"},
    {"id": "text", "name": "Text"},
    {"id": "video", "name": "Video"},
    {"id": "document", "name": "Document"},
]

#: Hours of the day tracked by the "Around the Clock" achievement, bucketed
#: in the voter's local wall-clock time (see ``vtsearch.achievements._user_tz_offset_minutes``).
HOURS_OF_DAY: tuple[int, ...] = tuple(range(24))

#: Readme Reader docs.  Each entry pairs a doc with the code phrase printed at
#: the bottom of it.  The phrase is matched server-side: the user pastes their
#: guess and we compare it to ``_DOC_HASHES`` (SHA-256 of the normalised
#: phrase).  Phrases are kept in source so the test suite can verify each doc's
#: footer is in sync, but :func:`~vtsearch.achievements.get_full_state` never returns them to the
#: client; it returns only the per-doc read state.
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
        "path": "docs/user/USER_GUIDE.md",
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
