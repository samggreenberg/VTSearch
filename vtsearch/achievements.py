"""Achievement system: persistent per-user milestone tracking.

State lives inside ``data/settings.json`` under the ``achievement_state`` key,
so it round-trips through the settings sync source like every other setting.

Categories and their tier thresholds are declared statically in
:data:`ACHIEVEMENTS`.  Counters are incremented from hook points across the
codebase (vote toggle, dataset load completion, label import, find).  The UI
polls :func:`get_full_state` to render the Achievements tab and to discover
newly-unlocked tiers (``pending_announcements``); the client ACKs each
announcement via :func:`acknowledge` so the popup doesn't fire on refresh.
"""

from __future__ import annotations

import logging
from typing import Any

from vtsearch.settings import _ensure_loaded, _save, _settings_lock

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
]

_ACH_BY_ID: dict[str, dict[str, Any]] = {a["id"]: a for a in ACHIEVEMENTS}

#: Importer names whose successful dataset loads do NOT count toward
#: ``datasets_loaded``.  These are the demo/synthetic data paths users
#: don't have ownership over.
EXCLUDED_DATASET_IMPORTERS: frozenset[str] = frozenset({"demo", "synthetic"})


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


def record_vote(detector_id: str = "") -> None:
    """Record one user vote and credit detector training on its first vote.

    Args:
        detector_id: ID of the detector the vote belongs to.  Empty string
            (no active detector) skips the training credit.
    """
    with _settings_lock:
        s = _ensure_loaded()
        state = _ensure_state(s)
        state["counters"]["votes_cast"] += 1
        if detector_id:
            trained = state["trained_detector_ids"]
            if detector_id not in trained:
                trained.append(detector_id)
                state["counters"]["detectors_trained"] += 1
        _save(s)


def record_dataset_load(importer_name: str) -> None:
    """Record one dataset load (skipping demos/synthetic)."""
    if importer_name in EXCLUDED_DATASET_IMPORTERS:
        return
    with _settings_lock:
        s = _ensure_loaded()
        state = _ensure_state(s)
        state["counters"]["datasets_loaded"] += 1
        _save(s)


def record_detector_import(detector_id: str) -> None:
    """Record one detector receiving imported labels.  Dedupes by detector_id."""
    if not detector_id:
        return
    with _settings_lock:
        s = _ensure_loaded()
        state = _ensure_state(s)
        imported = state["imported_detector_ids"]
        if detector_id in imported:
            return
        imported.append(detector_id)
        state["counters"]["detectors_imported"] += 1
        _save(s)


def record_find(n_scored: int) -> None:
    """Record *n_scored* media items processed by a Find operation."""
    if n_scored <= 0:
        return
    with _settings_lock:
        s = _ensure_loaded()
        state = _ensure_state(s)
        state["counters"]["find_media"] += int(n_scored)
        _save(s)


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
        s = _ensure_loaded()
        state = _ensure_state(s)
        achievements: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for a in ACHIEVEMENTS:
            cid = a["id"]
            counter = int(state["counters"].get(cid, 0))
            tier_idx = _current_tier_idx(cid, counter)
            announced_idx = int(state["announced"].get(cid, -1))
            next_threshold: int | None = (
                a["tiers"][tier_idx + 1] if tier_idx + 1 < len(a["tiers"]) else None
            )
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
        return {
            "tier_names": list(TIER_NAMES),
            "achievements": achievements,
            "pending_announcements": pending,
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
        s = _ensure_loaded()
        state = _ensure_state(s)
        prev = int(state["announced"].get(category_id, -1))
        if tier_idx <= prev:
            return False
        state["announced"][category_id] = tier_idx
        _save(s)
        return True
