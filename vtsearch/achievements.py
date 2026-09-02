"""Achievement system: persistent per-user milestone tracking.

State lives in the current user's per-user settings file under the
``achievement_state`` key, so milestones are isolated per user and
round-trip through that user's settings sync source like every other
per-user setting.

Categories and their tier thresholds are declared statically in
:data:`ACHIEVEMENTS`.  Counters are incremented from hook points across the
codebase (vote toggle, dataset load completion, label import, find).  The UI
polls :func:`get_full_state` to render the Achievements tab and to discover
newly-unlocked tiers.

Two independent server-side watermarks track how far the user has been
notified, so a one-time toast and a persistent notification dot don't have
to share state:

- ``toasted`` drives ``pending_toasts``: the milestones whose toast hasn't
  fired yet.  The client pops a toast for each, then advances the watermark
  via :func:`mark_toasted`, so the toast fires exactly once per real unlock
  and never replays on the next app start.
- ``announced`` drives ``pending_announcements``: the milestones the user
  hasn't yet seen in the Achievements panel.  This is what lights the
  notification dot.  It clears only when the user opens the panel, which ACKs
  each announcement via :func:`acknowledge`.

The dot therefore stays lit until the user actually looks at the panel, while
the toast is a fire-once affair independent of whether they ever open it.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from vtsearch.achievements_catalog import (
    ACHIEVEMENTS,
    DOCS,
    EXCLUDED_DATASET_IMPORTERS,
    HOURS_OF_DAY,
    MEDIA_TYPES,
    STREAK_GAP_SECONDS,
    TIER_NAMES,
    _ACH_BY_ID,
    _DOC_BY_ID,
    _DOC_HASHES,
    _hash_phrase,
)
from vtsearch.auth import get_current_user
from vtsearch.settings import mutate_user, snapshot_user


#: The static catalog is re-exported so callers — routes, tests, the shim —
#: keep reaching these names through ``vtsearch.achievements``; the split into
#: :mod:`vtsearch.achievements_catalog` is internal.
__all__ = [
    "ACHIEVEMENTS",
    "DOCS",
    "EXCLUDED_DATASET_IMPORTERS",
    "HOURS_OF_DAY",
    "MEDIA_TYPES",
    "STREAK_GAP_SECONDS",
    "TIER_NAMES",
    "acknowledge",
    "get_full_state",
    "mark_toasted",
    "record_dataset_load",
    "record_detector_import",
    "record_doc_phrase",
    "record_find",
    "record_vote",
    "wipe_state",
]

logger = logging.getLogger(__name__)


def _ensure_state(settings: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable ``achievement_state`` sub-dict inside *settings*.

    Initializes missing keys in place so callers don't have to. Must be
    called from inside a ``mutate_user`` mutator so the mutations are
    written back to disk under the cross-process settings lock.
    """
    state = settings.get("achievement_state")
    if not isinstance(state, dict):
        state = {}
        settings["achievement_state"] = state
    counters = state.setdefault("counters", {})
    announced = state.setdefault("announced", {})
    toasted = state.setdefault("toasted", {})
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
        if cid not in toasted or not isinstance(toasted[cid], int):
            toasted[cid] = -1
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


def _is_disabled() -> bool:
    """Return True when the current user opted out of achievement tracking.

    Recording hooks become no-ops and :func:`get_full_state` returns a
    zeroed shell when this is set. The check resolves the per-user
    setting on every call, so toggling the flag takes effect immediately
    without restarting the process.
    """
    try:
        from vtsearch.settings import get_enable_achievements

        return not bool(get_enable_achievements())
    except Exception:
        return False


def wipe_state() -> None:
    """Clear the current user's stored ``achievement_state`` entirely.

    Called from the settings route when ``enable_achievements`` flips
    from True to False so the counters reset to zero (and stay there
    while the feature remains off). Idempotent: running it on an
    already-empty state is a no-op write.
    """

    def _apply(cache: dict[str, Any]) -> None:
        cache.pop("achievement_state", None)

    mutate_user(_apply)


def _user_tz_offset_minutes() -> int:
    """Minutes to subtract from UTC to reach the voter's local wall-clock time.

    Read from the ``X-Timezone-Offset`` request header, which the frontend
    sets to the browser's ``Date.prototype.getTimezoneOffset()`` value (the
    difference, in minutes, between UTC and local time: positive west of UTC,
    negative east of it).  Subtracting it from a UTC instant yields the user's
    local wall clock, so the "Around the Clock" hours and "Your Days are
    Numbered" days bucket by the time the user actually saw on their screen.

    Returns 0 (i.e. UTC) when there is no request context or header — CLI
    runs, background threads, and tests all fall back to UTC.
    """
    try:
        from flask import request

        raw = request.headers.get("X-Timezone-Offset")
    except RuntimeError:
        # No Flask request context (CLI mode, background thread, etc.)
        return 0
    if raw is None:
        return 0
    try:
        offset = int(raw)
    except (TypeError, ValueError):
        return 0
    # Clamp to the real-world range (UTC-14 .. UTC+14) so a malformed or
    # hostile header can't shove buckets into nonsense values.
    return max(-14 * 60, min(14 * 60, offset))


def record_vote(
    detector_id: str = "",
    media_type: str = "",
    *,
    now: float | None = None,
    tz_offset_minutes: int | None = None,
    count_streak: bool = True,
) -> None:
    """Record one user vote and credit every vote-driven achievement.

    Credits ``votes_cast`` always, ``detectors_trained`` on a detector's first
    vote, ``media_types_touched`` on a media type's first vote, ``days_active``
    on the first vote of each local calendar day, ``hours_voted`` on the first
    vote within each local hour-of-day bucket, and (when *count_streak* is set)
    updates the ``vote_streak`` watermark (longest run of consecutive votes
    with gaps ≤ 10 minutes).

    Days and hours bucket by the voter's local wall-clock time, resolved from
    the request's timezone offset (see :func:`_user_tz_offset_minutes`), so the
    milestones reflect the clock the user actually saw rather than UTC.

    Args:
        detector_id: ID of the detector the vote belongs to.  Empty string
            (no active detector) skips the training credit.
        media_type: Canonical media type id (``"audio"``/``"image"``/etc.)
            for the item being voted on.  Empty string skips the
            ``media_types_touched`` credit.
        now: Override the current unix timestamp (seconds since epoch); only
            used by tests.  Default uses :func:`time.time`.
        tz_offset_minutes: Override the local-time offset (UTC minus local, in
            minutes) instead of reading it from the request header; only used
            by tests.  Default resolves :func:`_user_tz_offset_minutes`.
        count_streak: When True (the default), this vote participates in the
            Marathoner ``vote_streak`` watermark.  Bulk / non-individual vote
            paths (Verified Good/Bad on a hand-selected set, fill-from-sort,
            label import) pass False: they still credit every other vote
            achievement, but a batch of N items must not manufacture an
            N-long "consecutive individual votes" streak, and they leave the
            running streak (and its ``last_vote_ts`` clock) untouched so a real
            hand-clicked run on either side of them stays intact.
    """
    if _is_disabled():
        return
    ts = time.time() if now is None else float(now)
    offset = _user_tz_offset_minutes() if tz_offset_minutes is None else int(tz_offset_minutes)
    local_dt = datetime.fromtimestamp(ts, tz=timezone.utc) - timedelta(minutes=offset)
    date_str = local_dt.strftime("%Y-%m-%d")
    hour = local_dt.hour

    mutate_user(lambda cache: _credit_vote(cache, ts, detector_id, media_type, date_str, hour, count_streak))


def _credit_vote(
    cache: dict[str, Any],
    ts: float,
    detector_id: str,
    media_type: str,
    date_str: str,
    hour: int,
    count_streak: bool = True,
) -> None:
    """Apply one vote's credits to *cache* in place. See :func:`record_vote`."""
    state = _ensure_state(cache)
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

    # The Marathoner streak measures sustained *individual* hand-clicking, so
    # bulk / non-individual votes are transparent to it: they neither advance
    # nor reset the running streak, and they leave ``last_vote_ts`` alone so
    # the gap is always measured between two real hand-clicks.
    if count_streak:
        last_ts = float(state.get("last_vote_ts") or 0.0)
        if last_ts <= 0.0 or (ts - last_ts) > STREAK_GAP_SECONDS:
            current = 1
        else:
            current = int(state.get("current_streak") or 0) + 1
        state["current_streak"] = current
        state["last_vote_ts"] = ts
        if current > counters["vote_streak"]:
            counters["vote_streak"] = current


def record_dataset_load(importer_name: str) -> None:
    """Record one dataset load (skipping demos/synthetic)."""
    if _is_disabled():
        return
    if importer_name in EXCLUDED_DATASET_IMPORTERS:
        return

    def _apply(cache: dict[str, Any]) -> None:
        state = _ensure_state(cache)
        state["counters"]["datasets_loaded"] += 1

    mutate_user(_apply)


def record_detector_import(detector_id: str) -> None:
    """Record one detector receiving imported labels.  Dedupes by detector_id."""
    if _is_disabled():
        return
    if not detector_id:
        return

    def _apply(cache: dict[str, Any]) -> None:
        state = _ensure_state(cache)
        imported = state["imported_detector_ids"]
        if detector_id in imported:
            return
        imported.append(detector_id)
        state["counters"]["detectors_imported"] += 1

    mutate_user(_apply)


def record_find(n_scored: int) -> None:
    """Record *n_scored* media items processed by a Find operation."""
    if _is_disabled():
        return
    if n_scored <= 0:
        return

    def _apply(cache: dict[str, Any]) -> None:
        state = _ensure_state(cache)
        state["counters"]["find_media"] += int(n_scored)

    mutate_user(_apply)


def record_doc_phrase(phrase: str) -> dict[str, Any]:
    """Check a user-submitted phrase against every doc and credit on match.

    The match is case-insensitive and whitespace-normalised. Anything that
    survives :func:`_normalize_phrase` and SHA-256-hashes to a registered doc
    counts.  A phrase that matches a doc already in ``docs_read_ids`` is
    reported as ``already_read`` (idempotent, no double-credit).

    Returns a result dict with:

    - ``matched`` (bool): whether the phrase corresponds to a known doc.
    - ``doc_id`` / ``doc_name`` (str | None): identifying the matched doc.
    - ``already_read`` (bool): True when the doc was previously credited.
    """
    if _is_disabled():
        return {"matched": False, "doc_id": None, "doc_name": None, "already_read": False}
    h = _hash_phrase(phrase)
    matched_id: str | None = None
    for doc_id, doc_hash in _DOC_HASHES.items():
        if h == doc_hash:
            matched_id = doc_id
            break

    if matched_id is None:
        return {"matched": False, "doc_id": None, "doc_name": None, "already_read": False}

    doc = _DOC_BY_ID[matched_id]
    already_read = False

    def _apply(cache: dict[str, Any]) -> None:
        nonlocal already_read
        state = _ensure_state(cache)
        read_ids = state["docs_read_ids"]
        if matched_id in read_ids:
            already_read = True
            return
        read_ids.append(matched_id)
        state["counters"]["docs_read"] = len(read_ids)

    mutate_user(_apply)

    return {
        "matched": True,
        "doc_id": matched_id,
        "doc_name": doc["name"],
        "already_read": already_read,
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
            "pending_toasts": [ <same shape as pending_announcements>, ... ],
            "docs": [{"id", "name", "path", "read"}, ...],
            "media_types": [{"id", "name", "seen"}, ...],
            "hours": [{"hour": <0..23>, "seen": <bool>}, ...],
        }

    ``pending_announcements`` lights the notification dot (it clears only when
    the user opens the panel and ACKs via :func:`acknowledge`), while
    ``pending_toasts`` drives the one-time unlock toast (the client pops each,
    then advances the watermark via :func:`mark_toasted`).  The two are
    independent so the toast fires once without forcing the dot to clear.

    The ``media_types`` and ``hours`` arrays back the "Multi Media" and
    "Around the Clock" expandable panels: each entry's ``seen`` flag says
    whether the user has cast a vote in that media type / hour-of-day bucket.
    """
    # A user who opted out gets the same payload built from an empty state:
    # ``_ensure_state`` fills a fresh dict with zeroed counters and -1
    # watermarks, which :func:`_state_payload` renders as every category
    # locked, nothing pending, and nothing seen. Building the shell through
    # the same code path is what keeps the two branches from drifting.
    if _is_disabled():
        return _state_payload(_ensure_state({}))
    return _state_payload(_ensure_state(snapshot_user(get_current_user())))


def _state_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Render an ``achievement_state`` dict into the frontend payload.

    Pure: takes the state, walks the static catalog, and returns the shape
    documented on :func:`get_full_state`. Both the enabled and the opted-out
    branches go through here, so there is exactly one definition of the
    response shape.
    """
    achievements: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    toasts: list[dict[str, Any]] = []
    for a in ACHIEVEMENTS:
        cid = a["id"]
        counter = int(state["counters"].get(cid, 0))
        tier_idx = _current_tier_idx(cid, counter)
        announced_idx = int(state["announced"].get(cid, -1))
        toasted_idx = int(state["toasted"].get(cid, -1))
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

        def _milestone(i: int, _a: dict[str, Any] = a, _cid: str = cid) -> dict[str, Any]:
            return {
                "id": _cid,
                "name": _a["name"],
                "icon": _a["icon"],
                "tier_idx": i,
                "tier_name": TIER_NAMES[i],
                "threshold": _a["tiers"][i],
            }

        pending.extend(_milestone(i) for i in range(announced_idx + 1, tier_idx + 1))
        toasts.extend(_milestone(i) for i in range(toasted_idx + 1, tier_idx + 1))

    read_ids = set(state.get("docs_read_ids", []))
    seen_types = set(state.get("media_types_seen", []))
    seen_hours = set(state.get("hours_seen", []))
    return {
        "tier_names": list(TIER_NAMES),
        "achievements": achievements,
        "pending_announcements": pending,
        "pending_toasts": toasts,
        "docs": [{"id": d["id"], "name": d["name"], "path": d["path"], "read": d["id"] in read_ids} for d in DOCS],
        "media_types": [{"id": m["id"], "name": m["name"], "seen": m["id"] in seen_types} for m in MEDIA_TYPES],
        "hours": [{"hour": h, "seen": h in seen_hours} for h in HOURS_OF_DAY],
    }


def acknowledge(category_id: str, tier_idx: int) -> bool:
    """Mark a tier as announced (seen in the panel) so the dot clears.

    Opening the panel also implies the toast need never fire, so this advances
    the ``toasted`` watermark alongside ``announced`` (a tier the user has
    already read about in the panel shouldn't pop a toast later).

    Returns True if the announced index advanced, False otherwise.
    """
    if category_id not in _ACH_BY_ID:
        return False
    if tier_idx < 0 or tier_idx >= len(TIER_NAMES):
        return False

    advanced = False

    def _apply(cache: dict[str, Any]) -> None:
        nonlocal advanced
        state = _ensure_state(cache)
        if tier_idx > int(state["toasted"].get(category_id, -1)):
            state["toasted"][category_id] = tier_idx
        prev = int(state["announced"].get(category_id, -1))
        if tier_idx <= prev:
            return
        state["announced"][category_id] = tier_idx
        advanced = True

    mutate_user(_apply)
    return advanced


def mark_toasted(category_id: str, tier_idx: int) -> bool:
    """Mark a tier's unlock toast as shown so it isn't popped again.

    The client calls this right after rendering the toast.  Advancing the
    ``toasted`` watermark keeps the milestone out of future ``pending_toasts``
    lists, so the toast fires exactly once and never replays on app restart.
    Unlike :func:`acknowledge` it leaves ``announced`` untouched, so the
    notification dot stays lit until the user opens the panel.

    Returns True if the toasted index advanced, False otherwise.
    """
    if category_id not in _ACH_BY_ID:
        return False
    if tier_idx < 0 or tier_idx >= len(TIER_NAMES):
        return False

    advanced = False

    def _apply(cache: dict[str, Any]) -> None:
        nonlocal advanced
        state = _ensure_state(cache)
        prev = int(state["toasted"].get(category_id, -1))
        if tier_idx <= prev:
            return
        state["toasted"][category_id] = tier_idx
        advanced = True

    mutate_user(_apply)
    return advanced
