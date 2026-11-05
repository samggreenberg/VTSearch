"""Labelset source sync: keep a detector's labels current at its linked source.

A detector linked to a :class:`~vtscore.labels.sources.base.LabelsetSource`
(see :mod:`vtscore.labels.binding` for where the link comes from) has its
labels kept at that source as well as in its own JSON file, so the copy
outside the server is current rather than only as fresh as the last time
someone remembered to export (#4913).

**Roles.**  One engine - one slot and one pace per detector - serves plain
bindings and detector sharing alike; what a push or a pull does depends on
the detector's registry entry (:func:`sync_role`):

* :data:`SYNC_ROLE_PLAIN` - a binding without a ``remote_ref``, or the
  inherited deployment default: the paced ``save`` and the merging pulls
  described below.
* :data:`SYNC_ROLE_OWNER` - a ``local`` or ``forked`` detector whose
  binding names a shared container (a ``remote_ref``) at the active sharing
  source.  The push is ``share_save``: the container is *rewritten* to hold
  exactly the file's labelset, with its manifest.  Nothing is ever pulled
  into the file, not even before the first push - merging the container in
  would import exactly the foreign records tamper detection exists to
  catch; the owner's file is authoritative.  Each push's outcome lands on
  the binding: ``sync_error`` (a failure's message, cleared by the next
  success) and ``last_synced`` (moved by a success only), so a failure
  outlives a reload, which a toast would not.
* :data:`SYNC_ROLE_LINKED` - a received, read-only mirror: never pushed,
  never pulled.  Its file changes only through a refresh, and that write is
  the pull, so it schedules nothing.
* :data:`SYNC_ROLE_UNAVAILABLE` - a sharing binding whose source is not the
  active one, or not installed: nothing, logged once per detector.  It is
  never pointed at a different source.
* :data:`SYNC_ROLE_NONE` - nothing to sync with.

**Push** (:func:`schedule_labelset_push`).  Every write of a detector JSON
file schedules one - :mod:`vtscore.detectors.store` calls
:func:`note_detector_written` - so votes, label imports, corrections and
renames all reach the source without each route having to remember to.  The
payload is the detector file's **whole labelset** (every dataset's labels,
not just the active dataset's votes) plus its ``detector_meta``, read when the
push runs, so it is always the newest state.  Like every labelset export it
leaves behind the numbers this server's embedder and head measured
(:func:`~vtscore.datasets.labelset.labelset_for_export`).  Pushes are paced
per detector:

* never two at once for one detector - a change landing while a push is in
  flight rides the next one;
* at least the ``labelset_source_push_interval`` setting's seconds between the
  starts of two pushes (the app's ``labelset_source_push_interval_seconds``),
  so a slow target or a burst of votes costs one write per window;
* never sooner than ``_DEBOUNCE_DELAY`` after the change that scheduled it,
  so a burst inside that window is one push;
* a push whose payload matches what the source is known to hold is skipped;
* a plain detector's first push in a process pulls first (see below), so a
  push never overwrites labels only the source held - the backup of a
  detector this server lost - and a source that cannot be read is not pushed
  to at all.

A failed push is logged and leaves the source as it was (an owner's is also
recorded on its binding); the next change schedules another.
:func:`flush_pending_label_syncs` lands every pending push now, ignoring the
pace - tests use it, and an ``atexit`` hook calls it so the
newest labels reach the source on a normal shutdown (a SIGKILL still loses
whatever was waiting).

**Pull** (:func:`sync_from_labelset_source`, :func:`reconcile_labelset_source`;
plain bindings only - both return early for every other role).
Reads the source and **merges** its labels into the detector file: labels the
source has and the file lacks are added, and when both hold one item with
different labels the caller's ``prefer`` decides.  A pull never removes a
label.  The manual sync prefers the source (the user asked for its labels);
the pull when a detector loads or is linked prefers the file, which is never
older than what it last pushed.  A pull writes the file only when it changed
something, and that write schedules a push of the merged labels, so a source
that was behind catches up.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabeledElement, LabelSet
    from vtscore.labels.binding import ResolvedLabelsetSource
    from vtscore.labels.sharing import ActiveSharingSource
    from vtscore.state.core import DetectorContext

logger = logging.getLogger(__name__)

__all__ = [
    "SYNC_ROLES",
    "SYNC_ROLE_LINKED",
    "SYNC_ROLE_NONE",
    "SYNC_ROLE_OWNER",
    "SYNC_ROLE_PLAIN",
    "SYNC_ROLE_UNAVAILABLE",
    "flush_pending_label_syncs",
    "note_detector_written",
    "reconcile_labelset_source",
    "reset_label_sync_for_tests",
    "schedule_labelset_push",
    "sync_from_labelset_source",
    "sync_role",
    "sync_to_labelset_source",
]

#: :func:`sync_role` of a detector that syncs nowhere.
SYNC_ROLE_NONE = "none"
#: An ordinary binding (or the inherited default): paced ``save``, merging pulls.
SYNC_ROLE_PLAIN = "plain"
#: The owner of a shared container at the active sharing source: paced ``share_save``, no pull.
SYNC_ROLE_OWNER = "owner"
#: A received, read-only mirror: never pushed, never pulled.
SYNC_ROLE_LINKED = "linked"
#: A sharing binding whose source is not the active one, or not installed: nothing.
SYNC_ROLE_UNAVAILABLE = "unavailable"
#: Every :func:`sync_role`.
SYNC_ROLES = frozenset({SYNC_ROLE_NONE, SYNC_ROLE_PLAIN, SYNC_ROLE_OWNER, SYNC_ROLE_LINKED, SYNC_ROLE_UNAVAILABLE})

# The shortest wait between a change and the push it schedules.  200ms
# coalesces a rapid voting burst into one push while staying short enough that
# the source matches the UI within a tick when the pace allows it.
_DEBOUNCE_DELAY = 0.2

# How long the atexit flush waits for an in-flight push before giving up, so
# a hung target cannot hang interpreter shutdown.
_EXIT_FLUSH_TIMEOUT = 30.0


@dataclass
class _Slot:
    """Pacing state for one detector's pushes, keyed by detector name."""

    timer: threading.Timer | None = None
    #: Bumped whenever a timer is armed or cancelled; a timer whose token no
    #: longer matches fires as a no-op (it lost a race with a flush).
    token: int = 0
    in_flight: bool = False
    #: A change arrived while a push was in flight; push again after it.
    dirty: bool = False
    #: ``time.monotonic()`` at the start of the last push, or ``None``.
    last_started: float | None = None


# Guards ``_slots``, every slot's fields, ``_last_synced``, ``_last_shared``
# and ``_unavailable_noted``.  A leaf lock: nothing is called out to while it
# is held except ``Timer.start``.
_slots_cv = threading.Condition()
_slots: dict[str, _Slot] = {}
# Detector name → digest of the payload a plain source is known to hold (the
# last push, or the last pull).  A push of the same payload is skipped.
_last_synced: dict[str, str] = {}
# Detector name → (source name, remote_ref, digest) of an owner's last
# successful container push.  Kept apart from ``_last_synced`` so a detector
# that moves from a plain binding to a container (or to another container) is
# never skipped on the strength of what a different target holds.
_last_shared: dict[str, tuple[str, str, str]] = {}
# Ids of detectors whose sharing binding was already logged as unavailable.
_unavailable_noted: set[str] = set()


# ---------------------------------------------------------------------------
# Identity and context helpers
# ---------------------------------------------------------------------------


def _identify(name: str, detector_id: str = "") -> tuple[dict[str, Any] | None, str]:
    """Return ``(registry entry, detector id)`` for the detector named *name*.

    Falls back to a loaded context of that name when the registry has none
    (library callers that bind a source on a bare :class:`DetectorContext`).
    """
    from vtscore.detectors.registry import find_by_name, get_detector  # noqa: PLC0415

    entry = get_detector(detector_id) if detector_id else None
    if entry is None or entry.get("name") != name:
        entry = find_by_name(name)
    if entry is not None:
        return entry, entry.get("id", "") or detector_id
    if not detector_id:
        from vtscore.state.core import loaded_detector_contexts  # noqa: PLC0415

        for ctx in loaded_detector_contexts():
            if ctx.name == name:
                return None, ctx.detector_id
    return None, detector_id


def _template_context(name: str, detector_id: str) -> DetectorContext:
    """A context whose name / id resolve the source's ``{detector_*}`` templates.

    The loaded context when there is one, else a throwaway that carries only
    the identity - the push and pull run off the request, and the request's
    own active detector (if any) is not necessarily this one.
    """
    from vtscore.state.core import DetectorContext, get_detector_context  # noqa: PLC0415

    ctx = get_detector_context(detector_id) if detector_id else None
    if ctx is not None and ctx.name == name:
        return ctx
    return DetectorContext(detector_id, name=name)


def _owner(entry: dict[str, Any] | None) -> str | None:
    """The user a detector's sync runs as: its creator.

    Fixed per detector, so a server path confined to a user's data dir names
    one file however many readers vote on a shared detector.
    """
    return (entry or {}).get("created_by") or None


def _digest(labelset: LabelSet) -> str:
    """A content digest of *labelset*, the same one a sharing manifest records.

    Order-independent (:func:`~vtscore.labels.sharing.labelset_digest`), so a
    source that hands labels back reordered is not pushed to again for it.
    """
    from vtscore.labels.sharing import labelset_digest  # noqa: PLC0415

    return labelset_digest(labelset)


def _push_interval() -> float:
    """The configured gap between pushes of one detector, in seconds."""
    from vtscore.config import DEFAULT_LABELSET_SOURCE_PUSH_INTERVAL, get_setting  # noqa: PLC0415

    try:
        return max(0.0, float(get_setting("labelset_source_push_interval")))
    except Exception:  # noqa: BLE001 - unreadable settings
        return DEFAULT_LABELSET_SOURCE_PUSH_INTERVAL


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    """Where a detector's sync goes, resolved: its role and what that role needs."""

    role: str
    #: :data:`SYNC_ROLE_PLAIN`: the resolved binding.
    resolved: ResolvedLabelsetSource | None = None
    #: :data:`SYNC_ROLE_OWNER`: the active sharing source, the container, and
    #: the binding's own field values with the setting's fixed ones laid over.
    active: ActiveSharingSource | None = None
    remote_ref: str = ""
    field_values: dict[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """Whether there is something to push to, with every required field filled in."""
        if self.role == SYNC_ROLE_PLAIN:
            return self.resolved is not None and self.resolved.ready
        return self.role == SYNC_ROLE_OWNER and not self.missing_fields


def _target_for(entry: dict[str, Any] | None, detector_id: str = "") -> _Target:
    """Resolve the role of the detector with registry *entry* (see :func:`sync_role`)."""
    from vtscore.detectors.registry import is_linked  # noqa: PLC0415
    from vtscore.labels.binding import (  # noqa: PLC0415
        _missing_required,
        binding_matches_active_source,
        resolve_labelset_source,
        sharing_binding,
    )

    if is_linked(entry):
        return _Target(SYNC_ROLE_LINKED)
    binding = sharing_binding(entry)
    if binding is not None:
        from vtscore.labels.sharing import _field_values, active_sharing_source  # noqa: PLC0415

        active = active_sharing_source() if binding_matches_active_source(entry) else None
        if active is None:
            return _Target(SYNC_ROLE_UNAVAILABLE)
        values = _field_values(binding, active)
        return _Target(
            SYNC_ROLE_OWNER,
            active=active,
            remote_ref=str(binding["remote_ref"]),
            field_values=values,
            missing_fields=_missing_required(active.source, values),
        )
    resolved = resolve_labelset_source(entry, detector_id=detector_id)
    if resolved is None:
        return _Target(SYNC_ROLE_NONE)
    return _Target(SYNC_ROLE_PLAIN, resolved=resolved)


def sync_role(entry: dict[str, Any] | None, *, detector_id: str = "") -> str:
    """What the sync engine does for the detector whose registry entry is *entry*: one of :data:`SYNC_ROLES`.

    In order: a ``linked`` entry is :data:`SYNC_ROLE_LINKED` whatever its
    binding; an entry whose binding carries a ``remote_ref`` is
    :data:`SYNC_ROLE_OWNER` when that binding's source is the active sharing
    source (:func:`~vtscore.labels.binding.binding_matches_active_source`) and
    :data:`SYNC_ROLE_UNAVAILABLE` otherwise
    (:func:`~vtscore.labels.binding.sharing_note` says why); anything else is
    :data:`SYNC_ROLE_PLAIN` when
    :func:`~vtscore.labels.binding.resolve_labelset_source` finds a source,
    else :data:`SYNC_ROLE_NONE`.  The module docstring says what each role's
    push and pull do.

    Args:
        entry: The detector's registry entry, or ``None`` for a detector with
            no entry (only an in-memory context binding can make that one
            plain).
        detector_id: The detector's id, for that context binding; defaults to
            ``entry["id"]``.
    """
    return _target_for(entry, detector_id).role


def _note_unavailable(entry: dict[str, Any] | None) -> None:
    """Log, once per detector, that its sharing binding is not in use and why."""
    from vtscore.labels.binding import sharing_note  # noqa: PLC0415

    key = str((entry or {}).get("id") or (entry or {}).get("name") or "")
    with _slots_cv:
        if key in _unavailable_noted:
            return
        _unavailable_noted.add(key)
    logger.warning(
        "Not syncing detector %r: %s",
        (entry or {}).get("name", ""),
        sharing_note(entry) or "its sharing source is not available",
    )


# ---------------------------------------------------------------------------
# Push scheduling
# ---------------------------------------------------------------------------


def _delay_for(slot: _Slot, interval: float) -> float:
    if slot.last_started is None:
        return _DEBOUNCE_DELAY
    return max(_DEBOUNCE_DELAY, slot.last_started + interval - time.monotonic())


def _arm_locked(name: str, slot: _Slot, delay: float) -> None:
    """Arm *slot*'s timer.  Caller holds ``_slots_cv``."""
    slot.token += 1
    timer = threading.Timer(delay, _fire, args=(name, slot.token))
    timer.daemon = True
    slot.timer = timer
    timer.start()


def schedule_labelset_push(detector_name: str, *, detector_id: str = "") -> None:
    """Schedule a paced push of *detector_name*'s labels to its labelset source.

    Returns immediately.  A no-op when the detector syncs nowhere, is a
    received (linked) detector, has a sharing binding whose source is not the
    active one (logged once per detector), or its source is missing a
    required field value.  Otherwise the push - a plain ``save``, or an
    owner's ``share_save`` (see :func:`sync_role`) - runs on a background
    timer as soon as the pace allows (see the module docstring); calls made
    while one is already scheduled or in flight are folded into it.
    """
    if not detector_name:
        return
    entry, detector_id = _identify(detector_name, detector_id)
    target = _target_for(entry, detector_id)
    if target.role == SYNC_ROLE_UNAVAILABLE:
        _note_unavailable(entry)
        return
    if not target.ready:
        return
    interval = _push_interval()
    with _slots_cv:
        if target.role == SYNC_ROLE_OWNER and entry is not None:
            # In use again: a later lapse is worth logging again.
            _unavailable_noted.discard(str(entry.get("id") or ""))
        slot = _slots.setdefault(detector_name, _Slot())
        if slot.in_flight:
            slot.dirty = True
            return
        if slot.timer is not None:
            # The armed push reads the detector file when it fires, so it
            # already carries this change.
            return
        _arm_locked(detector_name, slot, _delay_for(slot, interval))


def note_detector_written(data: dict[str, Any]) -> None:
    """Hook for :mod:`vtscore.detectors.store`: a detector file was (re)written.

    A write inside :func:`~vtscore.detectors.store.allow_linked_write` - a
    received detector's mirror being received or refreshed - schedules
    nothing: that write *is* the pull.  Checked here rather than left to the
    role, because a receive writes the file before the entry that makes it
    ``linked`` exists.  Never raises - losing a sync must not fail the write
    that triggered it.
    """
    try:
        from vtscore.detectors.store import _in_linked_write  # noqa: PLC0415

        if _in_linked_write():
            return
        name = data.get("name") if isinstance(data, dict) else None
        if isinstance(name, str) and name:
            schedule_labelset_push(name)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to schedule a labelset source push")


def sync_to_labelset_source() -> None:
    """Schedule a push of the active detector's labels to its labelset source.

    Every detector-file write already schedules one (see
    :func:`note_detector_written`); this is for a caller that changed what the
    push carries without writing the file, and for library code that linked a
    source on a bare :class:`~vtscore.state.core.DetectorContext`.
    """
    from vtscore.state.core import get_active_detector_context  # noqa: PLC0415

    ctx = get_active_detector_context()
    if ctx is None or not ctx.detector_id:
        return
    name = ctx.name
    if not name:
        from vtscore.detectors.registry import get_detector  # noqa: PLC0415

        name = (get_detector(ctx.detector_id) or {}).get("name", "")
    schedule_labelset_push(name, detector_id=ctx.detector_id)


def _fire(name: str, token: int) -> None:
    """Timer callback: run the armed push unless a flush already took it."""
    with _slots_cv:
        slot = _slots.get(name)
        if slot is None or slot.timer is None or slot.token != token:
            return
        slot.timer = None
        slot.in_flight = True
    _run_push(name)


def _run_push(name: str) -> None:
    """Push *name* (its slot is already marked in flight), then re-arm if dirty."""
    started = time.monotonic()
    try:
        _push_detector(name)
    finally:
        interval = _push_interval()
        with _slots_cv:
            slot = _slots.setdefault(name, _Slot())
            slot.in_flight = False
            slot.last_started = started
            if slot.dirty:
                slot.dirty = False
                if slot.timer is None:
                    _arm_locked(name, slot, _delay_for(slot, interval))
            _slots_cv.notify_all()


def _push_payload(data: dict[str, Any]) -> LabelSet:
    """The labelset a push sends: the file's whole labelset plus ``detector_meta``, as an export carries it.

    No threshold, rank or score goes with it (#5131): those were measured by
    this server's embedder and head, and the receiver trains its own.  The
    file keeps them.
    """
    from vtscore.datasets.labelset import LabelSet, labelset_for_export  # noqa: PLC0415
    from vtscore.detectors.input_spec import build_detector_meta  # noqa: PLC0415

    payload = dict(data.get("labelset") or {})
    payload["detector_meta"] = build_detector_meta(data)
    return LabelSet.from_dict(labelset_for_export(payload))


def _push_detector(name: str) -> None:
    """Push the detector file's labelset wherever its role sends it, synchronously.

    The role is resolved again here, when the push runs, not when it was
    scheduled: an entry that was received, shared or unshared in between is
    pushed as it is now.
    """
    entry, detector_id = _identify(name)
    target = _target_for(entry, detector_id)
    if target.role == SYNC_ROLE_PLAIN and target.resolved is not None:
        _push_plain(name, entry, detector_id, target.resolved)
    elif target.role == SYNC_ROLE_OWNER and entry is not None:
        _push_container(name, entry, detector_id, target)
    elif target.role == SYNC_ROLE_UNAVAILABLE:
        _note_unavailable(entry)


def _push_plain(name: str, entry: dict[str, Any] | None, detector_id: str, resolved: ResolvedLabelsetSource) -> None:
    """A plain binding's push: pull first once per process, then ``save`` unless the source holds it already."""
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415
    from vtscore.state.core import override_detector_context  # noqa: PLC0415
    from vtscore.state.current_user import thread_user  # noqa: PLC0415

    if not resolved.ready:
        logger.warning(
            "Not syncing detector %r to %s: missing %s", name, resolved.source_name, ", ".join(resolved.missing_fields)
        )
        return
    source = resolved.source
    if source is None:
        return
    with _slots_cv:
        seen = name in _last_synced
    if not seen:
        # This process has not read the source yet, so it may hold labels the
        # file lacks - a backup of a detector this server lost and the user
        # just re-created.  Pushing blind would overwrite them; merge first
        # (the file wins a disagreement), and if the source cannot be read,
        # do not risk overwriting what it holds.
        remote, _changes = _pull(name, detector_id, prefer="local")
        if remote is None:
            logger.warning(
                "Not syncing detector %r to %s: its current contents could not be read", name, resolved.source_name
            )
            return
    data = _read_detector(_detector_path(name))
    if data is None:
        return
    labelset = _push_payload(data)
    digest = _digest(labelset)
    with _slots_cv:
        if _last_synced.get(name) == digest:
            return
    try:
        with thread_user(_owner(entry)), override_detector_context(_template_context(name, detector_id)):
            source.save(labelset, resolved.field_values)
    except Exception:  # noqa: BLE001 - a sync target is a side channel; the file is the record
        logger.exception("Failed to sync detector %r's labels to %s", name, resolved.source_name)
        return
    with _slots_cv:
        _last_synced[name] = digest


def _push_container(name: str, entry: dict[str, Any], detector_id: str, target: _Target) -> None:
    """An owner's push: rewrite its container to hold exactly the file's labelset, and record how it went.

    No pull first and no merge: the owner's file is authoritative, and a
    record a recipient added to the container is meant to be *gone* after
    this push, not imported.  The payload is the plain push's
    (:func:`_push_payload`) with its manifest (:func:`~vtscore.labels.sharing.share_payload`).
    """
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415
    from vtscore.labels.sharing import SharingError, share_payload  # noqa: PLC0415
    from vtscore.state.core import override_detector_context  # noqa: PLC0415
    from vtscore.state.current_user import thread_user  # noqa: PLC0415

    active = target.active
    if active is None:
        return
    if target.missing_fields:
        logger.warning(
            "Not pushing detector %r to its shared copy at %s: missing %s",
            name,
            active.source_name,
            ", ".join(target.missing_fields),
        )
        return
    data = _read_detector(_detector_path(name))
    if data is None:
        return
    labelset, meta = share_payload(data)
    sent = (active.source_name, target.remote_ref, meta.manifest.digest)
    with _slots_cv:
        if _last_shared.get(name) == sent:
            return
    try:
        # As the detector's creator, whoever voted: the source resolves the
        # caller itself, and a container only its owner may rewrite.
        with thread_user(_owner(entry)), override_detector_context(_template_context(name, detector_id)):
            active.source.share_save(target.remote_ref, labelset, meta, dict(target.field_values))
    except Exception as exc:  # noqa: BLE001 - a sync target is a side channel; the file is the record
        if isinstance(exc, SharingError):
            logger.warning("Failed to push detector %r to its shared copy at %s: %s", name, active.source_name, exc)
        else:
            logger.exception("Failed to push detector %r to its shared copy at %s", name, active.source_name)
        _record_push_outcome(entry, target.remote_ref, error=str(exc) or type(exc).__name__)
        return
    with _slots_cv:
        _last_shared[name] = sent
    _record_push_outcome(entry, target.remote_ref, error=None)


def _record_push_outcome(entry: dict[str, Any], remote_ref: str, *, error: str | None) -> None:
    """Record an owner push's outcome on the entry's binding: ``sync_error``, and ``last_synced`` on success.

    Merged into the binding as it is on disk now
    (:func:`~vtscore.detectors.registry.update_sharing_binding`), so a rename
    or an unshare that landed during the push is not undone.  A failure to
    record is logged; the push itself already happened (or did not).
    """
    from vtscore.detectors.registry import update_sharing_binding  # noqa: PLC0415

    changes: dict[str, Any] = {"sync_error": error}
    if error is None:
        changes["last_synced"] = time.time()
    try:
        update_sharing_binding(str(entry.get("id") or ""), remote_ref, **changes)
    except Exception:  # noqa: BLE001 - bookkeeping must not fail the push thread
        logger.exception("Failed to record the outcome of detector %r's push", entry.get("name", ""))


def _flush(timeout: float | None) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        with _slots_cv:
            while any(s.in_flight for s in _slots.values()):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    logger.warning("Gave up waiting for an in-flight labelset source push")
                    return
                _slots_cv.wait(remaining)
            due = []
            for name, slot in _slots.items():
                if slot.timer is not None:
                    slot.timer.cancel()
                    slot.timer = None
                    slot.token += 1
                    slot.in_flight = True
                    due.append(name)
        if not due:
            return
        for name in due:
            _run_push(name)


def flush_pending_label_syncs() -> None:
    """Run every scheduled push now, ignoring the pace, and wait for in-flight ones.

    Covers every role that pushes - a plain ``save`` and an owner's container
    push alike.  Used by tests (to assert on what the source holds) and at
    interpreter exit.
    """
    _flush(None)


def reset_label_sync_for_tests() -> None:
    """Cancel every scheduled push and forget what the sources hold (for conftest).

    Unlike :func:`flush_pending_label_syncs` this does **not** run the pending
    pushes - it discards them, so a push scheduled by one test cannot fire
    into the next.  An in-flight push is waited out first.
    """
    with _slots_cv:
        deadline = time.monotonic() + 10.0
        while any(s.in_flight for s in _slots.values()) and time.monotonic() < deadline:
            _slots_cv.wait(deadline - time.monotonic())
        for slot in _slots.values():
            if slot.timer is not None:
                slot.timer.cancel()
        _slots.clear()
        _last_synced.clear()
        _last_shared.clear()
        _unavailable_noted.clear()


# Land the newest labels at their sources on a normal exit (Ctrl-C, gunicorn
# SIGQUIT, ``sys.exit``).  SIGKILL / ``os._exit`` bypass atexit and lose
# whatever was waiting on the pace - the cost of not writing on every vote.
atexit.register(_flush, _EXIT_FLUSH_TIMEOUT)


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------


def _merge_labels(
    local: list[LabeledElement],
    remote: list[LabeledElement],
    *,
    prefer: str,
) -> tuple[list[LabeledElement], int]:
    """Merge *remote* labels into *local*, returning ``(merged, changes)``.

    Items are matched by :func:`~vtscore.datasets.labelset.element_identity_keys`
    (origin or content hash).  Local order is kept and remote-only labels are
    appended; when both sides hold one item with different labels, the
    ``prefer`` side (``"local"`` or ``"source"``) wins.  Nothing is removed.
    """
    from vtscore.datasets.labelset import element_identity_keys  # noqa: PLC0415

    remote_by_key: dict[Any, LabeledElement] = {}
    for el in remote:
        for key in element_identity_keys(el):
            remote_by_key.setdefault(key, el)

    merged: list[LabeledElement] = []
    matched: set[int] = set()
    changes = 0
    for el in local:
        match = next((remote_by_key[k] for k in element_identity_keys(el) if k in remote_by_key), None)
        if match is not None:
            matched.add(id(match))
            if prefer == "source" and match.label in ("good", "bad") and match.label != el.label:
                merged.append(match)
                changes += 1
                continue
        merged.append(el)

    seen = {k for el in merged for k in element_identity_keys(el)}
    for el in remote:
        if id(el) in matched or el.label not in ("good", "bad"):
            continue
        keys = element_identity_keys(el)
        if not keys or any(k in seen for k in keys):
            continue
        seen.update(keys)
        merged.append(el)
        changes += 1
    return merged, changes


def _pull(name: str, detector_id: str, *, prefer: str) -> tuple[LabelSet | None, int]:
    """Read *name*'s source and merge it into the detector file.

    Returns ``(what the source held, labels added or changed)``; the first is
    ``None`` when the detector syncs nowhere, is not a plain binding (only a
    plain binding is ever pulled - see :func:`sync_role`), its source is
    missing a field, or the read failed.
    """
    from vtscore.detectors.input_spec import apply_detector_meta  # noqa: PLC0415
    from vtscore.detectors.label_sync import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.registry import update_detector  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector, _write_detector  # noqa: PLC0415
    from vtscore.state.core import override_detector_context  # noqa: PLC0415
    from vtscore.state.current_user import thread_user  # noqa: PLC0415

    entry, detector_id = _identify(name, detector_id)
    target = _target_for(entry, detector_id)
    resolved = target.resolved
    if target.role != SYNC_ROLE_PLAIN or resolved is None or not resolved.ready or resolved.source is None:
        return None, 0
    try:
        with thread_user(_owner(entry)), override_detector_context(_template_context(name, detector_id)):
            remote = resolved.source.load_full(resolved.field_values)
    except Exception:  # noqa: BLE001 - an unreachable source must not fail a load
        logger.exception("Failed to read detector %r's labels from %s", name, resolved.source_name)
        return None, 0

    changes = 0
    if remote.elements or remote.detector_meta:
        path = _detector_path(name)
        with label_sync_write_lock:
            data = _read_detector(path)
            if data is not None:
                # The meta's input_spec (and media_type, when the file has
                # none) comes across; its threshold never does - the receiver
                # retrains and derives its own.
                changed = apply_detector_meta(data, remote.detector_meta)
                from vtscore.datasets.labelset import LabelSet  # noqa: PLC0415

                local = LabelSet.from_dict(data.get("labelset") or {})
                merged, changes = _merge_labels(local.elements, remote.elements, prefer=prefer)
                if changes:
                    labelset = dict(data.get("labelset") or {})
                    labelset["labels"] = [el.to_dict() for el in merged]
                    data["labelset"] = labelset
                    changed = True
                if changed:
                    _write_detector(path, data)
                if changes and entry is not None:
                    update_detector(entry["id"], num_training=len(merged))
    with _slots_cv:
        _last_synced[name] = _digest(remote)
    return remote, changes


def _rehydrate_loaded(detector_id: str) -> None:
    """Re-derive a loaded detector's votes from its file after a pull changed it."""
    from vtscore.detectors.dataset_sync import ensure_votes_match_active_dataset  # noqa: PLC0415
    from vtscore.state.core import get_detector_context, override_detector_context  # noqa: PLC0415

    ctx = get_detector_context(detector_id) if detector_id else None
    if ctx is None:
        return
    # Drop the cached labelset so the freshness check cannot mistake the file
    # for one it has already seen (its mtime cache may lag the write by a tick).
    ctx.cached_labelset = None
    with override_detector_context(ctx):
        ensure_votes_match_active_dataset()


def sync_from_labelset_source(
    detector_id: str | None = None,
    *,
    prefer: str = "source",
) -> list[dict[str, Any]] | None:
    """Pull the labels at a detector's labelset source into its file.

    Labels the source holds and the detector lacks are added; with
    ``prefer="source"`` (the default - the manual "sync now") the source's
    label also wins where the two disagree, and with ``prefer="local"`` the
    detector's does.  No label is removed.  A loaded detector's votes are
    re-derived from the merged file, and the merged labels are pushed back so
    the source catches up on anything only the detector had.

    Args:
        detector_id: The detector to sync; the active detector when omitted.
        prefer: ``"source"`` or ``"local"`` - which label wins a disagreement.

    Only a plain binding is pulled.  An owner's container is never pulled
    into its file (the file is authoritative), a received detector changes
    only through a refresh, and a sharing binding whose source is not active
    is left alone; for each of those this returns ``None`` at once, touching
    nothing.  A route tells them apart with :func:`sync_role`.

    Returns:
        The label dicts the source held, or ``None`` when the detector syncs
        nowhere, is not a plain binding, its source is missing a field value,
        the read failed, or the source holds nothing.
    """
    if prefer not in ("source", "local"):
        raise ValueError(f"prefer must be 'source' or 'local', got {prefer!r}")
    from vtscore.detectors.registry import get_detector  # noqa: PLC0415
    from vtscore.state.core import get_active_detector_context, get_detector_context  # noqa: PLC0415

    ctx = get_detector_context(detector_id) if detector_id else get_active_detector_context()
    did = detector_id or (ctx.detector_id if ctx is not None else "")
    entry = get_detector(did) if did else None
    name = (entry or {}).get("name") or (ctx.name if ctx is not None else "")
    if not name:
        return None
    if sync_role(entry, detector_id=did) != SYNC_ROLE_PLAIN:
        return None

    remote, changes = _pull(name, did, prefer=prefer)
    if remote is None:
        return None
    if changes:
        _rehydrate_loaded(did)
        from vtscore.host.achievements import record_achievement  # noqa: PLC0415

        record_achievement("detector_import", did)
    schedule_labelset_push(name, detector_id=did)
    if not remote.elements and not remote.detector_meta:
        return None
    return [el.to_dict() for el in remote.elements]


def reconcile_labelset_source(detector_name: str, *, detector_id: str = "") -> int:
    """Bring a detector and its labelset source level with each other.

    Run when a detector loads and when its link changes: pulls what the source
    holds into the file (the file's label wins a disagreement - it is never
    older than what it pushed), then schedules a push so the source gains
    anything only the file had.  This is how a detector whose server lost its
    data comes back: re-create it under the same name and the pull restores
    the labels its source kept.  Never raises.

    Does nothing at all for every role but a plain binding
    (:func:`sync_role`): an owner's file is authoritative and only ever pushed
    as it changes, a received detector is levelled by a refresh, and a
    sharing binding whose source is not active is left alone.

    Returns:
        How many labels the pull added or changed.
    """
    try:
        entry, did = _identify(detector_name, detector_id)
        role = sync_role(entry, detector_id=did)
        if role == SYNC_ROLE_UNAVAILABLE:
            _note_unavailable(entry)
        if role != SYNC_ROLE_PLAIN:
            return 0
        _remote, changes = _pull(detector_name, detector_id, prefer="local")
        schedule_labelset_push(detector_name, detector_id=detector_id)
        return changes
    except Exception:  # noqa: BLE001
        logger.exception("Failed to reconcile detector %r with its labelset source", detector_name)
        return 0
