"""Per-dataset blur state: whether each item of a dataset shows blurred.

An item's **effective** blur state is

* its *override*, if a user ever set one (the Blur / Unblur button), else
* whether any blur rule matches its raw ``custom_metadata``
  (:func:`vtscore.datasets.blur_rules.matches_blur_rules`).

Only the overrides are stored. The rules' verdict is never copied into a
stored bit, so changing the rules re-decides every item nobody has overruled
and leaves every overruled item alone. An override is a plain boolean, so
"the ingest called this sensitive and it is not" and "the ingest missed this
one" are the same write.

The state is one per **dataset**, shared by everyone who can read it; it has
nothing to do with a detector. It lives on the
:class:`~vtscore.state.DatasetContext`:

* ``_blur_overrides`` / ``_blur_overrides_revision`` (state): the
  ``{media_id: bool}`` overrides, read from the sidecar the first time
  anything asks (``None`` until then), and a counter bumped on every write.
  The dict is replaced on each write, never mutated, so a reader can hold it
  without a lock.
* ``_blurred_ids`` (a derived cache, the ``blur`` family): the ids whose
  effective state is blurred, with the key it was computed at - the
  ``media_revision``, the overrides revision and the rules. Any of the three
  moving invalidates it.

The overrides persist in a small JSON sidecar beside the dataset container,
``<stem>.blur.json``, holding ``{"<media_id>": true|false}``. Media ids are per
dataset file, so the sidecar is valid for exactly the file it sits beside, and
sharing the container's stem means ``registry.unregister_dataset`` sweeps it
with the dataset. It is not a container entry because replacing one rewrites
the whole archive, far too heavy for a click.

Writes leave the request thread: a click records the override in memory and
queues the sidecar's new text, and a timer lands it :data:`WRITE_DELAY_SECONDS`
later, atomically, so a burst of clicks coalesces into one write of the newest
state. A read of the sidecar sees a queued write before it lands, as the
detector writer's reads do (``vtscore.detectors.store``), so closing and
reopening a dataset inside the delay loses nothing. Queued writes are landed at
exit; :func:`flush_blur_writes` lands them on demand.

**Carrying overrides into a new dataset** (#5693). A flow that makes a new
dataset from an old one (a pickle export and re-import, promote, combine)
renumbers the media ids, often more than once on the way: the loader keeps a
file's ids, the pipeline renumbers its chunks, combine renumbers its merge. So
between reading a file and saving the new dataset, an override travels *on its
item*, as that media's :data:`BLUR_OVERRIDE_FIELD`, and every renumbering
carries it for free. The same field is how an exported container holds them
(:func:`~vtscore.datasets.loader.export_dataset_to_file` folds them in, one
per entry). The field never reaches a saved dataset's live medias: the save
takes it off (:func:`carry_blur_overrides`) and writes the sidecar instead
(:func:`write_blur_sidecar`), so a registered dataset's overrides still live
in exactly one place. A file with a sidecar beside it is read with the
sidecar winning over its own fields (:func:`stamp_blur_overrides`).

This module stores no vectors: the overrides are ids and booleans.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import threading
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from vtscore.datasets.blur_rules import matches_blur_rules
from vtscore.utils.io import atomic_write_json

if TYPE_CHECKING:
    from vtscore.state.dataset_context import DatasetContext

logger = logging.getLogger(__name__)

__all__ = [
    "BLUR_OVERRIDE_FIELD",
    "BLUR_SIDECAR_SUFFIX",
    "WRITE_DELAY_SECONDS",
    "blur_overrides",
    "blur_sidecar_path",
    "blurred_ids",
    "carry_blur_overrides",
    "effective_blurred",
    "flush_blur_writes",
    "pending_blur_writes",
    "read_blur_overrides",
    "removing_blur_sidecar",
    "set_blur_override",
    "stamp_blur_overrides",
    "take_blur_overrides",
    "write_blur_sidecar",
]

#: What a dataset container's blur sidecar is named after its stem.
BLUR_SIDECAR_SUFFIX = ".blur.json"

#: The field an item's override rides in, ``True`` / ``False``, while it is
#: carried into a new dataset: on a media dict between reading a file and
#: saving the new dataset, and on a container entry of an exported file.  A
#: saved dataset's live medias never carry it (see the module docstring).
BLUR_OVERRIDE_FIELD = "blur_override"

#: How long a queued sidecar write waits before it lands, so a burst of clicks
#: becomes one write.  Read when a write is queued.
WRITE_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# The effective state
# ---------------------------------------------------------------------------


def effective_blurred(
    override: bool | None, rules: Iterable[Mapping[str, Any]], custom_metadata: Mapping[Any, Any] | None
) -> bool:
    """One item's effective blur state: its *override* when it has one, else whether a rule matches.

    *override* is the stored override, or ``None`` when nobody set one.
    *rules* must be well formed (the app's ``settings.get_blur_rules()`` is),
    and *custom_metadata* is the media's own raw ``custom_metadata``.
    """
    if override is not None:
        return bool(override)
    return matches_blur_rules(rules, custom_metadata)


def _rules_key(rules: Iterable[Mapping[str, Any]]) -> str:
    """A value that changes whenever the rules do, for the blurred-id cache key."""
    return json.dumps(list(rules), sort_keys=True, default=repr)


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------


def blur_sidecar_path(pkl_path: str | Path) -> Path:
    """The blur sidecar of the dataset container at *pkl_path*: ``<stem>.blur.json`` beside it."""
    p = Path(pkl_path)
    return p.parent / f"{p.stem}{BLUR_SIDECAR_SUFFIX}"


def _parse_overrides(raw: Any, source: Path) -> dict[int, bool]:
    """The ``{media_id: bool}`` a sidecar's JSON holds; a malformed entry is skipped with a warning."""
    if not isinstance(raw, dict):
        logger.warning("Ignoring blur sidecar %s: expected a JSON object, got %s", source, type(raw).__name__)
        return {}
    overrides: dict[int, bool] = {}
    for key, value in raw.items():
        try:
            media_id = int(key)
        except (TypeError, ValueError):
            media_id = None
        if media_id is None or not isinstance(value, bool):
            logger.warning("Ignoring blur sidecar entry %r: %r in %s", key, value, source)
            continue
        overrides[media_id] = value
    return overrides


def _serialise(overrides: Mapping[int, bool]) -> dict[str, bool]:
    return {str(media_id): bool(overrides[media_id]) for media_id in sorted(overrides)}


def read_blur_overrides(pkl_path: str | Path) -> dict[int, bool]:
    """The overrides stored for the dataset container at *pkl_path*.

    A write still queued for its sidecar is what is returned, so a reader
    never sees the file lag a click. No sidecar is no overrides; an unreadable
    or malformed one is logged and read as none (the next write replaces it).
    """
    path = blur_sidecar_path(pkl_path)
    with _queue_lock:
        pending = _pending.get(path)
    if pending is not None:
        return dict(pending.overrides)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        logger.warning("Failed to read the blur sidecar %s", path, exc_info=True)
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Ignoring the blur sidecar %s: it is not valid JSON", path)
        return {}
    return _parse_overrides(raw, path)


# ---------------------------------------------------------------------------
# The overrides on a context
# ---------------------------------------------------------------------------

#: Serialises override writes and the first read of a context's sidecar, so
#: two clicks cannot both build on the same old dict and lose one of them, and
#: the sidecar text queued last is always the newest.
_overrides_lock = threading.Lock()


def _container_path(ctx: DatasetContext) -> str | None:
    """The container registered for *ctx*'s dataset, or ``None`` for a dataset with no file.

    A dataset built purely in memory (a test's, or one still being ingested
    under its provisional id) has no registry entry: its overrides live in
    memory only, until the import's save writes them beside the new file
    (:func:`carry_blur_overrides`) or the first write after it is registered
    persists them all.
    """
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    try:
        entry = get_dataset(ctx.dataset_id)
    except Exception:
        logger.warning("Could not look up the container of dataset %s", ctx.dataset_id, exc_info=True)
        return None
    return (entry or {}).get("pkl_path") or None


def _loaded_overrides(ctx: DatasetContext, pkl_path: str | None) -> Mapping[int, bool]:
    """*ctx*'s overrides, reading its sidecar the first time.  Caller holds ``_overrides_lock``."""
    overrides = ctx._blur_overrides
    if overrides is None:
        overrides = read_blur_overrides(pkl_path) if pkl_path else {}
        ctx._blur_overrides = overrides
    return overrides


def blur_overrides(ctx: DatasetContext) -> Mapping[int, bool]:
    """Every override stored for *ctx*'s dataset, ``{media_id: blurred}``.

    Read from the dataset's sidecar the first time, then held on the context.
    A dataset with no container (built in memory) starts with none. The
    mapping is never mutated afterwards: a write replaces it.
    """
    pkl_path = _container_path(ctx) if ctx._blur_overrides is None else None
    with _overrides_lock:
        return _loaded_overrides(ctx, pkl_path)


def set_blur_override(ctx: DatasetContext, media_id: int, blurred: bool) -> bool:
    """Record that *media_id* of *ctx*'s dataset is *blurred* (or not), for good.

    The override wins over the rules from now on, in either direction, and
    persists in the dataset's sidecar (queued; see :data:`WRITE_DELAY_SECONDS`).
    Setting the override an item already has changes nothing and writes
    nothing. The caller checks that *media_id* is one of the dataset's items.

    Returns whether anything changed.
    """
    blurred = bool(blurred)
    pkl_path = _container_path(ctx)
    with _overrides_lock:
        current = _loaded_overrides(ctx, pkl_path)
        if current.get(media_id) is blurred:
            return False
        updated = dict(current)
        updated[media_id] = blurred
        ctx._blur_overrides = updated
        ctx._blur_overrides_revision += 1
        # Queued under the lock, so the text queued last is the newest state.
        if pkl_path:
            _queue_write(pkl_path, updated)
    return True


# ---------------------------------------------------------------------------
# The blurred-id set
# ---------------------------------------------------------------------------


class _BlurredIds(NamedTuple):
    """What ``DatasetContext._blurred_ids`` caches: the set and the key it was computed at."""

    key: tuple[int, int, str]
    ids: frozenset[int]


def blurred_ids(ctx: DatasetContext, rules: Iterable[Mapping[str, Any]]) -> frozenset[int]:
    """The ids of *ctx*'s items whose effective blur state is blurred, under *rules*.

    Cached on the context and recomputed only when its medias, its overrides
    or the rules change. An override naming an id the dataset no longer holds
    is ignored. *rules* must be well formed (the app's
    ``settings.get_blur_rules()`` is).
    """
    from vtscore.state.primitives import _state_lock  # noqa: PLC0415

    rules = list(rules)
    pkl_path = _container_path(ctx) if ctx._blur_overrides is None else None
    with _overrides_lock:
        overrides = _loaded_overrides(ctx, pkl_path)
        overrides_revision = ctx._blur_overrides_revision
    # Taken before the medias are read: a mutation in between then leaves the
    # cached set keyed on the older revision, so it is recomputed next time
    # rather than served as current.
    key = (ctx.media_revision, overrides_revision, _rules_key(rules))
    cached = ctx._blurred_ids
    if cached is not None and cached.key == key:
        return cached.ids
    with _state_lock:
        medias = dict(ctx.medias)
    if rules:
        ids = frozenset(
            media_id
            for media_id, media in medias.items()
            if effective_blurred(overrides.get(media_id), rules, media.get("custom_metadata"))
        )
    else:
        # No rules: only an override can blur an item.
        ids = frozenset(media_id for media_id, blurred in overrides.items() if blurred and media_id in medias)
    # One slot holding key and set together, so two racing computations can
    # never leave one's set under the other's key.
    ctx._blurred_ids = _BlurredIds(key, ids)
    return ids


# ---------------------------------------------------------------------------
# The sidecar writer
# ---------------------------------------------------------------------------


class _PendingWrite:
    __slots__ = ("overrides", "pkl_path")

    def __init__(self, pkl_path: Path, overrides: Mapping[int, bool]) -> None:
        self.pkl_path = pkl_path
        self.overrides = overrides


#: Guards ``_pending`` and ``_timer``; held only briefly, never across a write.
_queue_lock = threading.Lock()
#: Sidecar path -> the newest overrides queued for it and not yet on disk.
_pending: dict[Path, _PendingWrite] = {}
#: The armed timer that lands the queue, or ``None`` when none is armed.
_timer: threading.Timer | None = None
#: Serialises physical writes, and holds them off while a dataset's files are
#: deleted (:func:`removing_blur_sidecar`).
_writer_lock = threading.RLock()


def _queue_write(pkl_path: str | Path, overrides: Mapping[int, bool]) -> None:
    """Queue *overrides* as the sidecar text of the container at *pkl_path*, and arm the timer."""
    global _timer
    path = blur_sidecar_path(pkl_path)
    with _queue_lock:
        _pending[path] = _PendingWrite(Path(pkl_path), overrides)
        if _timer is None:
            _timer = threading.Timer(WRITE_DELAY_SECONDS, _on_timer)
            _timer.daemon = True
            _timer.start()


def _on_timer() -> None:
    global _timer
    with _queue_lock:
        # Only this timer's own reference: a reset may have dropped it and a
        # later write armed a new one, which must stay cancellable.
        if _timer is threading.current_thread():
            _timer = None
    flush_blur_writes()


def pending_blur_writes() -> list[Path]:
    """The sidecar paths with a write queued and not yet on disk."""
    with _queue_lock:
        return list(_pending)


def flush_blur_writes(pkl_path: str | Path | None = None) -> None:
    """Land every queued sidecar write (or just the one for *pkl_path*) now, on the calling thread."""
    only = blur_sidecar_path(pkl_path) if pkl_path is not None else None
    with _writer_lock:
        while True:
            with _queue_lock:
                if only is not None:
                    pend = _pending.get(only)
                    item = (only, pend) if pend is not None else None
                else:
                    item = next(iter(_pending.items()), None)
            if item is None:
                return
            path, pend = item
            try:
                # A dataset deleted while its write waited has no file to sit
                # beside: landing the write would leave an orphan sidecar.
                if pend.pkl_path.is_file():
                    atomic_write_json(path, _serialise(pend.overrides))
            except Exception:  # noqa: BLE001 - the next write carries the whole state again
                logger.exception("Failed to write the blur sidecar %s", path)
            # The entry stays queued while it is written, so a concurrent read
            # sees it rather than the older file; it goes only if nothing newer
            # was queued meanwhile, else the loop writes again.
            with _queue_lock:
                if _pending.get(path) is pend:
                    del _pending[path]


@contextlib.contextmanager
def removing_blur_sidecar(pkl_path: str | Path) -> Iterator[None]:
    """Drop any write queued for *pkl_path*'s sidecar, and hold the writer off for the block.

    For deleting a dataset's files: inside the block the caller can remove the
    container and its sidecars without a queued write landing in between and
    putting the sidecar back. A write queued later finds the container gone
    and is skipped.
    """
    path = blur_sidecar_path(pkl_path)
    with _writer_lock:
        with _queue_lock:
            _pending.pop(path, None)
        yield


# ---------------------------------------------------------------------------
# Carrying overrides into a new dataset
# ---------------------------------------------------------------------------


def stamp_blur_overrides(medias: Mapping[int, MutableMapping[str, Any]], overrides: Mapping[int, bool]) -> None:
    """Put each of *overrides* on its item of *medias*, as that media's :data:`BLUR_OVERRIDE_FIELD`.

    *overrides* is keyed by the ids *medias* has now; one naming an id
    *medias* does not hold is dropped. An item's existing field is replaced,
    so stamping a file's sidecar over the fields its container carried lets
    the sidecar win, as it does for a dataset that is open.
    """
    for media_id, blurred in overrides.items():
        media = medias.get(media_id)
        if media is not None:
            media[BLUR_OVERRIDE_FIELD] = bool(blurred)


def take_blur_overrides(medias: Mapping[int, MutableMapping[str, Any]]) -> dict[int, bool]:
    """Take the carried :data:`BLUR_OVERRIDE_FIELD` off every item of *medias*; return ``{media_id: blurred}``.

    Keyed by the ids *medias* has now, so whatever renumbering the items went
    through on the way is already applied. A field that is not a boolean is
    removed and ignored.
    """
    overrides: dict[int, bool] = {}
    for media_id, media in medias.items():
        value = media.pop(BLUR_OVERRIDE_FIELD, None)
        if isinstance(value, bool):
            overrides[media_id] = value
    return overrides


def carry_blur_overrides(ctx: DatasetContext) -> dict[int, bool]:
    """Make the overrides *ctx*'s items carry its own, and return every override it holds.

    For the save at the end of an import: the carried fields come off *ctx*'s
    medias (:func:`take_blur_overrides`) and become the context's overrides.
    One already set on *ctx* in memory during the import is newer than
    anything carried in, so it wins. The caller writes what this returns
    beside the saved file (:func:`write_blur_sidecar`).
    """
    carried = take_blur_overrides(ctx.medias)
    with _overrides_lock:
        held = ctx._blur_overrides or {}
        if not carried:
            return dict(held)
        merged = {**carried, **held}
        ctx._blur_overrides = merged
        ctx._blur_overrides_revision += 1
        return dict(merged)


def write_blur_sidecar(pkl_path: str | Path, overrides: Mapping[int, bool]) -> None:
    """Make the sidecar of the container just written at *pkl_path* hold exactly *overrides*, now.

    For a flow that saves a new dataset: its overrides land beside the file
    before the dataset is registered, so it never opens without them. The
    write is synchronous and atomic and replaces any write queued for that
    sidecar. No overrides leaves no sidecar, so a dataset without any is saved
    exactly as before. A failure propagates: a save that cannot keep the
    overrides should fail rather than drop them silently.
    """
    path = blur_sidecar_path(pkl_path)
    with _writer_lock:
        with _queue_lock:
            _pending.pop(path, None)
        if overrides:
            atomic_write_json(path, _serialise(overrides))
        else:
            path.unlink(missing_ok=True)


def reset_blur_writes_for_tests() -> None:
    """Cancel the armed timer and drop every queued write without landing it (for conftest).

    A write in flight is waited out first, so nothing one test queued lands in
    the next.
    """
    global _timer
    with _writer_lock, _queue_lock:
        if _timer is not None:
            _timer.cancel()
            _timer = None
        _pending.clear()


# Land the newest overrides on a normal exit; a hard kill loses at most the
# last :data:`WRITE_DELAY_SECONDS` of clicks.
atexit.register(flush_blur_writes)
