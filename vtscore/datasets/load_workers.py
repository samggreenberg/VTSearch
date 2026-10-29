"""Dataset-load workers: the concurrency gates and the per-task harness every import flow runs on.

The two import pipelines — a full dataset load
(:mod:`vtscore.datasets.load_origin`) and a combine-flow staging import
(:mod:`vtscore.datasets.load_staging`) — plus the multi-dataset load
(:mod:`vtscore.datasets.load_multi`) share everything here: the download and
embed concurrency gates (:class:`~vtscore.concurrency.gate.ConcurrencyGate`)
and the controller that hands a load from one to the other, the task harness that
registers a load on ``loading_tasks`` and runs it on a daemon thread, the
request-field parsers, and the failure / terminal-state handling.

**Rebound globals live here and are read here.**
:func:`reset_load_gates_for_tests` rebinds ``_download_gate`` /
``_embed_gate``, and :func:`register_last_embedder_persistence_hook` rebinds
``_last_embedder_persistence_hook``.  A ``from … import`` of any of the three
is a copy frozen at import time, so every reader outside this module goes
through the module (``load_workers._embed_gate``), and the gates stay beside
the :class:`_LoadGateController` that reads them.
"""

from __future__ import annotations

import gc
import json
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from vtscore.config import get_setting
from vtscore.concurrency.gate import ConcurrencyGate
from vtscore.concurrency.progress import (
    CANCELLED_ERROR,
    CancelledError,
    clear_thread_progress,
    loading_tasks,
    set_thread_progress,
)
from vtscore.datasets.registry import unregister_dataset as _reg_unregister
from vtscore.datasets.stages._common import _STATUS_TO_STEP, _TOTAL_LOAD_STEPS, _origin_to_str
from vtscore.state import DatasetContext, clear_all


# Two independent gates control how many dataset loads can run concurrently
# in each phase.  The download/import phase is bandwidth- and disk-bound;
# the embedding phase is CPU/GPU- and RAM-bound.  Splitting the gates lets
# one dataset download while another is still embedding, instead of forcing
# strict end-to-end serialisation.  Limits are user-configurable via the
# ``max_concurrent_dataset_downloads`` and ``max_concurrent_dataset_embeddings``
# settings; defaults derive from the host's CPU/GPU counts (see
# :func:`vtscore.embedding.loader.default_concurrent_downloads` and
# :func:`vtscore.embedding.loader.default_concurrent_embeddings`).
_download_gate = ConcurrencyGate(lambda: get_setting("max_concurrent_dataset_downloads"))
_embed_gate = ConcurrencyGate(lambda: get_setting("max_concurrent_dataset_embeddings"))


def reset_load_gates_for_tests() -> None:
    """Rebind both load gates to fresh instances.  For test isolation.

    The gates are process globals, so under ``pytest -n auto`` every test in a
    worker shares them.  A test that leaves a background load thread running
    leaves that thread holding a permit, and the next test in the process sees
    a gate that is already partly full — which is how ``TestLoadingGates``
    became a lottery over xdist worker assignment (issue #3613).

    Rebinding rather than zeroing the counters is what makes this safe in the
    presence of a thread we could not stop: the leaked thread still holds — and
    eventually releases — the *old* gate object, so it cannot drive the new
    gate's count negative.  :class:`_LoadGateController` cooperates by
    remembering the gate object it acquired instead of re-resolving these
    globals at release time.
    """
    global _download_gate, _embed_gate

    _download_gate = ConcurrencyGate(lambda: get_setting("max_concurrent_dataset_downloads"))
    _embed_gate = ConcurrencyGate(lambda: get_setting("max_concurrent_dataset_embeddings"))


# ---------------------------------------------------------------------------
# App-side persistence hook
# ---------------------------------------------------------------------------
# The library remembers the user's per-media-type embedder pick by calling
# whatever the app installs here.  Default is a no-op so this module doesn't
# need to import ``vtsearch.settings`` (the Settings seam in
# ``../docs/architecture.md``).  ``vtsearch/shim/`` registers the
# real implementation; ``vtsearch.settings.set_last_embedder_for_media_type``
# is wired at app startup.
_last_embedder_persistence_hook: Callable[[str, str], None] | None = None


def register_last_embedder_persistence_hook(fn: Callable[[str, str], None]) -> None:
    """Install the callback used to persist the user's per-media-type embedder pick.

    The Flask app installs ``vtsearch.settings.set_last_embedder_for_media_type``
    as the hook at startup so library callers don't have to know about the
    user-pref persistence layer.  Library-only callers can leave the default
    in place (no persistence).
    """
    global _last_embedder_persistence_hook
    _last_embedder_persistence_hook = fn


def clear_dataset():
    """Clear the current dataset, votes, and all related state."""
    clear_all()


def _get_embedder_for_medias(media_dict: dict):
    """Resolve the embedder for *media_dict*.

    Imported lazily to keep this module's import graph shallow;
    ``vtscore.media`` pulls in the whole media-plugin discovery pass.
    """
    from vtscore.media import embedder_for_medias  # noqa: PLC0415

    return embedder_for_medias(media_dict)


def _recorded_embedder_name(media_dict: dict, requested: str) -> str:
    """The embedder name a timing row should carry for this load.

    The caller's pick wins when there is one. When there is not, the load still
    *used* an encoder — the media type's default, resolved deep inside the embed
    stage and stamped onto every media — and recording the blank instead files
    the run under the media rollup, so the exact ``(device, media, embedder)``
    cell the load cost model is keyed on can never be populated by an import
    that did not name one (#3345). Reads the name off the medias rather than resolving an
    embedder object, because by this point they carry it and the object is not
    needed.
    """
    if requested:
        return requested
    if not media_dict:
        return ""
    first = next(iter(media_dict.values()), None) or {}
    return str(first.get("embedder", "") or "")


def _parse_bool(value: Any) -> bool:
    """Coerce a request-supplied flag to ``bool``.

    Thin alias for :func:`vtscore.plugins.parse_checkbox`, kept so the many
    call sites in the load modules read unchanged; the coercion itself is shared
    with the CLI so the two cannot drift.
    """
    from vtscore.plugins import parse_checkbox  # noqa: PLC0415

    return parse_checkbox(value)


def _parse_embedder_list(value: Any) -> list[str] | None:
    """Coerce a request-supplied embedder list to ``list[str]`` (or ``None``).

    The v3 create-time three-role picker sends the bound trio (text / patch /
    structural picks, deduped) under the ``embedders`` field.  It arrives as a
    native list on JSON-body routes and as a string on multipart routes (a JSON
    array string, or a comma-separated fallback).  ``None`` / empty in →
    ``None`` (the caller falls back to the single ``embedder`` field — the
    pre-trio path).  Order is preserved and blanks/dupes are dropped.
    """
    if value is None:
        return None
    items: list[Any]
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else [parsed]
        except (TypeError, ValueError):
            items = text.split(",")
    out: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out or None


def _normalize_media_type(value: str) -> str:
    """Normalize a media type string (folder_import_name or type_id) to a canonical type_id."""
    value = (value or "").strip()
    if not value:
        return ""
    try:
        from vtscore.media import get_by_folder_name, normalize_type_id  # noqa: PLC0415

        try:
            return get_by_folder_name(value).type_id
        except KeyError:
            return normalize_type_id(value)
    except Exception:
        return value


def _parse_chain_field(raw: Any) -> list[dict] | None:
    """Decode a ``clipper_chain`` importer field value into a step list.

    The field may arrive as a JSON string (typical client encoding) or as
    a native list (programmatic callers). Returns ``None`` for missing /
    malformed values so the legacy single-clipper path stays in effect.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        import json as _json

        try:
            decoded = _json.loads(raw)
        except (TypeError, ValueError):
            return None
        if isinstance(decoded, list):
            return decoded
        return None
    return None


# ---------------------------------------------------------------------------
# Parallel-safe background loading
# ---------------------------------------------------------------------------


@dataclass
class _ImportTask:
    """Everything :func:`_start_import_task` sets up before the worker runs.

    ``tracker`` is the per-task :class:`~vtscore.concurrency.progress.ProgressTracker`
    (registered under ``task_id`` on ``loading_tasks``), and ``request_user`` is
    the caller identity the worker replays so per-user settings writes land in the
    right file.  ``total_steps`` is the flow's step structure, carried here so
    everything downstream (notably :class:`_LoadGateController`) reports the
    same shape the tracker was built with.
    """

    task_id: str
    tracker: Any
    request_user: str
    total_steps: int


def _start_import_task(
    *,
    prefix: str,
    display_name: str,
    total_steps: int,
    media_type: str = "",
    embedder: str = "",
    weights: list[float] | None = None,
    extra_fields: dict[str, Any] | None = None,
    created_by: str = "",
    source: str = "",
) -> _ImportTask:
    """Register a background import on ``loading_tasks``.

    The two import pipelines — a full dataset load
    (:mod:`~vtscore.datasets.load_origin`) and a combine-flow staging import
    (:mod:`~vtscore.datasets.load_staging`) — open identically: mint a task id, create the per-task
    tracker (so two concurrent imports never interleave one channel), and
    snapshot the user who asked for the work.  Only the step structure and the
    tracker's extra fields differ, so they are parameters here.

    The tracker publishes no remaining-time estimate (``publish_eta=False``).
    An import's cost is set by the network, the source's disks and the files
    themselves, and its estimate swung too wildly to be worth showing (#4667);
    the bar itself still fills.

    A failed import keeps its row until the user dismisses it
    (``keep_failure``): no dataset row is left to say it happened (#4973).
    *source* says what was being imported (the origin, as the user would name
    it), so that row can show what the import was reading.

    The caller writes its own first ``tracker.update`` (rather than this
    function writing a generic one) because the load flow subscribes its
    env-gated phase profiler in between: a profiler that missed the step-1
    update would date the download phase from the wrong instant.
    """
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    task_id = f"{prefix}{uuid4().hex[:8]}"
    request_user = created_by or get_current_user()
    tracker = loading_tasks.create_task(
        task_id,
        display_name,
        media_type=media_type,
        embedder=embedder,
        extra_fields=extra_fields,
        step_weights=weights,
        publish_eta=False,
        created_by=request_user,
        source=source,
        keep_failure=True,
    )
    return _ImportTask(
        task_id=task_id,
        tracker=tracker,
        request_user=request_user,
        total_steps=total_steps,
    )


def _describe_source(origin: dict | Callable[[], dict]) -> str:
    """Name what an import reads, for its task's ``source``; ``""`` if that fails.

    *origin* may be a callable building it, for the staging flow, which has no
    origin of its own to hand.  Either way the description runs plugin code
    (``build_origin`` / ``origin_display``), and a label for a failure row is
    not worth failing the import over.
    """
    try:
        return _origin_to_str(origin() if callable(origin) else origin)
    except Exception:
        return ""


def _spawn_import_worker(task: _ImportTask, body: Callable[[], None]) -> str:
    """Run *body* on a daemon thread scoped to *task*; return its ``task_id``.

    Closes the harness :func:`_start_import_task` opens.  *body* is a zero-arg
    closure over whatever the pipeline needs; everything around it is identical
    for both flows:

    * the requesting user is replayed into the worker, so background per-user
      settings writes resolve to the right file;
    * the thread-local progress hook is cleared however the body exits, so a
      later job on this thread cannot narrate itself on a dead channel;
    * ``mark_finished`` runs last, so a caller waiting on
      ``has_active_tasks() == False`` (or ``is_finished``) sees fully
      cleaned-up worker state — gates released, contexts restored;
    * the thread is registered with ``set_worker`` *before* it starts, so a
      cancel arriving in the same instant can tell "not started yet" from
      "nothing here" (#3167).
    """

    def run() -> None:
        from vtscore.state.current_user import thread_user  # noqa: PLC0415

        try:
            with thread_user(task.request_user):
                body()
        finally:
            clear_thread_progress()
            loading_tasks.mark_finished(task.task_id)

    worker = threading.Thread(target=run, daemon=True)
    loading_tasks.set_worker(task.task_id, worker)
    worker.start()
    return task.task_id


class _LoadGateController:
    """Tracks which import gate (download / embed) is currently held.

    Splits gate-acquisition concerns out of the task body: the importer
    runs under the download gate (bandwidth-bound), and we swap to the
    embed gate as soon as the importer signals it's started embedding so
    another dataset can begin downloading in parallel.

    Both import pipelines use it, so *total_steps* is a parameter rather than
    ``_TOTAL_LOAD_STEPS``: the "waiting for other datasets" messages have to
    carry the step structure of whichever flow is queueing, or the whole-job
    bar rescales the moment a load parks on a full gate.
    """

    def __init__(self, tracker, total_steps: int = _TOTAL_LOAD_STEPS) -> None:
        self._tracker = tracker
        self._total_steps = total_steps
        self._held: str | None = None
        #: The gate object :meth:`acquire` last took a permit from.  Held as an
        #: object rather than re-resolved from the module global at release
        #: time so a controller always releases *the gate it acquired*, even if
        #: the global has since been rebound (which is what
        #: :func:`reset_load_gates_for_tests` does between tests).
        self._held_gate: ConcurrencyGate | None = None

    @property
    def held(self) -> str | None:
        return self._held

    def acquire(self, gate: ConcurrencyGate, name: str, wait_msg: str) -> None:
        if gate.acquire(blocking=False):
            self._held = name
            self._held_gate = gate
            return
        self._tracker.update("loading", wait_msg, 0, 0, step=1, total_steps=self._total_steps)
        while not gate.acquire(timeout=0.5):
            self._tracker.check_cancelled()
        self._held = name
        self._held_gate = gate

    def acquire_download(self) -> None:
        self.acquire(_download_gate, "download", "Waiting for other datasets to finish downloading…")

    def swap_to_embed(self) -> None:
        if self._held == "embed":
            return
        if self._held == "download":
            self.release()
        self.acquire(_embed_gate, "embed", "Waiting for other datasets to finish embedding…")

    def release(self) -> None:
        if self._held_gate is not None:
            self._held_gate.release()
        self._held = None
        self._held_gate = None


def _make_stepped_progress(controller: _LoadGateController, pacer):
    """Build the importer-side progress callback.

    Routes status updates into the load's :class:`AdaptiveLoadPacer` (which
    maps them onto the unified bar) with the right step number, and triggers
    the download→embed gate swap on the first ``"embedding"`` status so a
    queued download can start.
    """

    def stepped(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        pacer.check_cancelled()
        if status == "idle":
            return
        if status == "embedding" and controller.held != "embed":
            controller.swap_to_embed()
        step = _STATUS_TO_STEP.get(status)
        pacer.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)

    return stepped


def _run_importer(load_fn, ctx: DatasetContext, stepped) -> None:
    """Invoke *load_fn* under thread-local progress, populating ctx.medias."""
    import inspect  # noqa: PLC0415

    set_thread_progress(stepped)
    try:
        sig = inspect.signature(load_fn)
        if sig.parameters:
            load_fn(ctx.medias)
        else:
            load_fn()
    finally:
        clear_thread_progress()


def _tag_origins(media_dict: dict, origin: dict) -> None:
    """Stamp *origin* onto medias that don't already carry one.

    Each media gets its own fresh copy of the origin dict (including a
    fresh ``params``).  Sharing one dict by reference across siblings
    means any later mutation of ``media["origin"]["params"]`` on one
    media silently corrupts every other media stamped by the same load;
    and that aliasing also survives pickle round-trips via backreferences.
    """
    for media in media_dict.values():
        if media.get("origin") is None:
            media["origin"] = {
                "importer": origin.get("importer", ""),
                "params": dict(origin.get("params", {})),
            }
        if not media.get("origin_name"):
            media["origin_name"] = media.get("filename", "")


def _warmup_embedder_async(media_dict: dict) -> None:
    """Warm up the embedder (model load + text-encoder prime) in a daemon thread.

    Fire-and-forget: the caller doesn't wait, and there is no progress
    surface; the dataset is usable for grid-browsing immediately, and
    text sort waits behind its own ``_embedder_load_lock`` (see
    ``vtsearch/routes/sorting.py:_load_embedder_with_progress``) on first
    use.  ``MediaEmbedder.load_models`` is idempotent and serialised by
    a per-class lock, so racing this thread against an on-demand sort
    load is safe.
    """

    def _run() -> None:
        emb = _get_embedder_for_medias(media_dict)
        if emb is None:
            return
        try:
            # Explicitly silent: this thread has no progress surface (see the
            # docstring), and an unscoped load would narrate itself on the
            # global dataset channel instead — a phantom import, right as the
            # real one finished (#3167).
            with emb.silent_progress():
                emb.load_models()
                emb.embed_text("warmup")
        except Exception:
            pass

    threading.Thread(target=_run, name="warmup-embedder", daemon=True).start()


#: What a cancelled import's tracker carries as its ``error``; see :func:`_failure_message`.
_CANCELLED_MESSAGE = CANCELLED_ERROR


def _failure_message(exc: BaseException, fallback: str) -> str:
    """Map a background-import exception onto the string shown to the user.

    Both import pipelines (a dataset load and a combine-flow staging import)
    run the same kinds of work — an importer, then an embed — so they fail in
    the same ways and must say the same things.  ``"Cancelled"`` in particular
    is load-bearing rather than cosmetic: ``dashboard-loading-tasks.service.ts``
    and ``toast.service.ts`` both test for that exact string to tell a
    user-requested stop from a genuine failure, so a flow that surfaced
    ``CancelledError``'s own text instead ("Operation cancelled by user") popped
    a red *failed* toast for a cancel the user asked for.

    *fallback* is the last-resort text for an exception carrying no message.
    The traceback is printed for the two cases where the server log is the only
    place the detail survives; a cancel and an OOM carry all they have in the
    returned string.
    """
    if isinstance(exc, CancelledError):
        return _CANCELLED_MESSAGE
    if isinstance(exc, ImportError):
        traceback.print_exc()
        return f"Missing dependency: {exc}. Install all required packages with: pip install -e '.[cpu,dev]'"
    if isinstance(exc, MemoryError):
        return "Out of memory: this dataset is too large. Try a smaller dataset or free up system RAM."
    traceback.print_exc()
    return str(exc) or repr(exc) or fallback


def _handle_load_failure(
    exc: BaseException,
    context_id: str,
    tracker,
    registry_entry_id: str | None = None,
    task_id: str | None = None,
) -> None:
    """Unregister the context and write the failure into *tracker*.

    If *registry_entry_id* is set, the on-disk registry entry (and its
    backing pkl) is also removed; this prevents an orphaned dashboard
    row when a load fails after
    :func:`~vtscore.datasets.stages.registry._register_and_migrate` has
    already written the entry.  The task named *task_id* then drops the
    ``dataset_id`` it was given at registration: its failure is kept until
    dismissed, and a later import that reuses the id (a demo's) must not
    find that row attached to its own dataset.
    """
    from vtscore.state.core import unregister_context  # noqa: PLC0415

    error = _failure_message(exc, "Unknown error during dataset loading")

    unregister_context(context_id)
    if registry_entry_id:
        try:
            _reg_unregister(registry_entry_id)
        except Exception:
            traceback.print_exc()
        if task_id:
            loading_tasks.set_dataset_id(task_id, "")
    gc.collect()
    tracker.update("idle", "", 0, 0, error=error, step=None, total_steps=None)


def _park_load_terminal(tracker, n_items: int) -> None:
    """Park *tracker* at a terminal state once the load thread has unwound.

    The failure paths already do this — :func:`_handle_load_failure` writes
    ``idle`` plus an ``error`` — but the success path historically wrote
    nothing, so a load that finished cleanly left its tracker on whatever
    "loading …" message happened to fire last.  Nothing ever cleared it:
    :meth:`~vtscore.concurrency.progress.LoadingTasksTracker.has_active_tasks`
    stayed true until the finished entry aged out, the dashboard's
    just-finished test (``status == "idle" && !error``) never fired so the
    registry refresh waited on the prune instead, and every external signal
    kept saying "still importing" about a thread that had exited (#3167).

    Terminal states belong in a ``finally``.  A tracker that already carries an
    error is left alone: that failure *is* its terminal state.
    """
    if tracker.get().get("error"):
        return
    tracker.update(
        "idle",
        f"Loaded {n_items} item(s)",
        n_items,
        n_items,
        step=_TOTAL_LOAD_STEPS,
        total_steps=_TOTAL_LOAD_STEPS,
    )
