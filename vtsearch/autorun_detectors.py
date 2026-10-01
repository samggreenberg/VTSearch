"""AutoRun: score a dataset with the current user's AutoRun detectors.

A user's **AutoRun** detectors - the Dashboard's AutoRun tab, stored as the
per-user ``autofind_detectors`` setting - are the ones they have finalized to
run unattended.  This module is where the app runs them against a dataset, for
two kinds of caller:

- ``POST /api/auto-detect`` scores the request's active dataset synchronously,
  reporting on the shared Find tracker (the API / scripted path).
- :func:`start_autorun_task` scores any loaded dataset in the background, as a
  task on the ``loading-tasks`` channel keyed to that dataset, so it renders
  inline on the dataset's Dashboard row and its Cancel button works.  The
  dataset ⋯ menu's **Run AutoRun** starts one directly; a web import starts one
  once the dataset is saved, unless the user unticked **Run AutoRun** in the
  Add Dataset dialog (:func:`import_post_load`).

The CLI's ``--autodetect`` has its own streaming pipeline in :mod:`vtscore.cli`
and does not come through here.

Either way a run ends the same: the results go to the user's Auto-Find exporter
when one is configured (:func:`run_autofind_export`).  A background run's
results are also held in memory for the user who started it
(:func:`get_autorun_run`), so the browser can show them in the AutoRun Results
dialog.  They are hit lists, never vectors, and only the most recent runs are
kept (:data:`MAX_KEPT_RUNS`, :data:`MAX_KEPT_HITS`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vtscore.concurrency.memory_budget import cap_workers_by_memory
from vtscore.concurrency.progress import CancelledError, find_progress, loading_tasks
from vtscore.detectors.model_loading import resolve_or_train_detector

if TYPE_CHECKING:  # pragma: no cover - import cycle / heavy-import avoidance
    from vtscore.detectors.training import ScoringRows
    from vtscore.state.core import DatasetContext

logger = logging.getLogger(__name__)

#: Task-id prefix of a background AutoRun on the ``loading-tasks`` channel.
TASK_PREFIX = "_autorun_"

#: How many finished background runs keep their results for the results
#: dialog.  A run's hit lists cover every media in the dataset, so this is a
#: bound on memory, not a history feature.
MAX_KEPT_RUNS = 16

#: And how many hit entries (Good and Bad, across detectors) they may hold
#: between them: a few runs over a large dataset are as heavy as many small
#: ones.  The oldest runs go first; the newest is always kept, whatever its size.
MAX_KEPT_HITS = 1_000_000

#: What started a background run: a finished web import, or the dataset ⋯
#: menu's Run AutoRun.  Carried on the task so the browser can tell a run it
#: asked for (open the results) from one that happened to it (offer them).
TRIGGERS = ("import", "manual")


class AutoRunUnavailable(Exception):
    """There is nothing to run.

    *message* is user-facing; *status* is the HTTP code the API answers with
    (400 for "nothing applies to this dataset", 404 for a named detector that
    is not on the user's AutoRun list).
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class AutoRunPlan:
    """The AutoRun detectors that apply to one dataset snapshot.

    ``detectors`` holds ``(name, detector JSON, registry entry)`` triples;
    ``missing`` names AutoRun entries whose detector file no longer exists.
    """

    media_type: str
    detectors: list[tuple[str, dict, dict | None]]
    missing: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Which detectors run
# ---------------------------------------------------------------------------


def detector_type(det_data: dict | None) -> str:
    """The locked embedder type of a detector JSON (legacy-migrated)."""
    from vtscore.detectors.embedder_type import detector_embedder_type_from_data  # noqa: PLC0415

    return detector_embedder_type_from_data(det_data or {})


def dataset_supplies_detector_type(det_data: dict | None, snap: dict) -> bool:
    """Whether *snap* binds an embedder of the detector's locked type.

    A detector scores its labels in the concrete embedder of its locked type the
    dataset supplies; when the dataset binds no embedder of that type the labels
    would re-embed in a foreign space (garbage scores), so the pair is
    incompatible.  A legacy/typeless detector is always compatible (resolved at
    first train via the score precedence).
    """
    from vtscore.embedding.binding import detector_dataset_compatible  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedder_names  # noqa: PLC0415

    bound = media_embedder_names(next(iter(snap.values()), {})) if snap else []
    return detector_dataset_compatible(detector_type(det_data), bound)


def _resolve_names(detector_name: str, media_type: str) -> list[str]:
    """The AutoRun detector names to consider: all of them, or the one named."""
    from vtsearch.settings import get_autofind_detectors  # noqa: PLC0415

    names = get_autofind_detectors()
    if detector_name:
        if detector_name not in names:
            raise AutoRunUnavailable(f"Detector '{detector_name}' is not on your AutoRun list", status=404)
        return [detector_name]
    if not names:
        raise AutoRunUnavailable("You have no AutoRun detectors. Move a detector to the AutoRun tab first.")
    return names


def _collect_for_media_type(names: list[str], media_type: str) -> tuple[list[tuple[str, dict, dict | None]], list[str]]:
    """Load detector data + registry entry for each name whose media type is *media_type*.

    Returns ``(detectors, missing)``: *missing* holds names whose detector file
    no longer exists on disk (a stale AutoRun reference).  Names whose media
    type simply doesn't match are skipped without being reported - those are
    legitimately inapplicable, not broken.
    """
    from vtscore.detectors.registry import find_by_name, list_detectors  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415

    detectors: list[tuple[str, dict, dict | None]] = []
    missing: list[str] = []
    for name in names:
        det_data = _read_detector(_detector_path(name))
        if det_data is None:
            missing.append(name)
            continue
        if det_data.get("media_type", "") != media_type:
            continue
        reg_entry = find_by_name(name)
        if reg_entry is None:
            # Fallback: also accept registry entries whose name matches.
            for entry in list_detectors():
                if entry.get("name") == name:
                    reg_entry = entry
                    break
        detectors.append((name, det_data, reg_entry))
    return detectors, missing


def plan_autorun(snap: dict, *, detector_name: str = "") -> AutoRunPlan:
    """Decide which of the current user's AutoRun detectors score *snap*.

    Keeps the detectors of *snap*'s media type whose locked embedder type the
    dataset can supply.  Raises :class:`AutoRunUnavailable` when that leaves
    nothing to run.
    """
    if not snap:
        raise AutoRunUnavailable("No medias loaded")
    media_type = next(iter(snap.values())).get("media_type", "audio")
    names = _resolve_names(detector_name, media_type)
    detectors, missing = _collect_for_media_type(names, media_type)
    if not detectors:
        message = f"None of your AutoRun detectors are for {media_type} datasets."
        if missing:
            message += f" Missing detector file(s) for: {', '.join(missing)}"
        raise AutoRunUnavailable(message)
    compatible = [trip for trip in detectors if dataset_supplies_detector_type(trip[1], snap)]
    if not compatible:
        raise AutoRunUnavailable(
            "None of your AutoRun detectors can score this dataset: it has no embedder "
            "of the kind they were built with."
        )
    return AutoRunPlan(media_type=media_type, detectors=compatible, missing=missing)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_detector(
    name: str,
    det_data: dict,
    reg_entry: dict | None,
    media_type: str,
    snap: dict,
    rows: ScoringRows,
    *,
    check_cancelled: Callable[[], None] = find_progress.check_cancelled,
    on_progress: Callable[..., None] | None = None,
    use_loaded_context: bool = True,
) -> tuple[str, dict] | None:
    """Train (or reuse) one detector and score every media in *snap*.

    *rows* is the :class:`~vtscore.detectors.training.ScoringRows` stack for
    this detector's score space, built once per run and shared by every
    detector that resolves to the same embedder.  Scoring goes through
    :func:`~vtscore.detectors.training.score_rows_with_model` - the app's one
    definition of what a head scores a media at - so a patch head is pooled
    over its media's rows exactly as ``/api/find-label`` does (issue #3544).

    Returns ``None`` when the detector could not be trained or scoring failed
    (logged), so one broken detector does not sink the run.
    """
    from vtscore.detectors.training import score_rows_with_model  # noqa: PLC0415
    from vtsearch.routes._media_response import media_info_for_response  # noqa: PLC0415

    # Poll BEFORE the catch-all try: CancelledError subclasses Exception, and
    # a cancel must propagate to the caller (via future.result()), not be
    # swallowed as a failed detector.
    check_cancelled()

    try:
        detector_id = reg_entry["id"] if reg_entry else name
        trained: list = []
        mlp, threshold, _diag = resolve_or_train_detector(
            detector_id,
            det_data,
            media_type,
            snap,
            progress_step=1,
            progress_total_steps=1,
            on_progress=on_progress,
            use_loaded_context=use_loaded_context,
            ctx_sink=trained,
        )
        if mlp is None:
            return None
        floor, balance = _line_states(name, trained[0] if trained else None)

        scores, _best_row = score_rows_with_model(mlp, rows)

        positive_hits = []
        negative_hits = []
        for cid, score in zip(rows.ids, scores, strict=True):
            clip_info = media_info_for_response(snap[cid])
            clip_info["score"] = round(score, 4)
            if score >= threshold:
                positive_hits.append(clip_info)
            else:
                negative_hits.append(clip_info)

        positive_hits.sort(key=lambda x: x["score"], reverse=True)
        negative_hits.sort(key=lambda x: x["score"], reverse=True)

        return name, {
            "detector_name": name,
            "threshold": round(threshold, 4),
            # The floor's and the balance's state on that cut: unchecked,
            # headless (#4247, #4272, #4413).
            "floor": floor,
            "balance": balance,
            "total_hits": len(positive_hits),
            "hits": positive_hits,
            "negative_hits": negative_hits,
        }
    except Exception:
        logger.exception("Auto-detect failed for detector %s", name)
        return None


def _line_states(name: str, det_ctx: Any) -> tuple[dict | None, dict | None]:
    """What the floor and the balance say about *name*'s cut in an AutoRun: unchecked, because nobody can vote.

    Both ride with the cut (``floor`` and ``balance``); which of the two drew
    the line is the ``line_preference`` setting (#4413), read on the thread
    that trained it, like the floor and the beta.  A headless run cannot
    spot-check its line (#4272), so the cut exported is the preference's
    unchecked set, and the log line is the record that it was never checked.
    """
    from vtscore.state.core import detector_balance_state, detector_floor_state  # noqa: PLC0415
    from vtscore.training.thresholds import FLOOR_UNCHECKED, aim_words  # noqa: PLC0415
    from vtsearch.state import get_beta, get_line_preference, get_min_precision  # noqa: PLC0415

    if det_ctx is None:
        return None, None
    floor = detector_floor_state(det_ctx, get_min_precision())
    balance = detector_balance_state(det_ctx, get_beta())
    drawn = balance if get_line_preference() == "balance" else floor
    if drawn is not None and drawn["status"] == FLOOR_UNCHECKED:
        logger.info(
            "Auto-detect: detector %s exports its top %d unchecked (%s); nobody is here to check it",
            name,
            drawn["count"],
            aim_words(drawn),
        )
    return floor, balance


def score_autorun(
    plan: AutoRunPlan,
    snap: dict,
    *,
    check_cancelled: Callable[[], None] = find_progress.check_cancelled,
    on_progress: Callable[..., None] | None = None,
    on_detector_done: Callable[[int, int], None] | None = None,
    use_loaded_context: bool = True,
) -> dict[str, Any]:
    """Score *snap* with every detector in *plan*; return the auto-detect results dict.

    *snap* must be the medias of the dataset the calling thread resolves as
    active (the request's dataset, or the one a background run pinned): the row
    stacks built here are cached on that context, and a cache keyed on a media
    id set is only safe to reuse for the dataset it was built from.

    Detectors run in a small thread pool.  *check_cancelled* is polled by every
    worker before it trains and again as each result is collected, and raises
    :class:`CancelledError` out of this function; *on_progress* is forwarded to each cold train (see
    :func:`~vtscore.detectors.model_loading.resolve_or_train_detector`), and
    *on_detector_done* ``(done, total)`` fires as each detector's result is
    collected.  The returned dict is the ``POST /api/auto-detect`` body without
    ``auto_export``.
    """
    from vtscore.detectors.training import scoring_rows_for_snap  # noqa: PLC0415
    from vtscore.embedding.binding import keying_embedder_for_type  # noqa: PLC0415
    from vtscore.state.core import get_active_context  # noqa: PLC0415
    from vtscore.state.current_user import get_current_user, thread_user  # noqa: PLC0415

    # Each detector scores in the concrete embedder of its locked type, so build
    # one row stack per distinct embedder the detectors call for (collapsing to
    # a single shared stack on the common single-embedder dataset, where every
    # type resolves to the same name).  A legacy/typeless detector falls back to
    # the dataset score precedence, matching resolve_or_train_detector's
    # cold-train space.
    default_score = get_active_context().routed_embedder("score")
    det_embedders: dict[str, str | None] = {}
    for dname, ddata, _entry in plan.detectors:
        keyed = keying_embedder_for_type(detector_type(ddata), snap)
        det_embedders[dname] = keyed or default_score

    # These are ``scoring_rows_for_snap`` rows - the app's single definition of
    # what a detector scores a media at - not an image-level matrix: the heads
    # ``resolve_or_train_detector`` returns are the app's own labelset-trained
    # ones, which are MaxPatch heads on a patch dataset (issue #3544).  On such
    # a dataset the build is the region matrix cached on the active dataset
    # context, so one stack per space costs nothing the first vote of a session
    # would not have paid anyway.
    row_stacks: dict[str | None, ScoringRows] = {}
    for emb in set(det_embedders.values()):
        row_stacks[emb] = scoring_rows_for_snap(snap, emb)

    # Size the worker cap off the biggest stack actually built.  The cap counts
    # *rows*, not media: a patch stack is H*W+1 rows per media and that is what
    # each worker's forward pass allocates.  ``row_stacks`` is never empty:
    # ``plan_autorun`` refuses a plan with no detectors.
    cap_rows = max(int(r.matrix.shape[0]) for r in row_stacks.values())
    cap_dim = max(int(r.matrix.shape[1]) if r.matrix.ndim > 1 else 0 for r in row_stacks.values())
    worker_cap = cap_workers_by_memory(
        cap_rows,
        cap_dim,
        max_workers=min(len(plan.detectors), 8),
    )

    # Pool threads start with no user, so a cold train there would read the
    # default user's calibration settings; replay the caller's.
    user = get_current_user()

    def _score_as_caller(name: str, data: dict, entry: dict | None) -> tuple[str, dict] | None:
        with thread_user(user):
            return score_detector(
                name,
                data,
                entry,
                plan.media_type,
                snap,
                row_stacks[det_embedders[name]],
                check_cancelled=check_cancelled,
                on_progress=on_progress,
                use_loaded_context=use_loaded_context,
            )

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=worker_cap) as pool:
        futures = [pool.submit(_score_as_caller, name, data, entry) for name, data, entry in plan.detectors]
        try:
            for done, future in enumerate(futures, start=1):
                outcome = future.result()
                # Workers only poll on entry, so a cancel that lands while the
                # last detector trains would otherwise be ignored.
                check_cancelled()
                if outcome is not None:
                    name, result = outcome
                    results[name] = result
                if on_detector_done is not None:
                    on_detector_done(done, len(futures))
        except CancelledError:
            # Drop the queued workers (already-running ones finish their current
            # detector, the queued ones bail at their entry poll).
            for f in futures:
                f.cancel()
            raise

    if results:
        from vtsearch.achievements import record_find  # noqa: PLC0415

        # Each detector scored the medias in its own embedder's stack, so count
        # them per detector rather than multiplying one stack's media count out.
        record_find(sum(len(row_stacks[det_embedders[name]].ids) for name in results))

    return {
        "media_type": plan.media_type,
        "detectors_run": len(results),
        "results": results,
        "missing_detectors": plan.missing,
    }


def run_autofind_export(response: dict) -> dict | None:
    """Run the configured Auto-Find results exporter on *response*.

    Returns ``None`` when no exporter is configured (the common case), or a
    status dict ``{exporter, success, message?, error?}`` otherwise. Export
    failures are reported in the status block rather than raised: the scored
    results are valuable on their own, so a misconfigured exporter must not
    sink the whole run.
    """
    from vtsearch.settings import (  # noqa: PLC0415
        get_autofind_exporter,
        get_autofind_exporter_field_values,
    )

    exporter_name = get_autofind_exporter()
    if not exporter_name:
        return None

    from vtscore.exporters import get_exporter  # noqa: PLC0415

    exporter = get_exporter(exporter_name)
    if exporter is None:
        return {"exporter": exporter_name, "success": False, "error": f"Unknown exporter '{exporter_name}'"}

    if "find_results" not in exporter.supported_payloads:
        # A saved Auto-Find choice can outlive the exporter's capabilities (a
        # plugin narrowed, or a labelset-only exporter picked before the
        # pickers filtered). Report it rather than handing it a shape it can't
        # read and mailing an empty summary.
        supported = ", ".join(sorted(exporter.supported_payloads)) or "nothing"
        return {
            "exporter": exporter_name,
            "success": False,
            "error": f"Exporter '{exporter_name}' cannot export find results (it supports: {supported})",
        }

    field_values = dict(get_autofind_exporter_field_values().get(exporter_name, {}))
    try:
        from vtscore.plugins.normalize import normalize_field_values  # noqa: PLC0415

        normalize_field_values(exporter, field_values)
        outcome = exporter.export_find_results(response, field_values) or {}
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, never raised
        logger.exception("Auto-Find export via %s failed", exporter_name)
        return {"exporter": exporter_name, "success": False, "error": str(exc)}

    status = {"exporter": exporter_name, "success": True, "message": outcome.get("message", "Export complete.")}
    for key, value in outcome.items():
        if key != "message":
            status[key] = value

    # An `open_url` reaches the browser, so it gets the same scheme allowlist
    # `POST /api/exporters/export` applies - a plugin must not be able to push
    # a `javascript:` URL to the frontend from here either. Unlike that route
    # this one drops the bad URL instead of failing: the export already
    # happened, and the scored results must survive a cosmetic field.
    if status.get("open_url") is not None:
        from vtscore.security.url_validation import validate_browser_url  # noqa: PLC0415

        try:
            status["open_url"] = validate_browser_url(str(status["open_url"]))
        except ValueError as exc:
            logger.error("Auto-Find exporter %r returned an unusable open_url: %s", exporter_name, exc)
            del status["open_url"]
            status["message"] = f"{status['message']} (the exporter returned an unusable URL, so nothing will open)"
    return status


# ---------------------------------------------------------------------------
# Background runs
# ---------------------------------------------------------------------------

_runs_lock = threading.Lock()
_runs: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _hit_count(record: dict[str, Any]) -> int:
    return sum(len(r.get("hits", [])) + len(r.get("negative_hits", [])) for r in record.get("results", {}).values())


def _keep_run(run_id: str, record: dict[str, Any]) -> None:
    """Keep *record* for the results dialog, evicting the oldest runs past the caps."""
    with _runs_lock:
        _runs[run_id] = record
        _runs.move_to_end(run_id)
        total = sum(_hit_count(r) for r in _runs.values())
        while len(_runs) > 1 and (len(_runs) > MAX_KEPT_RUNS or total > MAX_KEPT_HITS):
            _, evicted = _runs.popitem(last=False)
            total -= _hit_count(evicted)


def get_autorun_run(run_id: str, user: str) -> dict[str, Any] | None:
    """The kept results of background run *run_id*, if *user* started it.

    ``None`` for an unknown run, one that has aged out of the
    :data:`MAX_KEPT_RUNS` window, and another user's run alike, so a caller
    cannot tell which it was.
    """
    with _runs_lock:
        record = _runs.get(run_id)
    if record is None or record.get("owner") != user:
        return None
    return record


def clear_autorun_runs() -> None:
    """Forget every kept run.  For test isolation."""
    with _runs_lock:
        _runs.clear()


def _open_run_task(ctx: DatasetContext, trigger: str, media_type: str) -> tuple[str, dict[str, Any], Any]:
    """Register a background run's ``loading-tasks`` row; return ``(run_id, block, tracker)``.

    The row is keyed to the dataset (``dataset_id``) so it renders on the
    dataset's Dashboard row, and carries the ``autorun`` *block* naming the run,
    its owner and *trigger*.
    """
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    run_id = f"{TASK_PREFIX}{uuid4().hex[:8]}"
    dataset_name = ctx.dataset_display_name or ctx.dataset_id
    block: dict[str, Any] = {
        "run_id": run_id,
        "owner": get_current_user(),
        "trigger": trigger,
        "dataset_id": ctx.dataset_id,
        "dataset_name": dataset_name,
    }
    tracker = loading_tasks.create_task(
        run_id,
        f"AutoRun: {dataset_name}",
        dataset_id=ctx.dataset_id,
        media_type=media_type,
        extra_fields={"autorun": block},
    )
    return run_id, block, tracker


def _report_skipped_run(ctx: DatasetContext, trigger: str, reason: str) -> str:
    """Record that an AutoRun had nothing to run, as a row that finishes at once.

    Its ``autorun`` block carries ``skipped`` (the reason), which the owner's
    browser shows as a notice.  Riding the task row rather than a notification
    keeps it to the user whose import it was: notifications reach every client.
    """
    media_type = ctx.medias[next(iter(ctx.medias))].get("media_type", "") if ctx.medias else ""
    run_id, block, tracker = _open_run_task(ctx, trigger, media_type)
    tracker.update("idle", f"AutoRun skipped: {reason}", 0, 0, autorun={**block, "skipped": reason})
    loading_tasks.mark_finished(run_id)
    return run_id


def _dataset_snapshot(ctx: DatasetContext) -> dict:
    """A shallow copy of *ctx*'s medias, taken under the state lock.

    Read off the context object rather than through ``snapshot_medias()``:
    inside a request the active context is the request's own dataset, which is
    not necessarily *ctx*.
    """
    from vtscore.state.core import _state_lock  # noqa: PLC0415

    with _state_lock:
        return dict(ctx.medias)


def start_autorun_task(ctx: DatasetContext, *, trigger: str) -> str:
    """Start a background AutoRun of the current user's detectors on *ctx*.

    The run is a task on the ``loading-tasks`` channel keyed to the dataset
    (``dataset_id``), carrying an ``autorun`` block that names the run, its
    owner and *trigger* (one of :data:`TRIGGERS`) and, once finished, the
    summary counts.  The worker pins *ctx* as its dataset context and never
    touches a loaded detector's live context (the run is not the dataset the
    user is working in); cancelling the task stops it at the next detector.

    Raises :class:`AutoRunUnavailable` when none of the user's AutoRun
    detectors applies to the dataset.  Returns the task id, which is also the
    run id :func:`get_autorun_run` answers to.
    """
    from vtscore.state.core import thread_dataset_context  # noqa: PLC0415
    from vtsearch.threading import spawn  # noqa: PLC0415

    if trigger not in TRIGGERS:
        raise ValueError(f"Unknown AutoRun trigger {trigger!r}; expected one of {TRIGGERS}")

    snap = _dataset_snapshot(ctx)
    plan = plan_autorun(snap)

    run_id, block, tracker = _open_run_task(ctx, trigger, plan.media_type)
    dataset_id = block["dataset_id"]
    n_detectors = len(plan.detectors)
    running_message = f"Running {n_detectors} AutoRun detector{'s' if n_detectors != 1 else ''}…"
    tracker.update("loading", running_message, 0, n_detectors)

    def _on_detector_done(done: int, total: int) -> None:
        tracker.update("loading", f"Scored {done} of {total} AutoRun detectors…", done, total)

    def _on_train_progress(_status: str, message: str = "", *_args: Any, **_kwargs: Any) -> None:
        # A cold train's own phases ("Resolving 40 label origins…") are the
        # slow part of a run; show them without disturbing the detector count.
        snapshot = tracker.get()
        tracker.update("loading", message or running_message, snapshot["current"], snapshot["total"])

    def run() -> None:
        try:
            with thread_dataset_context(ctx):
                response = score_autorun(
                    plan,
                    snap,
                    check_cancelled=tracker.check_cancelled,
                    on_progress=_on_train_progress,
                    on_detector_done=_on_detector_done,
                    use_loaded_context=False,
                )
                auto_export = run_autofind_export(response)
            if auto_export is not None:
                response["auto_export"] = auto_export
            _keep_run(
                run_id,
                {
                    **block,
                    "created_at": time.time(),
                    **response,
                },
            )
            total_hits = sum(r.get("total_hits", 0) for r in response["results"].values())
            summary = {
                **block,
                "detectors_run": response["detectors_run"],
                "total_hits": total_hits,
                "missing_detectors": response["missing_detectors"],
                "auto_export": auto_export,
            }
            tracker.update(
                "idle",
                f"AutoRun found {total_hits} hit{'s' if total_hits != 1 else ''}",
                n_detectors,
                n_detectors,
                autorun=summary,
            )
        except CancelledError:
            tracker.update("idle", "", 0, 0, error="Cancelled")
        except Exception as exc:  # noqa: BLE001 - reported on the task, never raised off a daemon thread
            logger.exception("AutoRun on dataset %s failed", dataset_id)
            tracker.update("idle", "", 0, 0, error=str(exc) or repr(exc) or "AutoRun failed")
        finally:
            loading_tasks.mark_finished(run_id)

    # Registered before the thread starts; see ``LoadingTasksTracker.set_worker``.
    worker = spawn(run, name=f"autorun-{run_id[len(TASK_PREFIX) :]}", start=False)
    loading_tasks.set_worker(run_id, worker)
    worker.start()
    return run_id


# ---------------------------------------------------------------------------
# Import integration
# ---------------------------------------------------------------------------


def _parse_flag(raw: Any) -> bool | None:
    """``True`` / ``False`` for a submitted flag, ``None`` when it was not sent."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if not text:
        return None
    return text in ("1", "true", "yes", "on")


def _autorun_after_import(ctx: DatasetContext) -> None:
    """The load pipeline's ``post_load`` hook: start AutoRun on the new dataset.

    A user with no AutoRun detectors at all has nothing to be told, so that
    starts nothing and says nothing.  One whose detectors merely don't apply to
    this dataset (another media type, or an embedder type it lacks) asked for a
    run and gets none, so the skip is reported to them with its reason
    (:func:`_report_skipped_run`).
    """
    from vtsearch.settings import get_autofind_detectors  # noqa: PLC0415

    try:
        start_autorun_task(ctx, trigger="import")
    except AutoRunUnavailable as exc:
        if get_autofind_detectors():
            _report_skipped_run(ctx, "import", exc.message)


def import_post_load(raw_flag: Any) -> Callable[[DatasetContext], None] | None:
    """Resolve an import request's AutoRun choice into a load ``post_load`` hook.

    *raw_flag* is the request's ``autorun`` value.  When it was sent, it is
    remembered as the user's ``autorun_on_import`` setting - that is what the
    Add Dataset dialog's checkbox starts from next time - and decides this
    import.  When it was not (an API client that doesn't know the flag), the
    remembered setting decides.  Returns ``None`` when AutoRun should not run.
    """
    from vtsearch import settings  # noqa: PLC0415

    choice = _parse_flag(raw_flag)
    if choice is None:
        choice = settings.get_autorun_on_import()
    elif choice != settings.get_autorun_on_import():
        settings.set_autorun_on_import(choice)
    return _autorun_after_import if choice else None
