"""AutoFind: score a dataset with the current user's AutoFind detectors.

A user's **AutoFind** detectors - the Dashboard's AutoFind tab, stored as the
per-user ``autofind_detectors`` setting - are the ones they have finalized to
run unattended.  This module is where the app runs them against a dataset, for
two kinds of caller:

- ``POST /api/auto-detect`` scores the request's active dataset synchronously,
  reporting on the shared Find tracker (the API / scripted path).
- :func:`start_autofind_task` scores any loaded dataset in the background, as a
  task on the ``loading-tasks`` channel keyed to that dataset, so it renders
  inline on the dataset's Dashboard row and its Cancel button works.  The
  dataset ⋯ menu's **Run AutoFind** starts one directly; a web import starts one
  once the dataset is saved, unless the user unticked **Run AutoFind** in the
  Add Dataset dialog (:func:`import_post_load`).  The Dashboard's big
  **Find** button starts one per ticked dataset, restricted to the ticked
  detectors (``detector_ids``), which may be drafts as well as AutoFind ones;
  such a run is a *Find* (trigger ``find``), whose results the browser opens
  when it lands, where an AutoFind only says it is done (#4615).

The CLI's ``--autodetect`` has its own streaming pipeline in :mod:`vtscore.cli`
and does not come through here.

Either way a run ends the same: the results go to the user's AutoFind exporter
when one is configured (:func:`run_autofind_export`).  A background run's
results are also held in memory for the user who started it
(:func:`get_autofind_run`), so the browser can show them in the Find Results
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

#: Task-id prefix of a background AutoFind on the ``loading-tasks`` channel.
TASK_PREFIX = "_autofind_"

#: How many finished background runs keep their results for the results
#: dialog.  A run's hit lists cover every media in the dataset, so this is a
#: bound on memory, not a history feature.
MAX_KEPT_RUNS = 16

#: And how many hit entries (Good and Bad, across detectors) they may hold
#: between them: a few runs over a large dataset are as heavy as many small
#: ones.  The oldest runs go first; the newest is always kept, whatever its size.
MAX_KEPT_HITS = 1_000_000

#: What started a background run: a finished web import, the dataset ⋯ menu's
#: Run AutoFind (``manual``), or the Dashboard's big Find button with its
#: picked detectors (``find``).  Carried on the task so the browser can name
#: the run in its notice (#4615).
TRIGGERS = ("import", "manual", "find")


def run_label(trigger: str) -> str:
    """What a run of *trigger* is called on its task row: ``Find`` for the
    Find button's, ``AutoFind`` for the rest."""
    return "Find" if trigger == "find" else "AutoFind"


class AutoFindUnavailable(Exception):
    """There is nothing to run.

    *message* is user-facing; *status* is the HTTP code the API answers with
    (400 for "nothing applies to this dataset", 404 for a named detector that
    is not on the user's AutoFind list).
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class AutoFindPlan:
    """The AutoFind detectors that apply to one dataset snapshot.

    ``detectors`` holds ``(name, detector JSON, registry entry)`` triples;
    ``missing`` names AutoFind entries whose detector file no longer exists.
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
    """The AutoFind detector names to consider: all of them, or the one named."""
    from vtsearch.settings import get_autofind_detectors  # noqa: PLC0415

    names = get_autofind_detectors()
    if detector_name:
        if detector_name not in names:
            raise AutoFindUnavailable(f"Detector '{detector_name}' is not on your AutoFind list", status=404)
        return [detector_name]
    if not names:
        raise AutoFindUnavailable("You have no AutoFind detectors. Move a detector to the AutoFind tab first.")
    return names


def _resolve_ids(detector_ids: list[str]) -> list[dict]:
    """The registry entries of *detector_ids*, the detectors the caller picked.

    Any detector the caller can see qualifies, draft or AutoFind.  An id that
    names no detector, or one the caller may not access, is a 404 alike, so a
    caller cannot probe for other users' detectors.
    """
    from vtscore.detectors.registry import can_user_access_detector, get_detector  # noqa: PLC0415
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    if not detector_ids:
        raise AutoFindUnavailable("Select at least one detector to run.")
    user = get_current_user()
    entries: list[dict] = []
    for detector_id in dict.fromkeys(detector_ids):
        entry = get_detector(detector_id)
        if entry is None or not can_user_access_detector(detector_id, user):
            raise AutoFindUnavailable(f"Detector '{detector_id}' not found", status=404)
        entries.append(entry)
    return entries


def _collect_for_media_type(
    names: list[str],
    media_type: str,
    *,
    entries: dict[str, dict] | None = None,
) -> tuple[list[tuple[str, dict, dict | None]], list[str]]:
    """Load detector data + registry entry for each name whose media type is *media_type*.

    Returns ``(detectors, missing)``: *missing* holds names whose detector file
    no longer exists on disk (a stale AutoFind reference).  Names whose media
    type simply doesn't match are skipped without being reported - those are
    legitimately inapplicable, not broken.  *entries* maps a name to the
    registry entry the caller already resolved it from; other names are looked
    up by name.
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
        reg_entry = (entries or {}).get(name) or find_by_name(name)
        if reg_entry is None:
            # Fallback: also accept registry entries whose name matches.
            for entry in list_detectors():
                if entry.get("name") == name:
                    reg_entry = entry
                    break
        detectors.append((name, det_data, reg_entry))
    return detectors, missing


def plan_autofind(snap: dict, *, detector_name: str = "", detector_ids: list[str] | None = None) -> AutoFindPlan:
    """Decide which detectors score *snap*.

    By default those are the current user's AutoFind detectors (or the one of
    them *detector_name* names).  *detector_ids* replaces that list with the
    registry ids the caller picked, drafts included, without touching the
    user's AutoFind list.  Either way, keeps the detectors of *snap*'s media
    type whose locked embedder type the dataset can supply, and raises
    :class:`AutoFindUnavailable` when that leaves nothing to run.
    """
    if detector_name and detector_ids is not None:
        raise ValueError("Pass detector_name or detector_ids, not both")
    if not snap:
        raise AutoFindUnavailable("No medias loaded")
    media_type = next(iter(snap.values())).get("media_type", "audio")
    if detector_ids is not None:
        picked = {entry["name"]: entry for entry in _resolve_ids(detector_ids)}
        detectors, missing = _collect_for_media_type(list(picked), media_type, entries=picked)
        whose = "the selected detectors"
    else:
        names = _resolve_names(detector_name, media_type)
        detectors, missing = _collect_for_media_type(names, media_type)
        whose = "your AutoFind detectors"
    if not detectors:
        message = f"None of {whose} are for {media_type} datasets."
        if missing:
            message += f" Missing detector file(s) for: {', '.join(missing)}"
        raise AutoFindUnavailable(message)
    compatible = [trip for trip in detectors if dataset_supplies_detector_type(trip[1], snap)]
    if not compatible:
        raise AutoFindUnavailable(
            f"None of {whose} can score this dataset: it has no embedder of the kind they were built with."
        )
    return AutoFindPlan(media_type=media_type, detectors=compatible, missing=missing)


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
        balance = _line_state(name, trained[0] if trained else None)

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

        from vtscore.datasets.labelset import LabelSet  # noqa: PLC0415
        from vtscore.detectors.label_quota import served_quota  # noqa: PLC0415

        return name, {
            "detector_name": name,
            "threshold": round(threshold, 4),
            # The balance's state on that cut: unchecked, headless (#4247,
            # #4272, #4413).
            "balance": balance,
            # Which detector the labels gave: under the quota, the Goods'
            # centroid rather than a trained head (#4643).
            "label_quota": served_quota(mlp, LabelSet.from_dict(det_data.get("labelset") or {})),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
            "negative_hits": negative_hits,
        }
    except Exception:
        logger.exception("Auto-detect failed for detector %s", name)
        return None


def _line_state(name: str, det_ctx: Any) -> dict | None:
    """What the balance says about *name*'s cut in an AutoFind: unchecked, because nobody can vote.

    Rides with the cut (``balance``), read at the beta of the thread that
    trained it (#4413).  A headless run cannot spot-check its line (#4272), so
    the cut exported is the balance's unchecked set, and the log line is the
    record that it was never checked.  ``None`` for a detector with no trained
    context to ask, or with no balance (a library caller's choice; the app
    always sets one).
    """
    from vtscore.state.core import detector_balance_state  # noqa: PLC0415
    from vtscore.training.thresholds import BALANCE_UNCHECKED, aim_words  # noqa: PLC0415
    from vtsearch.state import get_beta  # noqa: PLC0415

    if det_ctx is None:
        return None
    balance = detector_balance_state(det_ctx, get_beta())
    if balance is not None and balance["status"] == BALANCE_UNCHECKED:
        logger.info(
            "Auto-detect: detector %s exports its top %d unchecked (%s); nobody is here to check it",
            name,
            balance["count"],
            aim_words(balance),
        )
    return balance


def score_autofind(
    plan: AutoFindPlan,
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
    # ``plan_autofind`` refuses a plan with no detectors.
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
    """Run the configured AutoFind results exporter on *response*.

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
        # A saved AutoFind choice can outlive the exporter's capabilities (a
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
        logger.exception("AutoFind export via %s failed", exporter_name)
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
            logger.error("AutoFind exporter %r returned an unusable open_url: %s", exporter_name, exc)
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


def get_autofind_run(run_id: str, user: str) -> dict[str, Any] | None:
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


def clear_autofind_runs() -> None:
    """Forget every kept run.  For test isolation."""
    with _runs_lock:
        _runs.clear()


def _open_run_task(ctx: DatasetContext, trigger: str, media_type: str) -> tuple[str, dict[str, Any], Any]:
    """Register a background run's ``loading-tasks`` row; return ``(run_id, block, tracker)``.

    The row is keyed to the dataset (``dataset_id``) so it renders on the
    dataset's Dashboard row, and carries the ``autofind`` *block* naming the run,
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
        f"{run_label(trigger)}: {dataset_name}",
        dataset_id=ctx.dataset_id,
        media_type=media_type,
        extra_fields={"autofind": block},
    )
    return run_id, block, tracker


def _report_skipped_run(ctx: DatasetContext, trigger: str, reason: str) -> str:
    """Record that an AutoFind had nothing to run, as a row that finishes at once.

    Its ``autofind`` block carries ``skipped`` (the reason), which the owner's
    browser shows as a notice.  Riding the task row rather than a notification
    keeps it to the user whose import it was: notifications reach every client.
    """
    media_type = ctx.medias[next(iter(ctx.medias))].get("media_type", "") if ctx.medias else ""
    run_id, block, tracker = _open_run_task(ctx, trigger, media_type)
    tracker.update("idle", f"AutoFind skipped: {reason}", 0, 0, autofind={**block, "skipped": reason})
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


def start_autofind_task(ctx: DatasetContext, *, trigger: str, detector_ids: list[str] | None = None) -> str:
    """Start a background AutoFind of the current user's detectors on *ctx*.

    The run is a task on the ``loading-tasks`` channel keyed to the dataset
    (``dataset_id``), carrying an ``autofind`` block that names the run, its
    owner and *trigger* (one of :data:`TRIGGERS`) and, once finished, the
    summary counts.  The worker pins *ctx* as its dataset context and never
    touches a loaded detector's live context (the run is not the dataset the
    user is working in); cancelling the task stops it at the next detector.

    The detectors are the user's AutoFind list, or the registry ids in
    *detector_ids* when given (see :func:`plan_autofind`).  Raises
    :class:`AutoFindUnavailable` when none of them applies to the dataset.
    Returns the task id, which is also the run id :func:`get_autofind_run`
    answers to.
    """
    from vtscore.state.core import thread_dataset_context  # noqa: PLC0415
    from vtsearch.threading import spawn  # noqa: PLC0415

    if trigger not in TRIGGERS:
        raise ValueError(f"Unknown AutoFind trigger {trigger!r}; expected one of {TRIGGERS}")

    snap = _dataset_snapshot(ctx)
    plan = plan_autofind(snap, detector_ids=detector_ids)

    run_id, block, tracker = _open_run_task(ctx, trigger, plan.media_type)
    dataset_id = block["dataset_id"]
    n_detectors = len(plan.detectors)
    # Picked detectors may be drafts, so they are not called AutoFind ones.
    noun = "AutoFind detector" if detector_ids is None else "detector"
    running_message = f"Running {n_detectors} {noun}{'s' if n_detectors != 1 else ''}…"
    tracker.update("loading", running_message, 0, n_detectors)

    def _on_detector_done(done: int, total: int) -> None:
        tracker.update("loading", f"Scored {done} of {total} {noun}s…", done, total)

    def _on_train_progress(_status: str, message: str = "", *_args: Any, **_kwargs: Any) -> None:
        # A cold train's own phases ("Resolving 40 label origins…") are the
        # slow part of a run; show them without disturbing the detector count.
        snapshot = tracker.get()
        tracker.update("loading", message or running_message, snapshot["current"], snapshot["total"])

    def run() -> None:
        try:
            with thread_dataset_context(ctx):
                response = score_autofind(
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
                f"{run_label(trigger)} found {total_hits} hit{'s' if total_hits != 1 else ''}",
                n_detectors,
                n_detectors,
                autofind=summary,
            )
        except CancelledError:
            tracker.update("idle", "", 0, 0, error="Cancelled")
        except Exception as exc:  # noqa: BLE001 - reported on the task, never raised off a daemon thread
            logger.exception("AutoFind on dataset %s failed", dataset_id)
            tracker.update("idle", "", 0, 0, error=str(exc) or repr(exc) or "AutoFind failed")
        finally:
            loading_tasks.mark_finished(run_id)

    # Registered before the thread starts; see ``LoadingTasksTracker.set_worker``.
    worker = spawn(run, name=f"autofind-{run_id[len(TASK_PREFIX) :]}", start=False)
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


def _autofind_after_import(ctx: DatasetContext) -> None:
    """The load pipeline's ``post_load`` hook: start AutoFind on the new dataset.

    A user with no AutoFind detectors at all has nothing to be told, so that
    starts nothing and says nothing.  One whose detectors merely don't apply to
    this dataset (another media type, or an embedder type it lacks) asked for a
    run and gets none, so the skip is reported to them with its reason
    (:func:`_report_skipped_run`).
    """
    from vtsearch.settings import get_autofind_detectors  # noqa: PLC0415

    try:
        start_autofind_task(ctx, trigger="import")
    except AutoFindUnavailable as exc:
        if get_autofind_detectors():
            _report_skipped_run(ctx, "import", exc.message)


def import_post_load(raw_flag: Any) -> Callable[[DatasetContext], None] | None:
    """Resolve an import request's AutoFind choice into a load ``post_load`` hook.

    *raw_flag* is the request's ``autofind`` value.  When it was sent, it is
    remembered as the user's ``autofind_on_import`` setting - that is what the
    Add Dataset dialog's checkbox starts from next time - and decides this
    import.  When it was not (an API client that doesn't know the flag), the
    remembered setting decides.  Returns ``None`` when AutoFind should not run.
    """
    from vtsearch import settings  # noqa: PLC0415

    choice = _parse_flag(raw_flag)
    if choice is None:
        choice = settings.get_autofind_on_import()
    elif choice != settings.get_autofind_on_import():
        settings.set_autofind_on_import(choice)
    return _autofind_after_import if choice else None
