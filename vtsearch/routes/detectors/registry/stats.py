"""Detector-registry routes that report on a detector's labelset.

Endpoints
---------
GET    /api/detectors/registry/<id>/stats                     The Stats window: labelset composition and provenance.
POST   /api/detectors/registry/<id>/browse-positives          Build a browse map of the detector's positives.
POST   /api/detectors/registry/<id>/browse-positives/release  Free that browse map.
"""

from __future__ import annotations

from flask_smorest import abort

from vtsearch.auth import get_current_user
from vtscore.detectors.embedder_type import detector_embedder_type_from_data
from vtscore.detectors.store import (
    _detector_path,
    _read_detector,
)
from vtsearch.routes.detectors.registry._common import detectors_registry_bp
from vtsearch.schemas.detectors import (
    DetectorBrowsePositivesReleaseResponseSchema,
    DetectorBrowsePositivesResponseSchema,
    DetectorRegistryStatsResponseSchema,
)


# ---------------------------------------------------------------------------
# GET /api/detectors/registry/<detector_id>/stats
# ---------------------------------------------------------------------------


def _resolved_positive_count(elements) -> int:
    """How many of *elements* (positives) resolve into the active dataset.

    Builds the media lookup once and resolves every element against it, so
    this is one pass rather than one rebuild per element. Returns 0 when no
    dataset is loaded.
    """
    from vtsearch.state import cached_media_lookups, resolve_media_ids, snapshot_medias

    snap = snapshot_medias()
    if not snap:
        return 0
    origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
    count = 0
    for el in elements:
        if resolve_media_ids(el.to_dict(), origin_lookup, md5_lookup, name_lookup):
            count += 1
    return count


def _active_dataset_name() -> str:
    """Display name of the active dataset, or ``""`` when none is loaded."""
    from vtsearch.state import get_active_context

    ds_id = get_active_context().dataset_id
    if not ds_id:
        return ""
    from vtscore.datasets.registry import get_dataset

    entry = get_dataset(ds_id)
    return (entry or {}).get("name", "") if entry else ""


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/stats")
@detectors_registry_bp.response(200, DetectorRegistryStatsResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def get_detector_stats(detector_id: str):
    """Return labelset composition and provenance for a registered detector.

    Counts/metadata only — no embeddings or head weights are read or
    returned (the labelset file is the canonical persisted form). The
    ``num_positive_resolved`` / ``active_dataset_name`` pair reports how
    much of the positive set the Browse button could currently project;
    ``test_verdicts`` is every test verdict the detector keeps, one per tested
    dataset, newest first and marked stale once the detector has been
    retrained since (#4526).  ``imported`` is how a label importer created the
    detector (#5334), or ``None``.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.line_verdicts import verdict_summaries
    from vtscore.detectors.registry import (
        can_user_access_detector,
        get_detector,
        get_loaded_detector_ids,
    )
    from vtscore.media import get_clipper
    from vtscore.state.core import get_detector_context
    from vtsearch.settings import get_autofind_detectors

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not can_user_access_detector(detector_id, get_current_user()):
        abort(403, message="You do not have access to this detector")

    name = entry.get("name", "") or ""
    data = _read_detector(_detector_path(name)) or {}

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    positives = [el for el in labelset.elements if el.label == "good"]
    negatives = [el for el in labelset.elements if el.label == "bad"]

    # Embedder: the loaded context's live value wins (it reflects the concrete
    # space the labels are currently resolved against); otherwise the registry's
    # recorded embedder (the concrete space it last trained in).  The detector
    # itself now persists only an embedder *type*, not a concrete name.
    if detector_id in get_loaded_detector_ids():
        ctx = get_detector_context(detector_id)
        embedder = (ctx.embedder if ctx is not None else "") or entry.get("embedder", "") or ""
    else:
        embedder = entry.get("embedder", "") or ""

    input_spec = data.get("input_spec") or {}
    raw_clipper = (input_spec.get("clipper", "") if isinstance(input_spec, dict) else "") or ""
    if not raw_clipper or raw_clipper.endswith("_default"):
        clipper_display = ""
    else:
        try:
            clipper_display = get_clipper(raw_clipper).display_name
        except KeyError:
            clipper_display = raw_clipper

    return {
        "name": name,
        "media_type": entry.get("media_type", "") or "",
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_total": len(labelset.elements),
        "num_positive_resolved": _resolved_positive_count(positives),
        "active_dataset_name": _active_dataset_name(),
        "embedder": embedder,
        "text_query": data.get("text_query", "") or "",
        "media_example": data.get("media_example", "") or "",
        "clipper": clipper_display,
        "embedder_type": detector_embedder_type_from_data(data),
        "created_at": entry.get("created_at"),
        "last_trained_at": entry.get("last_trained_at"),
        "created_by": entry.get("created_by", "default") or "default",
        "readers": entry.get("readers", []) or [],
        "autofind": name in set(get_autofind_detectors()),
        "test_verdicts": verdict_summaries(data),
        "imported": entry.get("imported") or None,
    }


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/<detector_id>/browse-positives
# POST /api/detectors/registry/<detector_id>/browse-positives/release
# ---------------------------------------------------------------------------


def _detector_browse_embedder(entry: dict, media_type: str) -> str:
    """The embedder to project a detector's positives with.

    Browsing a detector must not depend on whatever dataset is incidentally
    selected on the dashboard, so the registry's recorded embedder — the
    concrete space the detector last trained in — wins.  Since a detector now
    persists only an embedder *type* (not a concrete name), there is nothing
    finer to prefer; falls back to the media type's default embedder when the
    detector has never trained (no recorded embedder).
    """
    recorded = entry.get("embedder", "") or ""
    if recorded:
        return recorded
    from vtscore.media import embedders_for_type

    avail = embedders_for_type(media_type)
    return avail[0].name if avail else ""


def _run_positives_browse_build(
    tracker,
    task_id: str,
    detector_data: dict,
    dataset_id: str,
    *,
    embedder_name: str,
    cached_embeddings: dict | None,
    display_name: str,
) -> None:
    """Background worker: build + register the ephemeral browse context.

    Mirrors the error handling of the detector-load task: a cancel or an
    "empty after resolution" :class:`ValueError` surfaces as a task error on
    the detector row; anything else logs a traceback and surfaces its message.
    """
    from vtscore.concurrency.progress import CancelledError, detector_loading_tasks
    from vtscore.detectors.positives_browse import build_positives_browse_context
    from vtscore.state.core import register_context

    def _on_progress(current: int, total: int, message: str) -> None:
        tracker.check_cancelled()
        tracker.update("loading", message, current, total, step=1, total_steps=1)

    try:
        ctx = build_positives_browse_context(
            detector_data,
            dataset_id,
            embedder_name=embedder_name,
            cached_embeddings=cached_embeddings,
            display_name=display_name,
            on_progress=_on_progress,
        )
        register_context(ctx)
        tracker.update("idle", "", 0, 0, step=None, total_steps=None)
    except CancelledError:
        tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
    except ValueError as e:
        tracker.update("idle", "", 0, 0, error=str(e), step=None, total_steps=None)
    except Exception as e:
        import traceback as _tb

        _tb.print_exc()
        tracker.update(
            "idle", "", 0, 0, error=str(e) or repr(e) or "Unknown error preparing browse", step=None, total_steps=None
        )
    finally:
        detector_loading_tasks.mark_finished(task_id)


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/browse-positives", methods=["POST"])
@detectors_registry_bp.response(200, DetectorBrowsePositivesResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
@detectors_registry_bp.alt_response(409, description="The detector has no positive labels to browse.")
def browse_detector_positives(detector_id: str):
    """Prepare an in-memory VTSBrowse map of just this detector's positives.

    Resolves every positive label's origin to its file and embeds it with the
    **detector's** embedder (reusing the loaded detector's cached vectors when
    it's already in that space), assembles a throwaway ``DatasetContext`` whose
    media carry those embeddings + preview bytes, projects it, and registers it
    under a synthetic ``dataset_id`` the browse view can open. Nothing is
    persisted — the context (vectors and bytes) lives only in memory until
    released. Mixed-source detectors work: a positive needn't be in any loaded
    dataset, only origin-resolvable.

    Returns immediately with the ``dataset_id`` and the ``task_id`` whose
    progress the detector's dashboard row renders while the build runs.
    """
    from vtscore.concurrency.progress import detector_loading_tasks
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.positives_browse import detpos_dataset_id
    from vtscore.detectors.registry import (
        can_user_access_detector,
        get_detector,
        get_loaded_detector_ids,
    )
    from vtscore.state.core import get_detector_context, unregister_context
    from vtsearch.threading import spawn

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not can_user_access_detector(detector_id, get_current_user()):
        abort(403, message="You do not have access to this detector")

    name = entry.get("name", "") or ""
    data = _read_detector(_detector_path(name)) or {}
    media_type = entry.get("media_type", "") or data.get("media_type", "") or ""

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    if not any(el.label == "good" for el in labelset.elements):
        abort(409, message="This detector has no positive labels to browse.")

    embedder_name = _detector_browse_embedder(entry, media_type)

    # Reuse the loaded detector's cached label vectors only when they were built
    # in the same space we're about to project in; otherwise re-embed fresh.
    cached_embeddings = None
    if detector_id in get_loaded_detector_ids():
        det_ctx = get_detector_context(detector_id)
        if det_ctx is not None and det_ctx.embedder and det_ctx.embedder == embedder_name:
            cached_embeddings = dict(det_ctx.label_embeddings)

    dataset_id = detpos_dataset_id(detector_id)
    # Drop any stale context from a previous browse of this detector.
    unregister_context(dataset_id)

    task_id = f"_detbrowse_{detector_id[:8]}"
    tracker = detector_loading_tasks.create_task(
        task_id,
        name or detector_id,
        detector_id=detector_id,
        media_type=media_type,
        embedder=embedder_name,
    )
    tracker.update("loading", "Preparing browse…", 0, 0, step=1, total_steps=1)

    display_name = f"{name} — positives" if name else "Detector positives"
    spawn(
        lambda: _run_positives_browse_build(
            tracker,
            task_id,
            data,
            dataset_id,
            embedder_name=embedder_name,
            cached_embeddings=cached_embeddings,
            display_name=display_name,
        ),
        name=f"det-browse-{detector_id[:8]}",
    )
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "task_id": str(task_id),
        "media_type": media_type,
    }


@detectors_registry_bp.route(
    "/api/detectors/registry/<detector_id>/browse-positives/release",
    methods=["POST"],
)
@detectors_registry_bp.response(200, DetectorBrowsePositivesReleaseResponseSchema)
def release_detector_positives_browse(detector_id: str):
    """Free the ephemeral positives-browse context for *detector_id*.

    Called when the user leaves the browse view. Idempotent: ``released`` is
    ``False`` when there was nothing to drop.
    """
    from vtscore.detectors.positives_browse import detpos_dataset_id
    from vtscore.state.core import unregister_context

    dropped = unregister_context(detpos_dataset_id(detector_id))
    return {"ok": True, "released": dropped is not None}
