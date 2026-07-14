"""Blueprint for label management routes (export, import, fill-from-sort).

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

import logging

from flask_smorest import Blueprint, abort

from vtsearch.schemas.labels import (
    FillFromSortRequestSchema,
    FillFromSortResponseSchema,
    LabelsExportQuerySchema,
    LabelsExportResponseSchema,
    LabelsImportRequestSchema,
    LabelsImportResponseSchema,
)
from vtscore.detectors.dataset_sync import validated_vote_snapshot
from vtsearch.routes._shared import require_dataset_header, require_detector_header
from vtsearch.state import (
    apply_label,
    apply_label_with_click_time,
    build_media_lookup,
    resolve_media_ids,
    snapshot_medias,
)
from vtscore.utils.hits import build_media_hit

logger = logging.getLogger(__name__)

labels_bp = Blueprint(
    "labels",
    __name__,
    description="Export, import, and bulk-fill label assignments.",
)


def _select_vote_pools(
    label_filter: str,
    goods_only: bool,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    verified_ids: dict[int, None] | None = None,
) -> tuple[dict, dict]:
    """Pick the (goods, bads) dicts to feed into ``LabelSet.from_clips_and_votes``.

    ``label_filter == "corrections"`` returns both pools; the corrections
    filtering step happens after annotation, since "correction" depends on
    the find-initial labels, not on good vs bad.

    ``unverified`` / ``verified`` partition the pools by ``verified_ids`` (the
    Find-mode set of human-touched items): ``unverified`` is the left-panel
    work queue (the detector's calls the human hasn't acted on), ``verified``
    is the right-panel confirmed set.  See docs/plans/find-verification-workflow.md.

    Takes the vote dicts as parameters (rather than reading the module-level
    proxies) so the caller can pass an atomic snapshot from
    :func:`validated_vote_snapshot`, guaranteed to be keyed in the same
    dataset's cid space as the medias being composed with.
    """
    verified = verified_ids or {}
    if label_filter == "good":
        return good_votes, {}
    if label_filter == "bad":
        return {}, bad_votes
    if label_filter == "unverified":
        return (
            {cid: None for cid in good_votes if cid not in verified},
            {cid: None for cid in bad_votes if cid not in verified},
        )
    if label_filter == "verified":
        return (
            {cid: None for cid in good_votes if cid in verified},
            {cid: None for cid in bad_votes if cid in verified},
        )
    if label_filter == "corrections":
        return good_votes, bad_votes
    if label_filter:
        return good_votes, bad_votes
    return good_votes, ({} if goods_only else bad_votes)


def _make_correction_annotator(all_medias: dict):
    """Return a per-entry ``is_correction`` annotator, or ``None`` when no find-initial state exists.

    A label is a correction when the detector's pre-vote label
    (``find_initial_labels``) differs from the current label. When there is
    no find-initial state (no detector was run, or the vote came from
    outside Find), we return ``None`` so callers skip annotation entirely.

    The ``md5 -> media_id`` map is built once here so the returned closure is
    O(1) per entry, letting both the buffered and the streaming export paths
    annotate one element at a time without re-scanning ``all_medias``.
    """
    from vtsearch.state import get_find_initial_labels

    find_initial = get_find_initial_labels()
    if not find_initial:
        return None

    md5_to_id: dict[str, int] = {}
    for mid, m in all_medias.items():
        md5_val = m.get("md5")
        if md5_val and md5_val not in md5_to_id:
            md5_to_id[md5_val] = mid

    def annotate(entry: dict) -> None:
        media_id = md5_to_id.get(entry.get("md5"))
        if media_id is not None and media_id in find_initial:
            entry["is_correction"] = entry.get("label") != find_initial[media_id]
        else:
            entry["is_correction"] = False

    return annotate


def _annotate_corrections(result: dict, all_medias: dict) -> None:
    """Add ``is_correction`` to every label entry in *result* (mutates in place)."""
    annotate = _make_correction_annotator(all_medias)
    if annotate is None:
        return
    for entry in result["labels"]:
        annotate(entry)


def _build_entry_metadata(media: dict) -> dict:
    """Return the metadata blob for one labelled media (display, origin, and custom)."""
    from vtscore.media import get as get_media_type  # noqa: PLC0415

    try:
        meta = get_media_type(media.get("media_type", "audio")).display_metadata(media)
    except KeyError:
        meta = {}

    origin = media.get("origin")
    if isinstance(origin, dict):
        for k, v in origin.get("params", {}).items():
            meta.setdefault(k, v)

    importer_custom = media.get("custom_metadata")
    if importer_custom:
        meta.update(importer_custom)
    # Humanize "File Size" for export so rows read "8.0 KB" instead of raw
    # bytes, matching what the focus-view UI shows. ``display_metadata`` keeps
    # the raw int (the media-list API returns it and the UI formats it
    # client-side); only this export copy is stringified. Mirrors the frontend
    # formula in center-panel.component.ts (`formatMetadataValue`).
    fs = meta.get("File Size")
    if isinstance(fs, (int, float)) and not isinstance(fs, bool):
        meta["File Size"] = f"{fs / 1024:.1f} KB"
    return meta


_BASE_EXPORT_COLUMNS = ["label", "md5", "origin_name", "filename", "category", "origin"]


def _make_enricher(all_medias: dict):
    """Return a per-entry ``custom_metadata`` enricher.

    The returned callable mutates one label entry in place (attaching
    ``custom_metadata`` when the media has any) and returns the set of
    metadata keys it added, so a streaming caller can annotate elements one
    at a time.  The ``md5 -> media`` map is built once here.

    Flattens origin params so fields like ``contentID`` / ``mediaID`` /
    ``media_url`` surface as selectable export columns alongside the
    importer's own ``custom_metadata``.
    """
    md5_to_media = {m["md5"]: m for m in all_medias.values() if m.get("md5")}

    def enrich(entry: dict) -> set[str]:
        media = md5_to_media.get(entry.get("md5"))
        if not media:
            return set()
        meta = _build_entry_metadata(media)
        if not meta:
            return set()
        entry["custom_metadata"] = meta
        return set(meta.keys())

    return enrich


def _enrich_with_metadata(result: dict, all_medias: dict) -> None:
    """Attach ``custom_metadata`` per entry and the ``available_columns`` list."""
    enrich = _make_enricher(all_medias)
    all_meta_keys: set[str] = set()
    for entry in result["labels"]:
        all_meta_keys.update(enrich(entry))

    base_lower = {c.lower() for c in _BASE_EXPORT_COLUMNS}
    extra_keys = sorted(k for k in all_meta_keys if k.lower() not in base_lower)
    result["available_columns"] = _BASE_EXPORT_COLUMNS + extra_keys


@labels_bp.route("/api/labels/export")
@labels_bp.arguments(LabelsExportQuerySchema, location="query")
@labels_bp.response(200, LabelsExportResponseSchema)
def export_labels(query: dict):
    """Export labels as a :class:`~vtscore.datasets.labelset.LabelSet`.

    Each label entry includes the element's ``origin`` and ``origin_name``
    so consumers know exactly where each labeled element came from. The
    format is a superset of the legacy export format; old consumers
    that only read ``md5`` and ``label`` keys continue to work unchanged.
    """
    from vtscore.datasets.labelset import LabelSet

    label_filter = query["label_filter"]
    # Atomic (medias, good_votes, bad_votes, vote_region_boxes) snapshot so
    # the votes we compose with ``all_medias`` are guaranteed to be keyed in
    # the same dataset's cid space; even if a concurrent request rehydrates
    # the detector against a different dataset before this route finishes.
    snap = validated_vote_snapshot()
    goods, bads = _select_vote_pools(
        label_filter, query["goods_only"], snap.good_votes, snap.bad_votes, snap.verified_ids
    )

    all_medias = snap.medias
    labelset = LabelSet.from_clips_and_votes(
        all_medias,
        goods,
        bads,
        vote_region_boxes=snap.vote_region_boxes,
    )

    if query["format"] == "ndjson":
        return _stream_labels_ndjson(labelset, all_medias, label_filter, query["enrich"])

    result: dict = labelset.to_dict()

    _annotate_corrections(result, all_medias)
    if label_filter == "corrections":
        result["labels"] = [e for e in result["labels"] if e.get("is_correction")]

    if query["enrich"]:
        _enrich_with_metadata(result, all_medias)

    return result


def _stream_labels_ndjson(labelset, all_medias: dict, label_filter: str, enrich: bool):
    """Stream the labelset as newline-delimited JSON, one label entry per line (S13).

    Encodes one :class:`~vtscore.datasets.labelset.LabeledElement` at a time
    via :meth:`LabelSet.iter_dicts`, wrapped in ``flask.stream_with_context``
    so the buffered ``[e.to_dict() for e in elements]`` list (~50 MB at 100 k
    labels) is never held in memory.  Each line carries the same
    ``is_correction`` / ``custom_metadata`` annotations the buffered response
    would attach, applied in the same order (annotate corrections, drop
    non-corrections under ``label_filter=corrections``, then enrich).

    The top-level ``available_columns`` list has no place in NDJSON: it's a
    whole-set aggregate that can't be emitted before the last row is seen, and
    consumers of a streamed export derive columns from the rows themselves.
    """
    import json

    from flask import Response, stream_with_context

    annotate = _make_correction_annotator(all_medias)
    enricher = _make_enricher(all_medias) if enrich else None
    corrections_only = label_filter == "corrections"

    def generate():
        for entry in labelset.iter_dicts():
            if annotate is not None:
                annotate(entry)
            if corrections_only and not entry.get("is_correction"):
                continue
            if enricher is not None:
                enricher(entry)
            yield json.dumps(entry) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
    )


@labels_bp.route("/api/labels/import", methods=["POST"])
@labels_bp.arguments(LabelsImportRequestSchema)
@labels_bp.response(200, LabelsImportResponseSchema)
@require_dataset_header
@require_detector_header
def import_labels(body: dict):
    """Import labels from JSON, matching medias by origin+origin_name (MD5 fallback)."""
    labels = body["labels"]

    origin_lookup, md5_lookup, _ = build_media_lookup(snapshot_medias())

    applied = 0
    skipped = 0
    for entry in labels:
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        if not cids:
            skipped += 1
            continue

        # Round-trip region_box on good-vote imports.  ``LabeledElement.from_dict``
        # already coerces list↔tuple, so we just check shape and pass through.
        rb_raw = entry.get("region_box") if label == "good" else None
        region_box: tuple[float, float, float, float] | None = None
        if isinstance(rb_raw, (list, tuple)) and len(rb_raw) == 4 and all(isinstance(v, (int, float)) for v in rb_raw):
            region_box = (float(rb_raw[0]), float(rb_raw[1]), float(rb_raw[2]), float(rb_raw[3]))

        for cid in cids:
            # Importing a labelset is a bulk action, not consecutive individual
            # hand-clicks: credit the other vote achievements but not the
            # Marathoner streak.
            apply_label(cid, label, region_box=region_box, count_streak=False)
        applied += 1

    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

    sync_labels_to_loaded_detector()

    from vtscore.labels.sync import sync_to_labelset_source

    sync_to_labelset_source()

    return {"applied": applied, "skipped": skipped}


def _partition_candidates(
    sort_results: list[dict],
    thresh: float,
    sides: str,
    snap_good: dict[int, None],
    snap_bad: dict[int, None],
    snap_medias: dict,
) -> tuple[list[dict], list[dict]]:
    """Split sort results into (good, bad) candidate lists by threshold.

    Skips entries that are missing an id/score, carry a non-numeric score,
    are already voted (in *snap_good* / *snap_bad*), or are absent from
    *snap_medias*.  An entry scoring at or above *thresh* becomes a good
    candidate; below it, a bad candidate.  The *sides* selector then clears
    whichever list isn't wanted (``"good"`` drops bads, ``"bad"`` drops
    goods, ``"both"`` keeps both).
    """
    good_candidates = []
    bad_candidates = []
    for entry in sort_results:
        cid = entry.get("id")
        score = entry.get("score", entry.get("similarity"))
        if cid is None or score is None:
            continue
        if not isinstance(score, (int, float)):
            continue
        if cid in snap_good or cid in snap_bad:
            continue
        if cid not in snap_medias:
            continue
        if score >= thresh:
            good_candidates.append({"id": cid, "score": float(score)})
        else:
            bad_candidates.append({"id": cid, "score": float(score)})

    if sides == "good":
        bad_candidates = []
    elif sides == "bad":
        good_candidates = []
    # "both" keeps both lists
    return good_candidates, bad_candidates


@labels_bp.route("/api/labels/fill-from-sort", methods=["POST"])
@labels_bp.arguments(FillFromSortRequestSchema)
@labels_bp.response(200, FillFromSortResponseSchema)
@require_dataset_header
@require_detector_header
def fill_labels_from_sort(body: dict):
    """Fill labels from the current sort results.

    Assigns Good/Bad labels to currently-unlabeled medias based on their
    position relative to the sort threshold. With ``confirm=false`` (the
    default), returns counts only; with ``confirm=true``, applies the
    labels and returns the resulting data as a results dict suitable for
    any exporter.
    """
    sort_results = body["sort_results"]
    thresh = body["threshold"]
    sides = body["sides"]
    confirm = body["confirm"]

    # Atomic snapshot so the membership checks below use the same dataset's
    # cid space as the medias dict; a concurrent rehydrate on the detector
    # against a different dataset can't make us think an A-cid is "already
    # voted" when in fact we're scoring against B's medias.
    vote_snap = validated_vote_snapshot()
    snap_good = vote_snap.good_votes
    snap_bad = vote_snap.bad_votes
    snap_medias = vote_snap.medias

    # Find unlabeled medias above/below threshold
    good_candidates, bad_candidates = _partition_candidates(
        sort_results, thresh, sides, snap_good, snap_bad, snap_medias
    )

    if not confirm:
        return {
            "good_count": len(good_candidates),
            "bad_count": len(bad_candidates),
        }

    # Snapshot the vote state we're about to mutate so a failed persistence
    # pass can roll it back (mirroring ``apply_and_retrain``, audit H30):
    # without the rollback the 500 below tells the user the labels were NOT
    # committed while they stay live in memory and get silently persisted by
    # the next vote-triggered sync.
    from vtscore.state.core import _state_lock, get_active_detector_context

    det_ctx = get_active_detector_context()
    saved_good_votes = dict(det_ctx.good_votes)
    saved_bad_votes = dict(det_ctx.bad_votes)
    saved_region_boxes = dict(det_ctx.vote_region_boxes)
    saved_history = list(det_ctx.label_history)
    saved_click_times = dict(det_ctx.vote_click_times)
    saved_click_counter = det_ctx.click_counter

    # Apply labels
    for entry in good_candidates:
        apply_label_with_click_time(entry["id"], "good")

    for entry in bad_candidates:
        apply_label_with_click_time(entry["id"], "bad")

    # Persist labels to disk BEFORE building the response.  Letting a
    # silent disk-write failure here fall through to ``return {...}`` is
    # the C11 bug; the UI would treat the labels as committed while
    # ``detectors/<name>.json`` never received them.  ``sync_to_labelset_source``
    # is fire-and-forget by design (debounced background timer), so we
    # only guard against the unlikely synchronous scheduling failure.
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector
    from vtscore.labels.sync import sync_to_labelset_source

    try:
        sync_labels_to_loaded_detector()
    except Exception as exc:
        with _state_lock:
            det_ctx.good_votes.clear()
            det_ctx.good_votes.update(saved_good_votes)
            det_ctx.bad_votes.clear()
            det_ctx.bad_votes.update(saved_bad_votes)
            det_ctx.vote_region_boxes.clear()
            det_ctx.vote_region_boxes.update(saved_region_boxes)
            det_ctx.label_history.clear()
            det_ctx.label_history.extend(saved_history)
            det_ctx.vote_click_times.clear()
            det_ctx.vote_click_times.update(saved_click_times)
            det_ctx.click_counter = saved_click_counter
        logger.exception("fill_labels_from_sort: detector label sync failed")
        abort(500, message=f"Failed to persist labels to detector store: {exc}")

    try:
        sync_to_labelset_source()
    except Exception:
        logger.exception("fill_labels_from_sort: labelset source scheduling failed")

    # Build a results dict compatible with exporters.  Reuse the snapshot
    # taken at the top so the hit dicts reference the same dataset's media
    # entries we used for membership checks.
    good_hits = [
        build_media_hit(e["id"], snap_medias.get(e["id"], {}), e["score"], label="good") for e in good_candidates
    ]
    bad_hits = [build_media_hit(e["id"], snap_medias.get(e["id"], {}), e["score"], label="bad") for e in bad_candidates]

    media_type = "unknown"
    for media in snap_medias.values():
        media_type = media.get("media_type", "unknown")
        break

    results_dict = {
        "media_type": media_type,
        "detectors_run": 1,
        "results": {
            "fill_from_sort": {
                "detector_name": "fill_from_sort",
                "threshold": round(thresh, 4),
                "total_hits": len(good_hits),
                "hits": good_hits,
                "negative_hits": bad_hits,
            },
        },
    }

    return {
        "good_applied": len(good_candidates),
        "bad_applied": len(bad_candidates),
        "results": results_dict,
    }
