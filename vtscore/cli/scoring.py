"""Score a chunk of medias against the trained heads, and fold the per-chunk results together.

Detectors sharing a target type, embedder and re-clip are scored off one routed
snapshot; a converter or clipper that fans a media out into several items is
folded back to the source media by ``max``.  Media the scorer cannot embed are
skipped and announced, never fatal (#3179).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vtscore.cli import progress as cli_progress
from vtscore.cli.detectors import _detector_group_key
from vtscore.utils.hits import build_media_hit

if TYPE_CHECKING:
    from vtscore.cli.datasets import _RunDataset
    from vtscore.cli.detectors import _RoutedSnapshots
    from vtscore.detectors.training import ScoringRows


def _score_medias_with_detectors(
    medias: dict[int, dict[str, Any]],
    detector_mlps: dict[str, dict[str, Any]],
    routed: "_RoutedSnapshots | None" = None,
) -> dict[str, dict[str, Any]]:
    """Score *medias* against pre-trained detector heads, routing across types.

    *medias* may arrive unembedded and may mix source types.  For each group of
    detectors sharing a target ``media_type`` + embedder + re-clip granularity,
    the medias are routed to that target (native match scored directly, other
    types converted via a one-hop converter such as ``video2image``), optionally
    re-clipped to the detector's ``input_spec.clipper``, and embedded in the
    detector's space by
    :func:`~vtscore.detectors.converter_routing.route_and_embed`.  A converter or
    clipper that fans one source media out into several (a video into frames, a
    recording into tiles) produces several scores, which are aggregated back to
    the source media by ``max`` - a source is a positive hit when *any* of its
    sub-items clears the threshold ("find the needle").  Homogeneous single-type
    datasets with no re-clip take the identity route (one hit per media),
    byte-for-byte the pre-routing behaviour.

    *routed* is the first chunk's memo of those prepared snapshots, shared with
    the calibration pass that fitted these thresholds on them (issue #3647), so
    that chunk is routed once rather than once per pass.  ``None`` - every
    later chunk - prepares its own.
    """
    if not medias or not detector_mlps:
        return {}

    from collections import defaultdict  # noqa: PLC0415

    from vtscore.detectors.converter_routing import route_and_embed  # noqa: PLC0415
    from vtscore.detectors.training import scoring_rows_for_snap  # noqa: PLC0415

    # Detectors sharing a (target type, embedder, re-clip spec) share one
    # routed+clipped+embedded snapshot, so a dataset scored by two image/siglip
    # detectors with the same granularity is prepared once, not per detector.
    groups: dict[tuple[str, str, str, tuple], list[str]] = defaultdict(list)
    for det_name, info in detector_mlps.items():
        key = _detector_group_key(
            info.get("media_type") or "",
            info.get("embedder") or "",
            info.get("clipper") or "",
            info.get("clipper_params") or {},
        )
        groups[key].append(det_name)

    results: dict[str, dict[str, Any]] = {}
    for key, det_names in groups.items():
        target_type, embedder_name, clipper, clipper_params_items = key
        if not target_type:
            # Legacy detector with no declared media_type: score every media
            # directly, one hit per media (media with no usable vector are
            # skipped and reported). The rows are built in the shared score embedder
            # (``embedder_name`` is part of this group's key, so every detector
            # here trained in it), not each media's primary vector - on a trio
            # dataset those differ. Typed detectors take the routing path below.
            results.update(_score_direct_all(det_names, detector_mlps, medias, embedder_name))
            continue
        if routed is not None:
            scoring_medias, scoring_to_source = routed.get(key)
        else:
            scoring_medias, scoring_to_source = route_and_embed(
                medias,
                target_type,
                embedder_name,
                clipper=clipper,
                clipper_params=dict(clipper_params_items),
            )
        if not scoring_medias:
            continue
        # One row build per group; every head in the group forwards the same
        # rows, so anything the builder had to skip is announced once here
        # rather than once per detector.
        rows = scoring_rows_for_snap(scoring_medias, embedder_name or None)
        _emit_skipped_medias(_skipped_ids(scoring_medias, rows.ids), embedder_name)
        if not rows.ids:
            continue
        for det_name in det_names:
            results[det_name] = _score_one_detector(
                det_name,
                detector_mlps[det_name],
                medias,
                rows,
                scoring_to_source,
            )

    if results:
        from vtscore.host.achievements import record_achievement

        record_achievement("find", len(medias) * len(results))

    return results


def _skipped_ids(snapshot: dict[int, dict[str, Any]], scored_ids: list[int]) -> list[int]:
    """The ids in *snapshot* that the row builder could not score.

    :func:`~vtscore.detectors.training.scoring_rows_for_snap` drops a media it
    cannot build a row for and reports the survivors as ``rows.ids``, so the
    difference between the two *is* the skip list - no second pass over the
    snapshot, and no way for the reported skips to disagree with what was
    actually scored.
    """
    if len(scored_ids) == len(snapshot):
        return []
    scored = set(scored_ids)
    return [cid for cid in sorted(snapshot) if cid not in scored]


def _emit_skipped_medias(skipped: list[int], embedder_name: str = "") -> None:
    """Tell the user which media were left out of a scoring pass, and why.

    Skipping is the deliberate policy for a media the scorer cannot embed - a
    corrupt image, an unresolvable thin path, a vector of the wrong width - so
    one bad file doesn't abort a long run (issue #3179).  Silent skipping would
    be worse than the crash it replaces, though: the exported hit count would
    just be quietly short.  So every skip is announced on the CLI's own event
    stream, where ``--progress-format json`` consumers can see it too.
    """
    if not skipped:
        return
    cli_progress.emit(
        "medias_skipped",
        text=(
            f"Skipped {len(skipped)} media with no usable embedding under "
            f"{embedder_name or '(primary)'}: {skipped[:10]}{'…' if len(skipped) > 10 else ''}"
        ),
        skipped=len(skipped),
        skipped_ids=skipped[:100],
        embedder=embedder_name,
    )


def _score_direct_all(
    det_names: list[str],
    detector_mlps: dict[str, dict[str, Any]],
    medias: dict[int, dict[str, Any]],
    embedder_name: str = "",
) -> dict[str, dict[str, Any]]:
    """Score *det_names* directly against every media's *embedder_name* vector.

    The legacy path for detectors that declare no ``media_type``: they score
    whatever embeddings the dataset already holds, one hit per media.  The rows
    are built in *embedder_name* - the concrete space these detectors trained
    in, shared across the group - so a trio dataset whose primary vector differs
    from that space is scored correctly rather than against each media's primary
    vector; empty *embedder_name* falls back to the primary vector, the
    single-embedder behaviour.  Media with no usable vector in that space are
    skipped and reported, not fatal (issue #3179): a folder where one image
    failed to embed should cost that one item, not the whole run.  The
    ``strict=True`` zip guards against an id/score length mismatch (audit M11).

    Scoring goes through :func:`~vtscore.detectors.training.scoring_rows_for_snap`
    + :func:`~vtscore.detectors.training.score_rows_with_model`, i.e. the same
    geometry the GUI's Find scores at and the same one the detector's threshold
    was cut on - on a patch dataset that is the max over the media's patch rows,
    not the image-level vector alone (issue #3180).  The rows are built once for
    the whole group and re-forwarded per head.
    """
    from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap  # noqa: PLC0415

    rows = scoring_rows_for_snap(medias, embedder_name or None)
    all_ids = rows.ids
    _emit_skipped_medias(_skipped_ids(medias, all_ids), embedder_name)
    if not all_ids:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for det_name in det_names:
        info = detector_mlps[det_name]
        mlp = info["mlp"]
        threshold = info["threshold"]
        scores, _best_region = score_rows_with_model(mlp, rows)

        positive_hits: list[dict[str, Any]] = []
        negative_hits: list[dict[str, Any]] = []
        for cid, score in zip(all_ids, scores, strict=True):
            hit = build_media_hit(cid, medias[cid], score)
            if score >= threshold:
                positive_hits.append(hit)
            else:
                negative_hits.append(hit)
        positive_hits.sort(key=lambda x: x["score"], reverse=True)
        negative_hits.sort(key=lambda x: x["score"], reverse=True)

        out[det_name] = {
            "detector_name": det_name,
            "threshold": round(threshold, 4),
            "balance": info.get("balance"),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
            "negative_hits": negative_hits,
        }
    return out


def _score_one_detector(
    det_name: str,
    info: dict[str, Any],
    source_medias: dict[int, dict[str, Any]],
    rows: "ScoringRows",
    scoring_to_source: dict[int, int],
) -> dict[str, Any]:
    """Score one detector over a routed snapshot and fold scores to source media.

    Runs the head over *rows* - the scoring geometry of the routed snapshot,
    built once per detector group by
    :func:`~vtscore.detectors.training.scoring_rows_for_snap` - then reduces
    per-clip scores to one score per source media via ``max`` and builds the hit
    from the *source* media so a video routed through ``video2image`` surfaces
    as a single hit on the video, not one per frame.

    Because the rows come from the shared builder, a patch dataset is scored by
    max-pooling each media's patch rows, exactly as the GUI's Find does and as
    the threshold this compares against was cut (issue #3180).  That builder is
    also what drops a clip it cannot score: ``route_and_embed`` has already
    dropped the ones it could not embed, leaving only the residue it does not
    check for - a vector whose *width* disagrees with the rest (a stale
    pre-computed vector from another model).  Same policy either way: skip the
    item, score the rest (issue #3179).  The caller announces the skips once for
    the group, since the rows - and therefore the skips - are shared by every
    head in it.
    """
    from vtscore.detectors.training import score_rows_with_model  # noqa: PLC0415

    mlp = info["mlp"]
    threshold = info["threshold"]
    scores, _best_region = score_rows_with_model(mlp, rows)

    # Aggregate clip-level scores back to the source media, keeping the best
    # (max) score per source.
    best_by_source: dict[int, float] = {}
    for scoring_id, score in zip(rows.ids, scores, strict=True):
        src_id = scoring_to_source[scoring_id]
        prev = best_by_source.get(src_id)
        if prev is None or score > prev:
            best_by_source[src_id] = float(score)

    positive_hits: list[dict[str, Any]] = []
    negative_hits: list[dict[str, Any]] = []
    for src_id, score in best_by_source.items():
        hit = build_media_hit(src_id, source_medias[src_id], score)
        if score >= threshold:
            positive_hits.append(hit)
        else:
            negative_hits.append(hit)
    positive_hits.sort(key=lambda x: x["score"], reverse=True)
    negative_hits.sort(key=lambda x: x["score"], reverse=True)

    return {
        "detector_name": det_name,
        "threshold": round(threshold, 4),
        "balance": info.get("balance"),
        "total_hits": len(positive_hits),
        "hits": positive_hits,
        "negative_hits": negative_hits,
    }


def _merge_detector_results(
    accumulated: dict[str, dict[str, Any]],
    new_chunk: dict[str, dict[str, Any]],
) -> None:
    """Merge detector results from a new chunk into *accumulated* in-place.

    Hits are appended, not sorted: this path holds every hit in RAM by
    design, so it defers ordering to a single final :func:`_sort_detector_results`
    pass rather than re-sorting the growing list on every chunk.
    """
    for det_name, det_result in new_chunk.items():
        if det_name not in accumulated:
            accumulated[det_name] = det_result
        else:
            accumulated[det_name]["hits"].extend(det_result["hits"])
            accumulated[det_name]["total_hits"] += det_result["total_hits"]
            if "negative_hits" in det_result:
                accumulated[det_name].setdefault("negative_hits", []).extend(det_result["negative_hits"])


def _sort_detector_results(accumulated: dict[str, dict[str, Any]]) -> None:
    """Sort every detector's hit lists by score descending, in place.

    Run once after all chunks have been merged, replacing the per-chunk sort
    that :func:`_merge_detector_results` used to do.
    """
    for det_result in accumulated.values():
        det_result["hits"].sort(key=lambda x: x["score"], reverse=True)
        if "negative_hits" in det_result:
            det_result["negative_hits"].sort(key=lambda x: x["score"], reverse=True)


def _build_multi_results_dict(
    detector_results: dict[str, dict[str, Any]],
    media_type: str = "unknown",
    dataset: "_RunDataset | None" = None,
) -> dict[str, Any]:
    """Build the full results dict from multi-detector scoring.

    *dataset* is the dataset this results set belongs to when the run produced
    several (a multi-dataset import, #4707); it rides along as a ``dataset``
    block so an exported file says which of the run's datasets it holds.  A
    single-dataset run carries no such block, as before.
    """
    results: dict[str, Any] = {
        "media_type": media_type,
        "detectors_run": len(detector_results),
        "results": detector_results,
    }
    if dataset is not None:
        results["dataset"] = dataset.to_dict()
    return results


def _detect_media_type(medias: dict[int, dict[str, Any]]) -> str:
    """Return the media type from the first media, or ``"unknown"``."""
    for media in medias.values():
        return media.get("media_type", "unknown")
    return "unknown"
