"""Cross-dataset local features and the labelset's Stage-2 structural re-rank.

A saved structural (SIFT-VLAD) detector sorted against a loaded dataset
re-derives its labelset's local features - from the active snapshot when an
element is loaded, else from its origin file - caches them on
:attr:`~vtscore.state.core.DetectorContext.label_local_features`, and
geometrically re-ranks the Stage-1 shortlist against the templates they give
(:func:`maybe_labelset_structural_rerank`).  A no-op for every non-structural
dataset.
"""

from __future__ import annotations

import logging
from typing import Any

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.labelset_embeddings import _lookups_for_snap
from vtscore.embedding.binding import keying_embedder_for_snap


log = logging.getLogger(__name__)


def _resolve_uncached_local_features(
    elem: LabeledElement,
    snap: dict[int, dict[str, Any]] | None,
    *,
    embedder,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> Any | None:
    """Re-derive *elem*'s :class:`StructuralFeatures`, not consulting the cache.

    Tries the in-dataset path first: when *elem* resolves to a cid in the
    active *snap* that already carries ``local_features``, reuse them (no I/O).
    Otherwise resolve the origin to a file via its importer and run the
    embedder's ``local_features_forward`` to detect features freshly.  The
    **full** (unfiltered) feature set is returned; any ``region_box`` is applied
    downstream at template-build time.  Returns ``None`` when neither path
    yields features.

    *lookups* is the pre-built ``build_media_lookup`` triple for *snap*; loop
    callers pass it so the cid resolution doesn't rebuild the tables per element.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.resolver import resolve_file_context
    from vtscore.media.embedder import media_from_path
    from vtscore.media.structural import StructuralFeatures

    if snap:
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            feats = snap[cid].get("local_features")
            if isinstance(feats, StructuralFeatures) and feats.count > 0:
                return feats

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None
        try:
            feats = embedder.local_features_forward(media_from_path(file_path))
        except Exception:
            log.warning(
                "labelset_structural: local_features_forward(%s) raised; "
                "this label won't contribute a structural template",
                elem.origin_name or elem.filename or "<unknown>",
                exc_info=True,
            )
            return None
    if feats is None or getattr(feats, "count", 0) == 0:
        return None
    return feats


def populate_label_local_features(
    det_ctx,
    labelset: LabelSet,
    *,
    snap: dict[int, dict[str, Any]] | None,
) -> int:
    """Ensure every labelled element has cached local features on *det_ctx*.

    A no-op (returns 0) unless the active dataset's embedder is structural
    (``supports_geometric_verification``) - non-structural detectors never need
    local features.  For a structural detector it re-derives the
    :class:`~vtscore.media.structural.StructuralFeatures` for each good/bad
    element (reusing the active dataset's stored features when the element is
    loaded, resolving the origin file otherwise) and caches them on
    ``det_ctx.label_local_features`` keyed by ``stable_element_id``.  The
    embedder-switch invalidation in
    :func:`~vtscore.detectors.labelset_embeddings._maybe_clear_cache_on_embedder_switch`
    clears this cache alongside the embedding cache.

    Returns the number of elements that have cached features after this pass.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    # Labels are resolved and embedded in the same space the model-invalidation
    # and re-embed checks key on, so the label cache stays valid across an
    # active-dataset switch that keeps binding the same concrete embedder of the
    # detector's type.  See ``docs/ARCHITECTURE.md`` → "Per-detector
    # embedder type".
    embedder_name = keying_embedder_for_snap(det_ctx, snap)
    embedder = None
    if embedder_name:
        from vtscore.media import get_embedder

        try:
            embedder = get_embedder(embedder_name)
        except (KeyError, ValueError):
            embedder = None
    if embedder is None or not getattr(embedder, "supports_geometric_verification", False):
        return 0

    cache: dict[str, Any] = det_ctx.label_local_features
    # Build the origin/md5/name lookup tables once and thread them through,
    # so this pass is O(N + labels) instead of O(labels × N).
    lookups = _lookups_for_snap(snap)
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        if eid in cache:
            continue
        feats = _resolve_uncached_local_features(elem, snap, embedder=embedder, lookups=lookups)
        if feats is not None:
            cache[eid] = feats
    return len(cache)


def _labelset_feature_snapshot(
    det_ctx,
    labelset: LabelSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, None], dict[str, None], dict[str, tuple[float, float, float, float]]]:
    """Project the cached local features into the chokepoint's vote/snap shape.

    Builds a synthetic ``feature_snap`` (``element_id -> {"local_features": ...}``)
    plus ``good_votes`` / ``bad_votes`` / ``region_boxes`` keyed by the same
    ``stable_element_id`` so :func:`maybe_structural_rerank` can build templates
    and train the verification classifier against the cross-dataset labelset
    exactly as it does against in-dataset votes.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    cache: dict[str, Any] = det_ctx.label_local_features
    feature_snap: dict[str, dict[str, Any]] = {}
    good_votes: dict[str, None] = {}
    bad_votes: dict[str, None] = {}
    region_boxes: dict[str, tuple[float, float, float, float]] = {}
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        feats = cache.get(eid)
        if feats is None:
            continue
        feature_snap[eid] = {"local_features": feats}
        if elem.label == "good":
            good_votes[eid] = None
            if elem.region_box is not None:
                region_boxes[eid] = elem.region_box
        else:
            bad_votes[eid] = None
    return feature_snap, good_votes, bad_votes, region_boxes


def maybe_labelset_structural_rerank(
    det_ctx,
    labelset: LabelSet,
    results: list[dict[str, Any]],
    threshold: float,
    snap: dict[int, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], float]:
    """Stage-2 structural re-rank for the saved-detector (labelset) sort path.

    The counterpart to the vote-driven re-rank wired into
    :func:`~vtscore.detectors.training.train_and_score`: when a saved structural
    detector is sorted against a (possibly different) loaded dataset, this
    re-derives the labelset's local features, builds the RegionYes templates
    from them, and geometrically re-ranks the active
    dataset's Stage-1 shortlist.  A no-op for non-structural datasets (gated on
    the active snapshot carrying ``local_features``) and when no labelled element
    yields a usable template.
    """
    from vtscore.training.structural_similarity import maybe_structural_rerank, snapshot_is_structural

    if not snap or not snapshot_is_structural(snap):
        return results, threshold
    populate_label_local_features(det_ctx, labelset, snap=snap)
    feature_snap, good_votes, bad_votes, region_boxes = _labelset_feature_snapshot(det_ctx, labelset)
    if not good_votes:
        return results, threshold
    return maybe_structural_rerank(
        results,
        threshold,
        snap,
        good_votes,
        region_boxes,
        det_ctx,
        feature_snap=feature_snap,
        bad_votes=bad_votes,
    )
