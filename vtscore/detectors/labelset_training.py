"""Train and score using a detector's full labelset, not just current votes.

The detector's saved labelset on disk is origin-keyed and dataset-agnostic.
At load time we resolve every element to a file via its origin importer,
embed it, and cache the resulting vector on
:attr:`~vtscore.state.core.DetectorContext.label_embeddings`.  Head
training (load-time and during interactive learned-sort) then iterates the
labelset directly, so labels from datasets that aren't currently loaded
still contribute.

This module holds the two training entry points, :func:`train_from_labelset`
and :func:`labelset_train_and_score`.  Each reads four helper modules, none of
which reads back:

* :mod:`~vtscore.detectors.labelset_embeddings` - resolve the labelset into
  the label-embedding caches (:func:`populate_label_embeddings`).
* :mod:`~vtscore.detectors.labelset_matrix` - assemble ``(X, y)`` from those
  caches (:func:`build_xy_from_labelset`) and name the labelled media.
* :mod:`~vtscore.detectors.labelset_structural` - the cross-dataset local
  features and the Stage-2 structural re-rank.
* :mod:`~vtscore.detectors.labelset_heads` - the heads the label quota gives:
  the Goods' centroid and a typed query's.

Their public names are re-exported here, so ``labelset_training.X`` resolves
as it did when this was one module.

**Patch targets** follow the caller: a stub has to be installed where the
name is looked up.  The entry points here call the helpers by name, so
patching ``populate_label_embeddings``, ``build_xy_from_labelset``,
``labeled_media_labels``, ``maybe_labelset_structural_rerank`` and the rest on
this module reaches them.  A helper called from inside its own module resolves
that module's global instead: stubbing ``_embed_one`` for
``populate_label_embeddings`` means patching
:mod:`~vtscore.detectors.labelset_embeddings`, and stubbing
``populate_label_local_features`` for ``maybe_labelset_structural_rerank``
means patching :mod:`~vtscore.detectors.labelset_structural` - not this
module, where the rebind is silently ignored.
"""

from __future__ import annotations

from typing import Any, Callable

from vtscore.datasets.labelset import LabelSet
from vtscore.detectors.labelset_embeddings import ProgressCallback, populate_label_embeddings
from vtscore.detectors.labelset_heads import (
    Haystack,
    _centroid_beta,
    _query_tier,
    face_ball_tier,
    _unvoted_mask,
    install_centroid_head,
    install_query_head,
)
from vtscore.detectors.labelset_matrix import (
    build_xy_from_labelset,
    hard_view_good_vectors,
    labeled_media_ids,
    labeled_media_labels,
    labelset_calibrating_groups,
    labelset_resolution_report,
)
from vtscore.detectors.labelset_structural import maybe_labelset_structural_rerank, populate_label_local_features

__all__ = [
    "Haystack",
    "ProgressCallback",
    "build_xy_from_labelset",
    "hard_view_good_vectors",
    "install_centroid_head",
    "install_query_head",
    "labeled_media_ids",
    "labeled_media_labels",
    "labelset_calibrating_groups",
    "labelset_resolution_report",
    "labelset_train_and_score",
    "maybe_labelset_structural_rerank",
    "populate_label_embeddings",
    "populate_label_local_features",
    "train_from_labelset",
]


def _n_goods(y_list: list[float]) -> int:
    """The Goods among the resolved labels (#5478's ball needs two)."""
    return sum(1 for y in y_list if y == 1.0)


def train_from_labelset(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    snap: dict[int, dict[str, Any]] | None,
    haystack_for: "Callable[[str], Haystack | None] | None" = None,
    on_progress: ProgressCallback | None = None,
    text_query: str = "",
) -> bool:
    """Populate the embedding cache, build (X, y), train, and store on *det_ctx*.

    Returns ``True`` when a detector was stored on *det_ctx*; otherwise leaves
    ``det_ctx.model`` untouched.  Which detector follows the label quota
    (#4643, :mod:`~vtscore.detectors.label_quota`) on the labels that resolved
    to a vector: none with no Good (``False``), the Goods' centroid
    (:func:`install_centroid_head`) under either quota, the trained head once
    both are met.  Either is stamped with *labelset*'s signature, so Find
    reuses it only until the labels change (issue #4204).

    *snap* does two jobs, and a caller that scores something other than what it
    loaded needs them separated.  It is the snapshot the labelset's elements
    are resolved against, and it is the haystack the threshold is calibrated
    on.  For the GUI those are the same set - it scores the dataset it loaded -
    but the CLI converts, re-clips and re-embeds before scoring, so calibrating
    on *snap* would realize the cut's quantile on a population inference never
    sees (issue #3647).  Such a caller passes *haystack_for*, which is invoked
    **after** the labels are embedded - so it is handed ``det_ctx.embedder``,
    the space they turned out to land in, which is the very thing a routing
    decision needs and which does not exist before this call.  Returning
    ``None`` from it keeps *snap* as the haystack.

    *text_query* is the typed query the detector was made from
    (:func:`~vtscore.detectors.query_head.detector_text_query`).  Until the
    quota gives a trained head, a detector with one gets that query's head
    (:func:`install_query_head`, #4758) in place of nothing or the centroid,
    its line drawn over the haystack, and on a single-vector space it keeps it
    past the quota until the labels number
    :data:`~vtscore.detectors.label_quota.QUERY_HOLD_LABELS` (#4736).  Where the
    haystack has no text slot it falls back to those.  The trained head it then
    fits on that space starts from the query as well as from the Goods (#4859).
    """
    populate_label_embeddings(
        det_ctx,
        labelset,
        media_type=media_type,
        snap=snap,
        on_progress=on_progress,
    )
    X_list, y_list, groups, score_rows = build_xy_from_labelset(det_ctx, labelset)
    from vtscore.detectors.label_quota import TIER_CENTROID, TIER_TRAINED, quota_from_groups

    quota_tier = quota_from_groups(groups).tier
    # The face crops' ball holds past the quota from beta 1 up (#5478); a typed query reads the quota's own tier.
    tier = face_ball_tier(quota_tier, media_type or det_ctx.media_type, _centroid_beta(det_ctx), _n_goods(y_list))
    # populate_label_embeddings stamped det_ctx.embedder with the space the
    # labels were embedded in; score the threshold's pass in that same space.
    haystack = haystack_for(det_ctx.embedder or "") if haystack_for is not None else None
    hay = haystack.medias if haystack is not None else snap
    # The labelled media, which every head's line is counted without (#3308,
    # #4779, #5048); a haystack names them in its own ids.
    labels = labeled_media_labels(labelset, snap)
    if haystack is not None:
        labels = {hid: labels[src] for hid, src in haystack.to_source.items() if src in labels}
    labelled = set(labels)
    if text_query and _query_tier(quota_tier, labelset, det_ctx.embedder):
        if (
            install_query_head(
                det_ctx, text_query, hay, labelset, media_type=media_type, beta=_centroid_beta(det_ctx), voted=labelled
            )
            is not None
        ):
            return True
    if tier == TIER_CENTROID:
        from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap

        rows = scoring_rows_for_snap(hay, det_ctx.embedder or None) if hay else None
        install_centroid_head(
            det_ctx,
            [x for x, y in zip(X_list, y_list, strict=True) if y == 1.0],
            (lambda head: score_rows_with_model(head, rows)[0]) if rows is not None else (lambda _head: []),
            labelset,
            beta=_centroid_beta(det_ctx),
            line_mask=_unvoted_mask(rows, labelled),
            ids=rows.ids if rows is not None else None,
        )
        return True
    if tier != TIER_TRAINED:
        return False

    from vtscore.detectors.query_head import typed_query_prior_vector
    from vtscore.detectors.training import train_and_threshold
    from vtscore.training.svm_prior import hard_view_rows, typed_query_prior

    # Pass det_ctx so the fold orderings are cached for a no-retrain re-cut
    # (otherwise a balance change can't move the cutoff — see train_and_threshold).
    # A typed-query detector's head starts from its query as well as the Goods (#4859),
    # and a hard-view face's Good weighs less in it (#5413).
    with (
        typed_query_prior(typed_query_prior_vector(text_query, det_ctx.embedder, media_type), text_query),
        hard_view_rows(hard_view_good_vectors(det_ctx, labelset, snap)),
    ):
        mlp, threshold = train_and_threshold(
            X_list,
            y_list,
            snap=snap,
            embedder_name=det_ctx.embedder or None,
            det_ctx=det_ctx,
            groups=groups,
            score_rows=score_rows,
            voted_ids=labelled,
            haystack=haystack.medias if haystack is not None else None,
            labels=labels,
        )
    from vtscore.detectors.model_loading import labelset_signature

    det_ctx.model = mlp
    det_ctx.model_labels_sig = labelset_signature(labelset)
    det_ctx.threshold = threshold
    return True


def labelset_train_and_score(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    clips_dict: dict[int, dict[str, Any]],
    inclusion_value: int | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float | None = None,
    rows: Any = None,
    on_progress: ProgressCallback | None = None,
    beta: float | None = None,
    label_quota: bool = False,
    text_query: str = "",
) -> tuple[list[dict[str, Any]], float, Any | None]:
    """Train the detector head on the full labelset, then score every media in *clips_dict*.

    Counterpart to :func:`~vtscore.detectors.training.train_and_score` that
    trains on cross-dataset labels.  It assembles ``(X_list, y_list)`` from
    the resolved labelset (populating the embedding cache on the way) and
    then defers the threshold → train → score → format tail to the shared
    :func:`~vtscore.detectors.training._train_and_score_xy` core, so the two
    pipelines stay in lock-step (region-aware scoring, NaN sanitisation,
    population-fused thresholding).  Scoring is still scoped to the active
    dataset's media, since that is what the user is sorting in the UI.

    ``calibration_fraction=None`` (no explicit user setting) resolves inside
    the shared core to the per-space production split for the detector's
    embedder (see
    :func:`~vtscore.detectors.training.resolve_calibration_fraction`).

    *rows* is an optional pre-built
    :class:`~vtscore.detectors.training.ScoringRows` for *clips_dict*, handed
    straight to the shared core so a caller scoring several detectors over one
    uncached snapshot (cross-dataset Find) stacks the corpus once - see
    :func:`~vtscore.detectors.training._train_and_score_xy`.

    *on_progress* is forwarded to :func:`populate_label_embeddings`, which is
    the expensive half whenever the labelset does not resolve into
    *clips_dict*: each unresolved element costs one origin fetch (and, on a
    patch detector, one ``patch_forward``).  A caller driving a progress bar -
    or wanting a cancellation checkpoint - passes it.

    *beta* is the balance to cut at (the line is the labels' line at it,
    #4413, #4452), or ``None`` for no balance (the Inclusion 0 cut).
    *inclusion_value* is deprecated
    (#4269): leave it unset; ``0`` is accepted with a ``DeprecationWarning`` and
    any other value raises ``ValueError``.

    *label_quota* applies the label quota (#4643) as
    :func:`train_from_labelset` does: a caller handing a detector out (a cold
    Find) passes ``True`` and gets the Goods' centroid under the quota, cut on
    *clips_dict*, and nothing with no Good.  The Train view's learned sort
    leaves it ``False``: the head it trains is the sort the user labels on, not
    a detector anyone takes away.  With *label_quota*, *text_query* (the
    detector's typed query, #4758) gives that query's head under the quota
    instead, and past it while the query holds (#4736), cut and scored on
    *clips_dict*'s text space, as :func:`train_from_labelset` does.  Whatever
    *label_quota* says, a trained head on the query's own single-vector space
    starts from *text_query* as well as from the Goods (#4859,
    :func:`~vtscore.detectors.query_head.typed_query_prior_vector`): the Train
    view's learned sort passes the detector's query for that alone.
    """
    from vtscore.config.core_config import _retired_inclusion
    from vtscore.detectors.query_head import typed_query_prior_vector
    from vtscore.detectors.training import _train_and_score_xy, score_rows_with_model
    from vtscore.detectors.training_scoring import _format_results
    from vtscore.training.svm_prior import hard_view_rows, typed_query_prior

    _retired_inclusion("labelset_train_and_score(inclusion_value=...)", inclusion_value)

    populate_label_embeddings(det_ctx, labelset, media_type=media_type, snap=clips_dict, on_progress=on_progress)
    X_list, y_list, groups, score_rows = build_xy_from_labelset(det_ctx, labelset)
    if label_quota:
        from vtscore.detectors.label_quota import TIER_CENTROID, TIER_NONE, quota_from_groups
        from vtscore.detectors.training import detector_score_embedder, scoring_rows_for_snap

        quota_tier = quota_from_groups(groups).tier
        tier = face_ball_tier(quota_tier, media_type or det_ctx.media_type, beta, _n_goods(y_list))
        labelled = set(labeled_media_labels(labelset, clips_dict))
        if text_query and _query_tier(quota_tier, labelset, det_ctx.embedder):
            installed = install_query_head(
                det_ctx, text_query, clips_dict, labelset, media_type=media_type, beta=beta, voted=labelled
            )
            if installed is not None:
                head, threshold, query_rows = installed
                scores, best_region = score_rows_with_model(head, query_rows)
                return _format_results(query_rows.ids, scores, best_region, clips_dict), threshold, head
        if tier == TIER_NONE:
            return [], 0.5, None
        if tier == TIER_CENTROID:
            if rows is None:
                rows = scoring_rows_for_snap(clips_dict, detector_score_embedder(det_ctx, clips_dict))
            centroid_rows = rows
            head, threshold = install_centroid_head(
                det_ctx,
                [x for x, y in zip(X_list, y_list, strict=True) if y == 1.0],
                lambda h: score_rows_with_model(h, centroid_rows)[0],
                labelset,
                beta=beta,
                line_mask=_unvoted_mask(centroid_rows, labelled),
                ids=centroid_rows.ids,
            )
            scores, best_region = score_rows_with_model(head, centroid_rows)
            results = _format_results(centroid_rows.ids, scores, best_region, clips_dict)
            results, threshold = maybe_labelset_structural_rerank(det_ctx, labelset, results, threshold, clips_dict)
            return results, threshold, head
    # A typed-query detector's head starts from its query as well as the Goods (#4859),
    # and a hard-view face's Good weighs less in it (#5413).
    with (
        typed_query_prior(typed_query_prior_vector(text_query, det_ctx.embedder, media_type), text_query),
        hard_view_rows(hard_view_good_vectors(det_ctx, labelset, clips_dict)),
    ):
        results, threshold, model = _train_and_score_xy(
            X_list,
            y_list,
            clips_dict,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            det_ctx=det_ctx,
            groups=groups,
            score_rows=score_rows,
            voted_ids=set(labelset_labels := labeled_media_labels(labelset, clips_dict)),
            rows=rows,
            labels=labelset_labels,
            beta=beta,
        )

    # Stage-2 structural re-rank for a saved structural detector reloaded
    # cross-dataset (the labelset counterpart to the vote-driven re-rank in
    # ``train_and_score``).  A no-op for every non-structural dataset.
    results, threshold = maybe_labelset_structural_rerank(det_ctx, labelset, results, threshold, clips_dict)
    return results, threshold, model
