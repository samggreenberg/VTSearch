"""The heads a labelset under the label quota gives: the Goods' centroid and a typed query's.

Under the quota (#4643, :mod:`~vtscore.detectors.label_quota`) a labelset's
detector is not a trained head: it is the Goods' centroid
(:func:`install_centroid_head`), or for a detector made from a typed query that
query's head (:func:`install_query_head`, #4758).  Both are stored on the
detector's context the same way (:func:`_store_quota_head`), with the line they
were drawn on and the ranking that line is counted on.  :class:`Haystack` names
the population a line is cut on when it is not the snapshot the labels
resolved against.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import numpy as np

from vtscore.datasets.labelset import LabelSet


log = logging.getLogger(__name__)


class Haystack(NamedTuple):
    """A calibration population that is not the label-resolution snapshot.

    ``medias`` is the snapshot the fold-anchored cut is realized on, and
    ``to_source`` maps each of its ids back to the id in *snap* it descends
    from - the identity for a snapshot that is just *snap* renumbered, the
    parent media for every converter output or re-clip sub-item.  The mapping
    is what lets the vote exclusion (#3308) still name the labelled items after
    routing has thrown their ids away.
    """

    medias: dict[int, dict[str, Any]]
    to_source: dict[int, int]


def _centroid_beta(det_ctx) -> float | None:
    """The balance a centroid's line is drawn at: the detector's own, as a trained head's (#4413, #4665)."""
    from vtscore.state import detector_beta, line_knobs
    from vtscore.state.core import DetectorContext

    return detector_beta(det_ctx) if isinstance(det_ctx, DetectorContext) else line_knobs()["beta"]


def _unvoted_mask(rows: Any, labelled: set[int]) -> list[bool] | None:
    """One flag per scored media in *rows*: whether it is not among the *labelled* (#4779)."""
    if rows is None:
        return None
    return [rid not in labelled for rid in rows.ids]


def install_centroid_head(
    det_ctx,
    goods: list[np.ndarray],
    score: Callable[[Any], Any],
    labelset: LabelSet | None,
    *,
    beta: float | None = None,
    line_mask: Any = None,
    ids: Sequence[int] | None = None,
) -> tuple[Any, float]:
    """Store the Goods' centroid head on *det_ctx* as its detector, and return ``(head, threshold)``.

    What a labelset under the quota gives (#4643): see
    :mod:`~vtscore.detectors.centroid_head`.  *score* scores a head over the
    corpus the line is cut on (:func:`~vtscore.detectors.centroid_head.fit_centroid`),
    and *beta* is the balance the line is drawn at (#4732): the detector's own,
    as a trained head's is.  *line_mask* flags the scored media the line counts:
    the unvoted ones (#4779, :func:`_unvoted_mask`).  The line is drawn by the
    detector's media type's rule (:func:`~vtscore.detectors.centroid_head.centroid_line_rule`, #5325).

    Every cache a trained head's retrain leaves for a later re-cut is dropped:
    the centroid has no calibration folds, no fitted cut and no labels line.  It
    keeps its own line instead (``det_ctx.centroid_line``), so a balance change
    redraws it on the cosines it was drawn on.  *ids*, one per score *score*
    returns, name the media it scores: given, the head's scores over them are
    kept as the ranking its line is counted on (#5048), the media *line_mask*
    leaves out marked voted, so the balance counts what the line keeps.
    Without them no ranking is kept, and a Find pass builds one from its own
    scores.  The head is stamped with *labelset*'s signature like a trained one
    (#4204).
    """
    from vtscore.detectors.centroid_head import centroid_line_rule, fit_centroid

    rule = centroid_line_rule(getattr(det_ctx, "media_type", None))
    head, threshold, line = fit_centroid(goods, score, rule=rule, beta=beta, line_mask=line_mask)
    voted = [] if ids is None or line_mask is None else [i for i, n in zip(ids, line_mask, strict=True) if not n]
    _store_quota_head(det_ctx, head, threshold, line, labelset, _quota_ranking(head, score, ids, voted))
    return head, threshold


def _quota_ranking(head: Any, score: Callable[[Any], Any], ids: Sequence[int] | None, voted: Iterable[int]) -> Any:
    """The ranking a quota head's line is counted on (#5048): *head*'s scores over *ids*, *voted* marked.

    *score* scores a head over the corpus the line was drawn on, one score per
    id, as a trained head's retrain ranks its final scores
    (:func:`~vtscore.detectors.training.train_and_threshold`).  ``None`` with
    no *ids*.
    """
    from vtscore.training.thresholds import LineRanking

    if ids is None:
        return None
    return LineRanking.from_scores(ids, score(head), voted)


def _store_quota_head(
    det_ctx, head: Any, threshold: float, line: Any, labelset: LabelSet | None, ranking: Any = None
) -> None:
    """Make *head*, a head the quota gives (the Goods' centroid or a typed query), *det_ctx*'s detector.

    Every cache a trained head's retrain leaves for a later re-cut is dropped;
    the head keeps its own line instead, and *ranking* (:func:`_quota_ranking`)
    as the ranking that line is counted on, or none.  Stamped with *labelset*'s
    signature.
    """
    from vtscore.detectors.model_loading import labelset_signature

    det_ctx.model = head
    det_ctx.model_labels_sig = labelset_signature(labelset)
    det_ctx.threshold = threshold
    det_ctx.calibration_cache = None
    det_ctx.anchored_cut_cache = None
    det_ctx.labels_line = None
    det_ctx.centroid_line = line
    det_ctx.line_ranking = ranking
    det_ctx.gate_passed = None


def _query_tier(resolved_tier: str, labelset: LabelSet, embedder: str | None) -> bool:
    """Whether a typed-query detector gives its query: under the quota, or past it while the query holds.

    Under the quota (#4758) by what resolved *and* by what is labelled.  The
    quota counts the labels that resolved; a labelset that meets it as written
    but whose origins failed to resolve is a resolution failure to report, not
    a detector under the quota, so it does not get the query.  Under the quota
    on a patch space, once the query has missed and a Good resolved to centre
    on, the Goods' centroid answers instead (#4800,
    :func:`~vtscore.detectors.label_quota.query_missed`).  Past the quota, the
    query holds on a single-vector space until
    :data:`~vtscore.detectors.label_quota.QUERY_HOLD_LABELS` labels (#4736, :func:`~vtscore.detectors.label_quota.query_holds`):
    *embedder* is the space the labels were embedded in.
    """
    from vtscore.detectors.label_quota import TIER_CENTROID, TIER_TRAINED, labelset_quota, query_holds, query_missed
    from vtscore.detectors.training_calibration import _patch_space_for

    quota = labelset_quota(labelset)
    if resolved_tier != TIER_TRAINED and quota.tier != TIER_TRAINED:
        missed = query_missed(quota.n_good, quota.n_bad, patch_space=_patch_space_for(embedder))
        return not (missed and resolved_tier == TIER_CENTROID)
    return query_holds(quota.n_good, quota.n_bad, patch_space=_patch_space_for(embedder))


def install_query_head(
    det_ctx,
    text: str,
    snap: dict[int, dict[str, Any]] | None,
    labelset: LabelSet | None,
    *,
    media_type: str,
    beta: float | None = None,
    voted: Iterable[int] = (),
) -> tuple[Any, float, Any] | None:
    """Store a typed query's head on *det_ctx* as its detector: ``(head, threshold, rows)``, or ``None``.

    What a typed-query detector gives until it has a trained head (#4758): see
    :mod:`~vtscore.detectors.query_head`.  *text* is embedded and *snap* scored
    in *snap*'s text space, and the line is the text sort's, drawn over every
    media at *beta*.  The head's scores over *snap* are kept as the ranking that
    line is counted on (#5048), with *voted* (the labelled media, in *snap*'s
    ids) marked voted as a trained head's retrain marks its labelset's, so the
    balance counts what the line keeps of the rest.  *rows* are those text-space
    rows, for a caller that scores the head next.  ``None``, with *det_ctx*
    untouched, when *snap* has no text slot or the query does not embed: the
    caller falls back to the centroid.
    """
    from vtscore.detectors.query_head import embed_detector_query, fit_query_head, text_embedder_for_snap
    from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap

    embedder = text_embedder_for_snap(snap)
    if not text or not snap or not embedder:
        return None
    vec = embed_detector_query(text, media_type, embedder)
    if vec is None:
        return None
    rows = scoring_rows_for_snap(snap, embedder)
    if not rows.ids:
        return None
    if int(np.asarray(vec).size) != int(rows.matrix.shape[1]):
        # A query that embeds in another width than the snapshot's text-slot
        # vectors cannot score them: the centroid answers, as with no text slot.
        log.warning(
            "Typed query %r embeds to %d dims but %r's vectors have %d: falling back to the centroid",
            text,
            int(np.asarray(vec).size),
            embedder,
            int(rows.matrix.shape[1]),
        )
        return None

    def score(h: Any) -> list[float]:
        return score_rows_with_model(h, rows)[0]

    head, threshold, line = fit_query_head(vec, score, beta=beta, embedder=embedder, text=text)
    _store_quota_head(det_ctx, head, threshold, line, labelset, _quota_ranking(head, score, rows.ids, voted))
    return head, threshold, rows
