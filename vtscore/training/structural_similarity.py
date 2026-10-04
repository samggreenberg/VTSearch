"""Stage-2 geometric re-rank by the inlier gate.

This is the chokepoint the structural-embedder design
(``docs/plans/structural-embedder.md``) calls for: the one place that knows
the **two-stage** rule, the way :func:`score_against_query` is the one place
that knows max-over-regions for patch embedders.

* **Stage 1** is the ordinary VLAD retrieval - it rides
  ``media["embeddings"]`` through the MLP / cosine sort unchanged and produces a
  ranked candidate list.
* **Stage 2** (here) takes the top-*K* of that list and, for each candidate,
  geometrically verifies it against the **templates** derived from the user's
  RegionYes votes (RANSAC similarity fit via the dataset's
  :class:`~vtscore.media.structural.StructuralMatcher`).  Candidates are
  re-ranked by the best template's inlier count, reported as the inlier gate's
  score.

The verification score is in ``[0, 1]`` and crosses
:data:`STRUCTURAL_DECISION_THRESHOLD` at ``DEFAULT_MIN_INLIERS``, so "score >=
threshold" means "geometrically a match".  Votes add templates; they do not
train a scorer.  A match-statistic MLP trained from the votes used to replace
the gate from 3 votes on, and it ranked worse than the inliers on documents and
photos alike, its accept decision included (#4169).

Library-tier and import-clean: no Flask, no app-tier imports.  ``torch`` and the
embedder registry are imported lazily so the pure data helpers
(:func:`filter_features_to_box`, :func:`build_templates`,
:func:`best_match_stats`) import without them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from vtscore.training.structural_stage1 import (
    VerificationCache,
    example_queries,
    snapshot_has_tiles,
    tiled_stage1,
    tiled_top_k,
    vote_queries,
)
from vtscore.media.structural import (
    DEFAULT_MIN_INLIERS,
    MatchStats,
    StructuralFeatures,
    StructuralMatcher,
)

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunable constants (pinned by the pre-impl spike; see the design doc's open
# questions on K, live-vs-on-demand, and the geometric model).
# --------------------------------------------------------------------------

DEFAULT_RERANK_TOP_K = 50
"""Stage-1 shortlist size fed to the Stage-2 RANSAC re-rank.

Too small misses true matches the coarse VLAD stage under-ranks; too large
blows the re-rank latency budget (Stage 2 is O(K * match * RANSAC)).  Candidates
beyond the shortlist are not geometrically verified and score 0 - the accepted
K trade-off for an instance-search tool.
"""

STRUCTURAL_DECISION_THRESHOLD = 0.5

#: Decimals a verification score is stored with, and the line is rounded to the same way
#: (#4464): an unrounded line dropped pages at exactly the Bad ceiling + 1 inliers whenever
#: their stored score rounded down. At 6 decimals two adjacent inlier counts stay apart up
#: to ~2,800 inliers (n / (n + 8) moves by 8 / ((n + 8)(n + 9))); at 4 only to ~275.
SCORE_DECIMALS = 6

#: The recall end of the balance (#4458, owner 2026-10-03). From beta >= RECALL_BETA (nearer the
#: preset 4 than 1, in log space) the returned set is the beta-1 line's set plus every verified page
#: with at least max(RECALL_MIN_INLIERS, ceil(RECALL_GOOD_FRACTION x the Goods' median leave-one-out
#: inliers)) inliers, whatever the Bad ceiling and the geometry cuts say. Class-split CV on FullMarks
#: tier m chose the floor for beta 4 (both folds alike); the union with the beta-1 set keeps the
#: slider monotone: F4 share of the best cut +0.066 [+0.030, +0.109] over clicks 0-25, no click worse.
#: Beta 1/4 and 1 keep the shipped line (round 2 found no rule that passed for them).
RECALL_BETA = 2.0
RECALL_MIN_INLIERS = 10
RECALL_GOOD_FRACTION = 0.25

#: Geometry cuts for the returned set before a detector's first Bad vote (#4440). Until
#: then #4367's Bad ceiling is just the 8-inlier gate, which passes hard negatives on
#: documents. A fit must also have inlier ratio >= this and median reprojection error <=
#: :data:`GEOMETRY_REPROJ_MAX` (normalised units). Both were fit on FullMarks tier ``s``'s
#: box-template pairs past the gate (#4434), and scored out of sample on tier ``m``:
#: +0.23 F1 at 10 clicks.
GEOMETRY_RATIO_MIN = 0.75
GEOMETRY_REPROJ_MAX = 0.004887
"""Decision boundary of the verification score.

The inlier gate maps ``inlier_count == DEFAULT_MIN_INLIERS`` to 0.5.
"""


# --------------------------------------------------------------------------
# RegionYes-as-template
# --------------------------------------------------------------------------


def filter_features_to_box(
    features: StructuralFeatures,
    box: Optional[tuple[float, float, float, float]],
) -> StructuralFeatures:
    """Keep only the keypoints whose location falls inside *box*.

    This is RegionYes-as-template: "find the Coca-Cola logo" boxes the logo and
    discards the surrounding clutter a whole-image template would drag in.  When
    *box* is ``None`` (a whole-image Yes) the features are returned unchanged.
    *box* is a normalised ``(x0, y0, x1, y1)`` and need not be ordered.

    If the box contains no keypoints the unfiltered features are returned - an
    empty template can never verify anything, so falling back to the full set is
    the more useful behaviour than silently producing a dead template.
    """
    if box is None:
        return features
    kp = features.keypoints_f32()
    if kp.shape[0] == 0:
        return features
    x0, x1 = sorted((float(box[0]), float(box[2])))
    y0, y1 = sorted((float(box[1]), float(box[3])))
    xs, ys = kp[:, 0], kp[:, 1]
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    if not inside.any():
        return features
    return StructuralFeatures(keypoints=kp[inside], descriptors=features.descriptors_f32()[inside])


def _local_features(media: Optional[dict]) -> Optional[StructuralFeatures]:
    """Return *media*'s stored ``local_features`` as a :class:`StructuralFeatures`."""
    if media is None:
        return None
    feats = media.get("local_features")
    return feats if isinstance(feats, StructuralFeatures) else None


def build_templates(
    good_votes: Any,
    snap: dict[Any, dict],
    region_boxes: dict[Any, tuple[float, float, float, float]],
) -> list[tuple[Any, StructuralFeatures]]:
    """Build one ``(media_id, template)`` per Good vote that carries features.

    RegionYes votes restrict the template to the boxed keypoints
    (:func:`filter_features_to_box`); whole-image Yes votes keep every keypoint.
    Good votes whose media has no ``local_features`` (or isn't in *snap*) are
    skipped.  The media id is retained so a caller can tell templates apart
    (``best_match_stats_many``'s *skip*).
    """
    templates: list[tuple[Any, StructuralFeatures]] = []
    for cid in good_votes:
        feats = _local_features(snap.get(cid))
        if feats is None or feats.count == 0:
            continue
        templates.append((cid, filter_features_to_box(feats, region_boxes.get(cid))))
    return templates


# --------------------------------------------------------------------------
# Max-over-templates verification
# --------------------------------------------------------------------------


def best_match_stats(
    template_features: list[StructuralFeatures],
    candidate: StructuralFeatures,
    matcher: StructuralMatcher,
) -> MatchStats:
    """Verify *candidate* against every template; return the strongest fit.

    Multiple RegionYes votes mean multiple templates; the score is the **max
    over templates** (keep the best geometric fit), directly analogous to
    patch's max-over-regions.  "Best" orders by ``(model_ok, inlier_count,
    inlier_ratio)`` so a plausible model always beats a degenerate one.
    """
    best = MatchStats()
    best_key = (False, -1, -1.0)
    for tpl in template_features:
        stats = matcher.verify(tpl, candidate)
        key = (stats.model_ok, stats.inlier_count, stats.inlier_ratio)
        if key > best_key:
            best_key = key
            best = stats
    return best


def best_match_stats_many(
    templates: list[tuple[Any, StructuralFeatures]],
    candidates: list[StructuralFeatures],
    matcher: StructuralMatcher,
    *,
    skip: Optional[Any] = None,
) -> list[MatchStats]:
    """Max-over-templates :class:`MatchStats` for *each* candidate, batched.

    The list form of :func:`best_match_stats`.  When the matcher exposes a
    ``verify_many`` (``SiftMatcher`` does), each template is matched against the
    **whole candidate list in one batched pass**, which is what lets the
    descriptor matching run as a single GPU-able distance computation instead of
    one ``cv2.BFMatcher`` call per pair.  A matcher without ``verify_many`` -
    any third-party :class:`~vtscore.media.structural.StructuralMatcher`, which
    the protocol never required to have one - falls back to the per-pair loop and
    behaves exactly as before.

    *skip*, when given, is called as ``skip(template_key, candidate_index)`` and
    suppresses that template for that candidate, e.g. to hold an item's own
    template out (leave-one-out).
    """
    if not templates or not candidates:
        return [MatchStats() for _ in candidates]

    verify_many = getattr(matcher, "verify_many", None)
    if verify_many is None:
        out: list[MatchStats] = []
        for i, cand in enumerate(candidates):
            usable = [tpl for key, tpl in templates if skip is None or not skip(key, i)]
            out.append(best_match_stats(usable, cand, matcher))
        return out

    best = [MatchStats() for _ in candidates]
    best_keys = [(False, -1, -1.0) for _ in candidates]
    for key, tpl in templates:
        for i, stats in enumerate(verify_many(tpl, candidates)):
            if skip is not None and skip(key, i):
                continue
            cur = (stats.model_ok, stats.inlier_count, stats.inlier_ratio)
            if cur > best_keys[i]:
                best_keys[i] = cur
                best[i] = stats
    return best


# --------------------------------------------------------------------------
# Verification score (the inlier gate)
# --------------------------------------------------------------------------


@dataclass
class VerificationScorer:
    """Maps a :class:`MatchStats` to a match score in ``[0, 1)``: ``n / (n + min_inliers)``.

    Monotone in the inlier count *n*, crossing :data:`STRUCTURAL_DECISION_THRESHOLD`
    exactly at *min_inliers* (so ``MatchStats.is_match`` and "score >= 0.5" agree).
    It never saturates, so a threshold above the gate (the Bad ceiling, #4367) is
    still a threshold on this scale (:meth:`threshold_for`).
    """

    min_inliers: int = DEFAULT_MIN_INLIERS
    #: Optional geometry cuts (#4440): a fit with a lower inlier ratio or a larger median
    #: reprojection error scores half its value, below the line but in the same order.
    ratio_min: Optional[float] = None
    reproj_max: Optional[float] = None
    #: With geometry cuts, a fit with at least this many inliers is not demoted however loose
    #: it is: the recall end of the balance keeps it (#4458).
    loose_ok_from: Optional[int] = None

    def score(self, stats: MatchStats) -> float:
        """The score for *stats*; 0 when RANSAC found no sane model."""
        if not stats.model_ok:
            return 0.0
        value = self.threshold_for(stats.inlier_count)
        loose = (self.ratio_min is not None and stats.inlier_ratio < self.ratio_min) or (
            self.reproj_max is not None and stats.median_reproj_error > self.reproj_max
        )
        if loose and self.loose_ok_from is not None and stats.inlier_count >= self.loose_ok_from:
            return value
        return value / 2.0 if loose else value

    def threshold_for(self, inliers: float) -> float:
        """The score at which a fit has exactly *inliers* inliers."""
        n = max(0.0, float(inliers))
        return float(n / (n + self.min_inliers)) if n > 0 else 0.0


# --------------------------------------------------------------------------
# Stage-1 -> Stage-2 chokepoint
# --------------------------------------------------------------------------


def structural_rerank(
    results: list[dict],
    snap: dict[Any, dict],
    template_features: list[StructuralFeatures],
    scorer: VerificationScorer,
    matcher: StructuralMatcher,
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    score_key: str = "score",
    template_keys: Optional[Sequence[Any]] = None,
    cache: Optional[Any] = None,
    parents: Optional[dict[Any, tuple[Any, StructuralFeatures]]] = None,
) -> list[dict]:
    """Re-rank Stage-1 *results* by geometric verification of the top-*K*.

    *results* is the Stage-1-sorted list of ``{"id", score_key, ...}`` dicts.
    The top-*K* are geometrically verified against *template_features*
    (max-over-templates) and re-ordered by the resulting verification score,
    then by raw inlier count (the gate saturates at ``2 * min_inliers``, and past
    that more inliers is still the stronger fit, #4169);
    each gets that score in *score_key* and the inlier bounding box in
    ``best_region`` (reusing patch's overlay machinery).  Candidates beyond the
    shortlist are left in Stage-1 order behind the re-ranked block and scored 0
    - they were never geometrically checked, so for an instance search they are
    "no confirmed match".

    Order and score stay consistent (an item's position matches its reported
    score within each block) so the existing threshold/colouring path needs no
    special-casing.  When there are no templates the input is returned unchanged.

    With a *cache* (a :class:`~vtscore.training.structural_stage1.VerificationCache`)
    and one *template_keys* entry per template, fits already computed on an
    earlier retrain are reused and only new (template, page) pairs are verified.
    *parents* (pruned template key -> unpruned ``(key, features)``) limits a
    stop-listed template to the pages its unpruned self passes (#4432).
    """
    if not results or not template_features:
        return list(results)

    head = results[:top_k]
    tail = results[top_k:]

    # Verify the whole shortlist in one batched pass rather than pair by pair:
    # the descriptor matching is the bulk of Stage-2 latency and batches into a
    # single (GPU-able) distance computation per template.
    verifiable = [(i, f) for i, e in enumerate(head) if (f := _local_features(snap.get(e.get("id")))) and f.count > 0]
    if cache is not None and template_keys is not None:
        batched = cache.best_many(
            list(zip(template_keys, template_features)),
            [(head[i].get("id"), f) for i, f in verifiable],
            matcher,
            parents=parents,
        )
    else:
        batched = best_match_stats_many([(None, tpl) for tpl in template_features], [f for _, f in verifiable], matcher)
    stats_by_pos = {pos: st for (pos, _), st in zip(verifiable, batched)}

    scored: list[tuple[float, int, float, dict]] = []
    for pos, entry in enumerate(head):
        verification = 0.0
        inliers = 0
        box: Optional[tuple[float, float, float, float]] = None
        stats = stats_by_pos.get(pos)
        if stats is not None:
            verification = scorer.score(stats)
            inliers = stats.inlier_count if stats.model_ok else 0
            box = stats.inlier_box
        new = dict(entry)
        stage1 = float(new.get(score_key, 0.0) or 0.0)
        new[score_key] = round(verification, SCORE_DECIMALS)
        if box is not None:
            new["best_region"] = [float(c) for c in box]
        else:
            new.pop("best_region", None)
        scored.append((verification, inliers, stage1, new))

    # Re-rank the shortlist by verification score, then inliers, breaking the
    # remaining ties by the Stage-1 score so a strong VLAD candidate wins among
    # equally-(un)verified items.
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    out = [s[3] for s in scored]

    for entry in tail:
        new = dict(entry)
        new[score_key] = 0.0
        new.pop("best_region", None)
        out.append(new)
    return out


# --------------------------------------------------------------------------
# App-facing glue (gated, no-op for non-structural datasets)
# --------------------------------------------------------------------------


def snapshot_is_structural(snap: dict[Any, dict]) -> bool:
    """True iff any media in *snap* carries populated ``local_features``."""
    return any(_local_features(m) is not None for m in snap.values())


def _resolve_matcher(snap: dict[Any, dict]) -> Optional[StructuralMatcher]:
    """Resolve the dataset's structural matcher from its embedder, or ``None``.

    Backend-agnostic: asks the embedder for its
    :class:`~vtscore.media.structural.StructuralMatcher` rather than hard-coding
    SIFT, so a learned-feature backend slots in unchanged.
    """
    name = ""
    for m in snap.values():
        name = m.get("embedder") or ""
        if name:
            break
    if not name:
        return None
    from vtscore.media import get_embedder  # noqa: PLC0415

    try:
        emb = get_embedder(name)
    except KeyError:
        return None
    return getattr(emb, "structural_matcher", None)


def maybe_structural_rerank(
    results: list[dict],
    threshold: float,
    snap: dict[Any, dict],
    good_votes: Any,
    region_boxes: dict[Any, tuple[float, float, float, float]],
    det_ctx: Any = None,
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    score_key: str = "score",
    feature_snap: Optional[dict[Any, dict]] = None,
    bad_votes: Any = None,
    beta: Optional[float] = None,
) -> tuple[list[dict], float]:
    """Apply the Stage-2 re-rank when the active dataset is structural.

    A no-op (returns ``(results, threshold)`` unchanged) for every non-
    structural dataset - gated on ``local_features`` being present, exactly as
    the patch path gates on ``patch_grid`` - so existing datasets pay zero
    cost and see no behaviour change.  For a structural dataset it builds the
    RegionYes templates, re-ranks the shortlist by the inlier gate, and returns
    the gate's boundary as the threshold.

    On a tiled dataset with Bad votes (*bad_votes*), the returned threshold is
    the Bad ceiling instead: a page must fit better than every Bad did
    (:func:`_bad_ceiling_threshold`, #4367). Bads still never enter the ranking
    (#4169).

    **On a tiled dataset** (``sift_vlad_doc``, pages carrying ``tile_vectors``)
    Stage 1 is replaced too.  The caller's *results* (the detector head's
    page-VLAD ranking, near chance on documents) give way to the tiled Stage 1:
    max over the Good boxes' queries x each page's tiles.  The shortlist grows to
    :func:`~vtscore.training.structural_stage1.tiled_top_k`, and fits are kept on
    *det_ctx* across retrains (#3928).  Bad votes do not enter Stage 2: the
    match-statistic MLP that learned from them ranked worse than the gate
    (#4169).

    *feature_snap* is the source of the template ``local_features`` (keyed the
    same way as *good_votes*),
    defaulting to *snap*.  The vote-driven path leaves it ``None`` because the
    voted media live in the active dataset; the **labelset** path passes a
    synthetic snapshot of re-derived cross-dataset features so a saved
    structural detector can verify against templates from datasets that aren't
    currently loaded.  The re-rank itself always runs over *snap* (the media the
    user is actually sorting).
    """
    if not snapshot_is_structural(snap):
        return results, threshold
    matcher = _resolve_matcher(snap)
    if matcher is None:
        return results, threshold
    feat_snap = feature_snap if feature_snap is not None else snap
    templates = build_templates(good_votes, feat_snap, region_boxes)
    if not templates:
        return results, threshold

    if det_ctx is not None:
        try:
            # The threshold returned below is the gate's boundary, not a
            # cut on the retrieval MLP's scale, so the MLP-scale estimators the
            # Stage-1 pass cached no longer describe it.  Left in place, the next
            # re-cut (a floor or Inclusion change, the acquisition cut) would
            # replace this boundary with an MLP-scale threshold and
            # apply it to verification scores.
            det_ctx.anchored_cut_cache = None
            det_ctx.calibration_cache = None
            det_ctx.line_ranking = None
        except Exception:  # noqa: BLE001 - request-missing sentinel refuses writes
            pass

    template_keys = None
    cache = None
    parents: dict[Any, tuple[Any, StructuralFeatures]] = {}
    if snapshot_has_tiles(snap):
        cache = _verification_cache(det_ctx)
        prune_tags: dict[Any, Any] = {}
        unpruned = dict(templates)
        if STOPLIST_POLICY != "off" and bad_votes:
            templates, prune_tags = _stoplist(templates, bad_votes, feat_snap, matcher, cache, region_boxes)
        boxed = {cid: tpl for cid, tpl in templates if region_boxes.get(cid) is not None}
        queries = vote_queries(good_votes, feat_snap, region_boxes, boxed)
        if queries is not None:
            results = tiled_stage1(snap, queries, score_key)
            top_k = tiled_top_k(len(results))
            template_keys = [
                (cid, region_boxes.get(cid), id(feat_snap[cid].get("local_features")), prune_tags.get(cid))
                for cid, _ in templates
            ]
            parents = {
                key: ((cid, region_boxes.get(cid), key[2], None), unpruned[cid])
                for key, (cid, _tpl) in zip(template_keys, templates)
                if key[3] is not None
            }
        else:
            cache = None

    threshold_out = STRUCTURAL_DECISION_THRESHOLD
    if template_keys is not None and cache is not None and bad_votes:
        threshold_out = _bad_ceiling_threshold(
            list(zip(template_keys, [tpl for _, tpl in templates])), bad_votes, feat_snap, matcher, cache
        )
    threshold_out, recall_floor = _recall_line(
        threshold_out, beta, template_keys, [tpl for _, tpl in templates], good_votes, feat_snap, matcher, cache
    )
    scorer = _line_scorer(template_keys is not None, bad_votes, feat_snap, loose_ok_from=recall_floor)
    reranked = _rerank_growing(
        results,
        snap,
        [tpl for _, tpl in templates],
        matcher,
        top_k=top_k,
        score_key=score_key,
        template_keys=template_keys,
        cache=cache,
        tiled=template_keys is not None,
        scorer=scorer,
        parents=parents,
    )
    return reranked, threshold_out


#: Stop-list from Bad votes (#4170 / #4180, pre-registered arms): ``"off"`` (shipped),
#: ``"all"`` prunes each Good template against every Bad, ``"gated"`` only against the
#: Bads that clear the gate for that template.
STOPLIST_POLICY = "off"
#: The Lowe ratio the stop-list's matches use (#4162's arm 6).
_STOPLIST_RATIO = 0.75


def _stoplist(
    templates: list[tuple[Any, StructuralFeatures]],
    bad_votes: Any,
    feature_snap: dict[Any, dict],
    matcher: StructuralMatcher,
    cache: Optional[VerificationCache],
    region_boxes: dict[Any, tuple[float, float, float, float]],
) -> tuple[list[tuple[Any, StructuralFeatures]], dict[Any, Any]]:
    """Each Good template without the descriptors a Bad page also matches; ``(templates, prune tags)``.

    A descriptor that passes the ratio test against a Bad vote's page is part of
    what lets that Bad match (a letterhead rule, a font's glyphs), so it is not
    the mark. #4180's guard keeps one that also passes against another Good's
    page. A tag per pruned template (the dropped indices) keys the verification
    cache, so a template that changes is re-verified and one that does not keeps
    its fits.
    """
    from vtscore.media.structural import ratio_test_matches  # noqa: PLC0415

    bads = {b: f for b in bad_votes if (f := _local_features(feature_snap.get(b))) is not None and f.count > 0}
    if not bads:
        return templates, {}
    goods = {cid: _local_features(feature_snap.get(cid)) for cid, _ in templates}
    out: list[tuple[Any, StructuralFeatures]] = []
    tags: dict[Any, Any] = {}
    for cid, tpl in templates:
        use = list(bads.values())
        if STOPLIST_POLICY == "gated" and cache is not None:
            key = (cid, region_boxes.get(cid), id(goods[cid]), None)
            fits = cache.best_many([(key, tpl)], list(bads.items()), matcher)
            use = [f for (b, f), s in zip(bads.items(), fits) if s.model_ok and s.inlier_count >= DEFAULT_MIN_INLIERS]
        desc = tpl.descriptors_f32()
        if not use or desc.shape[0] < 2:
            out.append((cid, tpl))
            continue
        drop = np.zeros(desc.shape[0], dtype=bool)
        for t_idx, _c in ratio_test_matches(desc, [f.descriptors_f32() for f in use], ratio=_STOPLIST_RATIO):
            drop[t_idx] = True
        others = [g for o, g in goods.items() if o != cid and g is not None and g.count > 0]
        if others and drop.any():
            confirmed = np.zeros(desc.shape[0], dtype=bool)
            for t_idx, _c in ratio_test_matches(desc, [g.descriptors_f32() for g in others], ratio=_STOPLIST_RATIO):
                confirmed[t_idx] = True
            drop &= ~confirmed
        if not drop.any() or drop.all():
            out.append((cid, tpl))
            continue
        keep = ~drop
        out.append((cid, StructuralFeatures(keypoints=tpl.keypoints_f32()[keep], descriptors=desc[keep])))
        tags[cid] = hash(np.flatnonzero(drop).tobytes())
    return out, tags


def _line_scorer(
    tiled: bool, bad_votes: Any, feature_snap: dict[Any, dict], *, loose_ok_from: Optional[int] = None
) -> VerificationScorer:
    """The scorer behind the returned set: geometry cuts on a tiled dataset with no Bad vote yet (#4440).

    Until a Bad exists the Bad ceiling is only the 8-inlier gate, so a verified page
    must also fit tightly (:data:`GEOMETRY_RATIO_MIN`, :data:`GEOMETRY_REPROJ_MAX`).
    At the recall end of the balance a fit with *loose_ok_from* inliers passes loose (#4458).
    """
    if tiled and not any(_local_features(feature_snap.get(b)) for b in (bad_votes or ())):
        return VerificationScorer(
            ratio_min=GEOMETRY_RATIO_MIN, reproj_max=GEOMETRY_REPROJ_MAX, loose_ok_from=loose_ok_from
        )
    return VerificationScorer()


def _recall_line(
    threshold: float,
    beta: Optional[float],
    template_keys: Optional[Sequence[Any]],
    template_features: list[StructuralFeatures],
    good_votes: Any,
    feature_snap: dict[Any, dict],
    matcher: StructuralMatcher,
    cache: Optional[VerificationCache],
) -> tuple[float, Optional[int]]:
    """``(line, floor)`` for the balance (#4458): at the recall end the beta-1 *threshold* or the floor's.

    Below :data:`RECALL_BETA`, or off a tiled dataset, it is *threshold* unchanged and no floor.
    """
    if template_keys is None or cache is None or beta is None or beta < RECALL_BETA:
        return threshold, None
    floor = _recall_floor(list(zip(template_keys, template_features)), good_votes, feature_snap, matcher, cache)
    return min(threshold, round(VerificationScorer().threshold_for(floor), SCORE_DECIMALS)), floor


def _recall_floor(
    templates: list[tuple[Any, StructuralFeatures]],
    good_votes: Any,
    feature_snap: dict[Any, dict],
    matcher: StructuralMatcher,
    cache: VerificationCache,
) -> int:
    """The recall end's inlier floor (#4458): max(10, ceil(0.25 x the Goods' median leave-one-out inliers)).

    A Good's leave-one-out fit is its best fit to the other Goods' templates, so the floor follows how
    well this detector's own marks match each other: a faint mark's floor stays at 10.
    """
    loo: list[int] = []
    for g in good_votes:
        feats = _local_features(feature_snap.get(g))
        others = [(k, t) for k, t in templates if k[0] != g]
        if feats is None or feats.count == 0 or not others:
            continue
        (stats,) = cache.best_many(others, [(g, feats)], matcher)
        loo.append(stats.inlier_count if stats.model_ok else 0)
    if len(loo) < 2:
        return RECALL_MIN_INLIERS
    return max(RECALL_MIN_INLIERS, math.ceil(RECALL_GOOD_FRACTION * float(np.median(loo))))


def _bad_ceiling_threshold(
    templates: list[tuple[Any, StructuralFeatures]],
    bad_votes: Any,
    feature_snap: dict[Any, dict],
    matcher: StructuralMatcher,
    cache: VerificationCache,
) -> float:
    """The returned set's line on a tiled dataset: above the best fit any Bad vote reached (#4367).

    The fixed 8-inlier gate passes hard negatives on a document page at 8,192
    keypoints. A Bad vote tells us how well a page that is not the mark can fit
    these templates, so a page is accepted only if it fits better than every Bad.
    That took the returned set's F1 from 0.43 to 0.85 at 25 clicks on FullMarks
    (#4367's pre-registered R1). With no Bads, or none that fit, it is the gate.
    """
    bads = [(b, f) for b in bad_votes if (f := _local_features(feature_snap.get(b))) is not None and f.count > 0]
    scorer = VerificationScorer()
    if not bads:
        return STRUCTURAL_DECISION_THRESHOLD
    fits = cache.best_many(templates, bads, matcher)
    ceiling = max((s.inlier_count if s.model_ok else 0) for s in fits)
    return round(scorer.threshold_for(max(scorer.min_inliers, ceiling + 1)), SCORE_DECIMALS)


def _rerank_growing(
    results: list[dict],
    snap: dict[Any, dict],
    templates: list[StructuralFeatures],
    matcher: StructuralMatcher,
    *,
    top_k: int,
    score_key: str,
    template_keys: Optional[Sequence[Any]],
    cache: Optional[VerificationCache],
    tiled: bool,
    scorer: Optional[VerificationScorer] = None,
    parents: Optional[dict[Any, tuple[Any, StructuralFeatures]]] = None,
) -> list[dict]:
    """:func:`structural_rerank`, then, on a tiled dataset, more blocks while the shortlist's tail still verifies.

    Growth follows :data:`~vtscore.training.structural_stage1.K_POLICY` (#4391).
    Under the shipped ``"fixed"`` policy this is exactly one re-rank.
    """
    from vtscore.training import structural_stage1 as s1  # noqa: PLC0415

    scorer = scorer or VerificationScorer()
    stage1_ids = [e.get("id") for e in results]
    if tiled and cache is None:
        # No detector to keep fits on (a one-off sort): still never verify a page twice while growing.
        cache = VerificationCache()
    while True:
        reranked = structural_rerank(
            results,
            snap,
            templates,
            scorer,
            matcher,
            top_k=top_k,
            score_key=score_key,
            template_keys=template_keys,
            cache=cache,
            parents=parents,
        )
        verified = {e["id"]: float(e.get(score_key, 0.0) or 0.0) for e in reranked[:top_k]}
        if not (tiled and s1.should_extend(stage1_ids, verified, top_k)):
            break
        top_k = min(len(results), top_k + s1.EXTEND_STEP, s1.TILED_K_CAP)
    s1.LAST_TOP_K = top_k
    return reranked


def _verification_cache(det_ctx: Any) -> Optional[VerificationCache]:
    """The detector's verification cache, created on first use; ``None`` without a context."""
    if det_ctx is None:
        return None
    try:
        cache = getattr(det_ctx, "structural_verification_cache", None)
        if cache is None:
            cache = VerificationCache()
            det_ctx.structural_verification_cache = cache
        return cache
    except Exception:  # noqa: BLE001 - request-missing sentinel refuses writes
        return None


def maybe_structural_rerank_example(
    results: list[dict],
    threshold: float,
    snap: dict[Any, dict],
    example_features: Optional[StructuralFeatures] | Sequence[Optional[StructuralFeatures]],
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    score_key: str = "score",
) -> tuple[list[dict], float]:
    """Stage-2 re-rank for the example-sort (seed-by-example) path.

    The templates are the **uploaded examples' own** local features rather than
    vote-derived ones: the user can crop an upload to the pattern they want to
    match before it is embedded, so the crop already restricts the template (no
    ``region_box`` filtering needed here).  *example_features* is one
    :class:`~vtscore.media.structural.StructuralFeatures` or a sequence of them
    - one per example - and a candidate is scored as the **max over
    templates**, the same rule :func:`maybe_structural_rerank` applies to a
    detector's RegionYes templates.  Several crops of one mark, or several
    marks, therefore widen what can verify rather than being averaged away
    (Stage 1 already folds the examples into a centroid; the geometry does
    not, because a VLAD centroid of two logos matches neither).

    The inlier gate (:class:`VerificationScorer`) scores the fits, with its
    boundary at :data:`STRUCTURAL_DECISION_THRESHOLD`, as on the vote path.

    A no-op for non-structural datasets and when no example yielded features
    (an empty template can never verify anything, so the Stage-1 cosine order
    is left intact); an empty template among usable ones is simply dropped.
    """
    if not snapshot_is_structural(snap):
        return results, threshold
    if example_features is None or isinstance(example_features, StructuralFeatures):
        candidates: list[Optional[StructuralFeatures]] = [example_features]
    else:
        candidates = list(example_features)
    templates = [f for f in candidates if f is not None and f.count > 0]
    if not templates:
        return results, threshold
    matcher = _resolve_matcher(snap)
    if matcher is None:
        return results, threshold
    if snapshot_has_tiles(snap) and (queries := example_queries(templates)) is not None:
        # Tiled dataset: the crops' VLADs against every page's tiles replace the
        # page-VLAD cosine, and the shortlist grows (#3928).
        results = tiled_stage1(snap, queries, score_key)
        top_k = tiled_top_k(len(results))
        from vtscore.training import structural_stage1 as s1  # noqa: PLC0415

        s1.LAST_TOP_K = top_k
    scorer = VerificationScorer()
    reranked = structural_rerank(
        results,
        snap,
        templates,
        scorer,
        matcher,
        top_k=top_k,
        score_key=score_key,
    )
    return reranked, STRUCTURAL_DECISION_THRESHOLD
