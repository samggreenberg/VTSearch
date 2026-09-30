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
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Optional

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
    """Maps a :class:`MatchStats` to a match score in ``[0, 1]``: the inlier gate.

    Continuous, crossing :data:`STRUCTURAL_DECISION_THRESHOLD` exactly at
    *min_inliers* (so ``MatchStats.is_match`` and "score >= 0.5" agree) and
    saturating at 1.0 by ``2 * min_inliers``.  :func:`structural_rerank` orders
    fits past saturation by their raw inlier count.
    """

    min_inliers: int = DEFAULT_MIN_INLIERS

    def score(self, stats: MatchStats) -> float:
        """The gate's score for *stats*; 0 when RANSAC found no sane model."""
        if not stats.model_ok:
            return 0.0
        return float(min(1.0, stats.inlier_count / (2.0 * self.min_inliers)))


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
    """
    if not results or not template_features:
        return list(results)

    head = results[:top_k]
    tail = results[top_k:]

    # Verify the whole shortlist in one batched pass rather than pair by pair:
    # the descriptor matching is the bulk of Stage-2 latency and batches into a
    # single (GPU-able) distance computation per template.
    verifiable = [(i, f) for i, e in enumerate(head) if (f := _local_features(snap.get(e.get("id")))) and f.count > 0]
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
        new[score_key] = round(verification, 4)
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
) -> tuple[list[dict], float]:
    """Apply the Stage-2 re-rank when the active dataset is structural.

    A no-op (returns ``(results, threshold)`` unchanged) for every non-
    structural dataset - gated on ``local_features`` being present, exactly as
    the patch path gates on ``patch_grid`` - so existing datasets pay zero
    cost and see no behaviour change.  For a structural dataset it builds the
    RegionYes templates, re-ranks the shortlist by the inlier gate, and returns
    the gate's boundary as the threshold.  Bad votes do not enter Stage 2: the
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

    scorer = VerificationScorer()
    reranked = structural_rerank(
        results,
        snap,
        [tpl for _, tpl in templates],
        scorer,
        matcher,
        top_k=top_k,
        score_key=score_key,
    )
    return reranked, STRUCTURAL_DECISION_THRESHOLD


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
