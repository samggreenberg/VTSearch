"""Stage-2 geometric re-rank + match-statistic verification classifier.

This is the chokepoint the structural-embedder design
(``docs/plans/structural-embedder.md``) calls for: the one place that knows
the **two-stage** rule, the way :func:`score_against_query` is the one place
that knows max-over-regions for patch embedders.

* **Stage 1** is the ordinary VLAD retrieval - it rides
  ``media["embedding"]`` through the MLP / cosine sort unchanged and produces a
  ranked candidate list.
* **Stage 2** (here) takes the top-*K* of that list and, for each candidate,
  geometrically verifies it against the **templates** derived from the user's
  RegionYes votes (RANSAC similarity fit via the dataset's
  :class:`~vtscore.media.structural.StructuralMatcher`).  Candidates are
  re-ranked by a verification score: either the learned **match-statistic
  classifier** (when there are enough votes to train it) or, before then, the
  cold-start inlier gate.

The verification score is a probability in ``[0, 1]``; the classifier's
decision boundary (and the cold-start gate's ``DEFAULT_MIN_INLIERS`` mapping)
both sit at :data:`STRUCTURAL_DECISION_THRESHOLD`, so "score >= threshold"
means "geometrically a match" in either regime.

Library-tier and import-clean: no Flask, no app-tier imports.  ``torch`` and the
embedder registry are imported lazily so the pure data helpers
(:func:`filter_features_to_box`, :func:`build_templates`,
:func:`best_match_stats`) import without them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from vtscore.media.structural import (
    DEFAULT_MIN_INLIERS,
    MatchStats,
    StructuralFeatures,
    StructuralMatcher,
    match_stats_to_features,
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

MIN_VERIFICATION_VOTES = 3
"""Vote count below which the match-statistic classifier is not trained.

Mirrors the safe-threshold GMM fallback the detector MLP uses below 6 labels:
with too few votes we fall back to :class:`VerificationScorer`'s cold-start
inlier gate instead of a useless classifier.
"""

STRUCTURAL_DECISION_THRESHOLD = 0.5
"""Decision boundary of the verification score.

The match-statistic classifier emits a sigmoid probability whose boundary is
0.5; the cold-start gate maps ``inlier_count == DEFAULT_MIN_INLIERS`` to 0.5
too, so a single threshold separates match/non-match in both regimes.
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
    skipped.  The media id is retained so the verification-classifier training
    can hold the item's own template out (leave-one-out) and avoid a trivial
    self-match positive.
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


# --------------------------------------------------------------------------
# Verification classifier (the genuinely-structural learnable)
# --------------------------------------------------------------------------


@dataclass
class VerificationScorer:
    """Maps a :class:`MatchStats` to a match probability in ``[0, 1]``.

    Wraps either the trained match-statistic classifier (*model*) or, before
    there are enough votes to train one, the cold-start inlier gate.  Both put
    their decision boundary at :data:`STRUCTURAL_DECISION_THRESHOLD`.
    """

    model: Optional[Any] = None  # nn.Sequential | None
    min_inliers: int = DEFAULT_MIN_INLIERS

    def score(self, stats: MatchStats) -> float:
        """Probability that *stats* represents a genuine instance match."""
        if self.model is None:
            # Cold-start: a continuous gate that crosses 0.5 exactly at
            # ``min_inliers`` (so ``MatchStats.is_match`` and "score >= 0.5"
            # agree) and saturates at 1.0 by ``2 * min_inliers``.
            if not stats.model_ok:
                return 0.0
            return float(min(1.0, stats.inlier_count / (2.0 * self.min_inliers)))

        import torch  # noqa: PLC0415

        from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

        feat = match_stats_to_features(stats)
        with torch.no_grad():
            x = torch.from_numpy(feat).unsqueeze(0).to(next(self.model.parameters()).device)
            prob = sigmoid_to_finite_scores(self.model(x))[0]
        # ``sigmoid_to_finite_scores`` sentinels non-finite logits to -1.0;
        # clamp that to 0.0 so a destabilised classifier never out-ranks a real
        # match (and never reports a negative "probability").
        return float(prob) if prob >= 0.0 else 0.0


def train_verification_classifier(
    templates: list[tuple[Any, StructuralFeatures]],
    good_votes: Any,
    bad_votes: Any,
    snap: dict[Any, dict],
    matcher: StructuralMatcher,
) -> Optional[Any]:
    """Train the match-statistic classifier from RegionYes / No votes.

    Each labelled item is verified against the templates (max-over-templates),
    its :class:`MatchStats` stacked into the fixed-D
    :func:`~vtscore.media.structural.match_stats_to_features` vector, and a tiny
    MLP trained over those vectors: RegionYes / Yes that verify are positives,
    No are negatives.  The classifier's decision boundary **is** the calibrated
    match threshold, so threshold calibration falls out for free.

    A Good vote's own template is held out when computing its positive example
    (leave-one-out) so a trivial self-match doesn't dominate training.  Returns
    ``None`` (cold-start) when there are fewer than
    :data:`MIN_VERIFICATION_VOTES` votes, when either class ends up empty, or
    when training is otherwise impossible - the caller then uses the inlier
    gate.
    """
    if len(good_votes) + len(bad_votes) < MIN_VERIFICATION_VOTES:
        return None

    feats: list[np.ndarray] = []
    labels: list[float] = []
    all_templates = [tpl for _, tpl in templates]

    for cid in good_votes:
        cand = _local_features(snap.get(cid))
        if cand is None or cand.count == 0:
            continue
        # Hold out this item's own template (leave-one-out) to avoid a trivial
        # self-match positive.  If it was the only template, there is nothing to
        # verify against, so skip it as a training example.
        loo = [tpl for tc, tpl in templates if tc != cid]
        if not loo:
            continue
        feats.append(match_stats_to_features(best_match_stats(loo, cand, matcher)))
        labels.append(1.0)

    for cid in bad_votes:
        cand = _local_features(snap.get(cid))
        if cand is None or cand.count == 0:
            continue
        feats.append(match_stats_to_features(best_match_stats(all_templates, cand, matcher)))
        labels.append(0.0)

    num_pos = sum(1 for v in labels if v == 1.0)
    num_neg = len(labels) - num_pos
    if len(feats) < 2 or num_pos == 0 or num_neg == 0:
        return None

    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    X = torch.from_numpy(np.stack(feats).astype(np.float32, copy=False))
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    try:
        return train_model(X, y, X.shape[1])
    except ValueError:
        # train_model rejects single-class data; guarded above, but stay defensive.
        return None


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
    (max-over-templates) and re-ordered by the resulting verification score;
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

    scored: list[tuple[float, float, dict]] = []
    for entry in head:
        feats = _local_features(snap.get(entry.get("id")))
        verification = 0.0
        box: Optional[tuple[float, float, float, float]] = None
        if feats is not None and feats.count > 0:
            stats = best_match_stats(template_features, feats, matcher)
            verification = scorer.score(stats)
            box = stats.inlier_box
        new = dict(entry)
        stage1 = float(new.get(score_key, 0.0) or 0.0)
        new[score_key] = round(verification, 4)
        if box is not None:
            new["best_region"] = [float(c) for c in box]
        else:
            new.pop("best_region", None)
        scored.append((verification, stage1, new))

    # Re-rank the shortlist by verification score, breaking ties by the Stage-1
    # score so a strong VLAD candidate wins among equally-(un)verified items.
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out = [s[2] for s in scored]

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
    bad_votes: Any,
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
    the patch path gates on ``patch_regions`` - so existing datasets pay zero
    cost and see no behaviour change.  For a structural dataset it builds the
    RegionYes templates, trains (or cold-starts) the verification classifier,
    re-ranks the shortlist, and returns the classifier's decision boundary as
    the threshold.  The trained classifier is carried on *det_ctx* (in-memory,
    re-derived every retrain) alongside the retrieval MLP.

    *feature_snap* is the source of the template / classifier-candidate
    ``local_features`` (keyed the same way as *good_votes* / *bad_votes*),
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

    classifier = train_verification_classifier(templates, good_votes, bad_votes, feat_snap, matcher)
    if det_ctx is not None:
        try:
            det_ctx.verification_classifier = classifier
        except Exception:  # noqa: BLE001 - request-missing sentinel refuses writes
            pass

    scorer = VerificationScorer(model=classifier)
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
    example_features: Optional[StructuralFeatures],
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    score_key: str = "score",
) -> tuple[list[dict], float]:
    """Stage-2 re-rank for the example-sort (seed-by-example) path.

    The template is the **uploaded example's own** local features rather than a
    vote-derived one: the user can crop the upload to the pattern they want to
    match before it is embedded, so the crop already restricts the template (no
    ``region_box`` filtering needed here).  There are no votes, so there is no
    match-statistic classifier to train - the cold-start inlier gate
    (:class:`VerificationScorer` with no model) scores the fits, with its
    boundary at :data:`STRUCTURAL_DECISION_THRESHOLD` like every other regime.

    A no-op for non-structural datasets and when the example yielded no
    features (an empty template can never verify anything, so the Stage-1
    cosine order is left intact).
    """
    if not snapshot_is_structural(snap):
        return results, threshold
    if example_features is None or example_features.count == 0:
        return results, threshold
    matcher = _resolve_matcher(snap)
    if matcher is None:
        return results, threshold
    scorer = VerificationScorer()  # cold-start: example-sort carries no votes
    reranked = structural_rerank(
        results,
        snap,
        [example_features],
        scorer,
        matcher,
        top_k=top_k,
        score_key=score_key,
    )
    return reranked, STRUCTURAL_DECISION_THRESHOLD
