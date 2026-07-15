"""Evaluate learned-sort cost over simulated voting iterations.

For each combination of seed *s*, dataset *d*, and target category *c*:

1. Load the dataset and split medias into **D_sim** (simulation) and
   **D_test** (held-out) using *s* to control the random split.
2. Assign ground-truth labels based on *c*: medias whose ``"category"``
   matches *c* are positive (``good``), others are negative (``bad``).
3. Create a shuffled voting sequence from D_sim (order controlled by *s*).
4. Iterate through the voting sequence.  At each step *t* (once at least
   one good **and** one bad vote exist), train a model on votes so far,
   find a threshold, score D_test, and record the inclusion-weighted cost
   (``fpr_weight * FPR + fnr_weight * FNR``).

The result is a :class:`pandas.DataFrame` with columns
``seed, dataset, category, t, n_good, n_bad, cost, fpr, fnr``.

``n_good``/``n_bad`` are the number of good/bad votes the model was trained
on for that row. The very first scored step has only one of each, so its
``cost``/``fpr``/``fnr`` are extremely noisy; these counts let downstream
analysis filter or weight rows by how many votes actually informed them
rather than treating a 1-vs-1 model as if it were as reliable as a 50-vs-50
one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import torch

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.labels import media_is_positive, region_box_for_category
from vtscore.training.mlp import _auto_hidden_dim, train_model
from vtscore.training.thresholds import (
    calculate_cross_calibration_threshold,
    calculate_safe_threshold,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _inclusion_weights(inclusion: int) -> tuple[float, float]:
    """Return ``(fpr_weight, fnr_weight)`` for a given inclusion value."""
    if inclusion >= 0:
        return 1.0, 2.0**inclusion
    return 2.0 ** (-inclusion), 1.0


def _split_media_ids(
    clips_dict: dict[int, dict[str, Any]],
    sim_fraction: float,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Randomly partition media IDs into simulation and test sets."""
    all_ids = sorted(clips_dict.keys())
    shuffled = rng.permutation(all_ids).tolist()
    n_sim = max(1, int(len(shuffled) * sim_fraction))
    return shuffled[:n_sim], shuffled[n_sim:]


VOTE_ORDERS: tuple[str, ...] = ("shuffle", "balanced", "ensemble_std")
"""Supported ``vote_order`` strategies for :func:`simulate_voting_iterations`.

- ``shuffle`` - the historical default: vote on the simulation set in a
  seeded random order.
- ``balanced`` - greedily interleave Good and Bad votes so the running
  label set stays class-balanced (the earliest trainable step is 1-vs-1,
  and the counts never drift far apart afterwards).
- ``ensemble_std`` - active-learning order: at each step train an
  N-member MLP ensemble on the votes so far, score the un-voted pool, and
  vote next on the item the ensemble disagrees about most (highest
  member-to-member sigmoid std).  The ensemble is used *only* to pick the
  next vote; the per-step cost is still measured with the single
  production MLP, exactly as the other orders measure it.
"""


def _make_vote_sequence(
    sim_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    rng: np.random.RandomState,
) -> list[tuple[int, str]]:
    """Build a shuffled list of ``(media_id, label)`` pairs from simulation IDs."""
    votes = [(cid, "good" if media_is_positive(clips_dict[cid], target_category) else "bad") for cid in sim_ids]
    order = rng.permutation(len(votes))
    return [votes[i] for i in order]


def _balanced_vote_sequence(
    sim_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    rng: np.random.RandomState,
) -> list[tuple[int, str]]:
    """Interleave Good/Bad votes so the running label set stays balanced.

    Splits the sim votes by ground-truth class, shuffles each class, then
    greedily appends whichever class is currently under-represented.  The
    first two votes are therefore one Good and one Bad (training starts at
    ``t=2`` with a 1-vs-1 model), and the good/bad counts stay within one
    of each other for the whole sequence - isolating "does class balance
    during voting help?" from the vote *content*, which is identical to
    the ``shuffle`` order's pool.
    """
    goods = [(cid, "good") for cid in sim_ids if media_is_positive(clips_dict[cid], target_category)]
    bads = [(cid, "bad") for cid in sim_ids if not media_is_positive(clips_dict[cid], target_category)]
    rng.shuffle(goods)
    rng.shuffle(bads)

    out: list[tuple[int, str]] = []
    gi = bi = 0
    n_good = n_bad = 0
    while gi < len(goods) or bi < len(bads):
        # Take a Good when both remain and Goods are not ahead; otherwise
        # take whichever class still has items left.
        take_good = (gi < len(goods) and bi < len(bads) and n_good <= n_bad) or bi >= len(bads)
        if take_good and gi < len(goods):
            out.append(goods[gi])
            gi += 1
            n_good += 1
        else:
            out.append(bads[bi])
            bi += 1
            n_bad += 1
    return out


def _good_training_vec(
    media: dict[str, Any],
    target_category: str,
    region_voting: bool,
) -> np.ndarray:
    """Return the training vector for one Good vote on *media*.

    With *region_voting* the simulated user drags the ground-truth box around
    the object: when *media* carries a stored ``patch_grid`` and an annotated
    region for *target_category*, the box is pooled on-the-fly via
    :func:`vtscore.detectors.training.pool_box_from_media` (the same path the
    live region-vote flow uses).  Falls back to the whole-image embedding when
    region voting is off, the media has no patch grid (single-vector
    embedders), or no box is annotated for this category - exactly an
    image-level Good vote.
    """
    if region_voting:
        from vtscore.detectors.training import pool_box_from_media  # noqa: PLC0415

        pooled = pool_box_from_media(media, region_box_for_category(media, target_category))
        if pooled is not None:
            return pooled
    return media_embedding(media)


def _blend_safe_threshold(
    threshold: float,
    model: torch.nn.Sequential,
    region_aware: bool,
    sim_clips: dict[int, dict[str, Any]] | None,
    X_all_clips: Any,
    n_labels: int,
) -> float:
    """Blend *threshold* with a GMM threshold over the simulation set's scores.

    Region-aware datasets score the sim set via region max-pool (matching the
    test-set scoring); single-vector datasets use the pre-computed whole-image
    matrix *X_all_clips*.  Kept separate so the per-step loop stays flat.
    """
    import torch  # noqa: PLC0415

    if region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        assert sim_clips is not None
        all_scores = [r["score"] for r in score_media_with_model(model, sim_clips)]
    else:
        with torch.no_grad():
            X_eval = X_all_clips.to(next(model.parameters()).device)
            all_scores = torch.sigmoid(model(X_eval)).squeeze(1).cpu().tolist()
    return calculate_safe_threshold(threshold, all_scores, n_labels)


def _evaluate_on_test(
    model: torch.nn.Sequential,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    region_aware: bool = False,
) -> dict[str, float]:
    """Score *test_ids* with *model* and return inclusion-weighted cost, FPR, FNR.

    When *region_aware* the test media carry ``patch_regions`` (a patch
    embedder), so scoring max-pools the MLP over every region of each image -
    exactly the live detector's inference for patch datasets (an image scores
    by its best-matching region).  Otherwise each media is scored by its single
    whole-image vector, the fast path used for every single-vector dataset.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not test_ids:
        return {"cost": float("nan"), "fpr": float("nan"), "fnr": float("nan")}

    if region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        test_clips = {cid: clips_dict[cid] for cid in test_ids}
        score_map = {r["id"]: r["score"] for r in score_media_with_model(model, test_clips)}
        scores = [score_map[cid] for cid in test_ids]
    else:
        embs = np.array([media_embedding(clips_dict[cid]) for cid in test_ids])
        X = torch.tensor(embs, dtype=torch.float32)

        with torch.no_grad():
            X = X.to(next(model.parameters()).device)
            scores = torch.sigmoid(model(X)).squeeze(1).cpu().tolist()

    true_labels = [1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in test_ids]

    total_pos = sum(1 for lbl in true_labels if lbl == 1.0)
    total_neg = len(true_labels) - total_pos

    fp = fn = 0
    for score, label in zip(scores, true_labels, strict=True):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 0.0:
            fp += 1
        elif predicted == 0 and label == 1.0:
            fn += 1

    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0

    fpr_weight, fnr_weight = _inclusion_weights(inclusion)
    cost = fpr_weight * fpr + fnr_weight * fnr

    return {"cost": round(cost, 6), "fpr": round(fpr, 6), "fnr": round(fnr, 6)}


def _build_train_xy(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    region_voting: bool,
) -> tuple[list[np.ndarray], list[float]]:
    """Assemble the ``(X_list, y_list)`` training data from the current votes.

    Good votes region-pool their ground-truth box when *region_voting* is on
    (and the media supports it); bad votes always train on the whole-image
    vector - matching the live detector, where only Yes-votes carry a region.
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for vid in good_votes:
        X_list.append(_good_training_vec(clips_dict[vid], target_category, region_voting))
        y_list.append(1.0)
    for vid in bad_votes:
        X_list.append(media_embedding(clips_dict[vid]))
        y_list.append(0.0)
    return X_list, y_list


def _score_step(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    region_aware: bool,
    inclusion: int,
    safe_thresholds: bool,
    sim_clips: dict[int, dict[str, Any]] | None,
    X_all_clips: Any,
    calibrate_count: int,
    calibration_fraction: float,
    test_ids: list[int],
    t: int,
    seed: int,
    dataset_name: str,
    start_time: float,
) -> dict[str, Any]:
    """Train one production MLP on the current votes and score the test set.

    This is the per-step body shared by every ``vote_order``: it trains and
    thresholds exactly the way the live ``_train_and_score_xy`` /
    ``train_and_threshold`` pipeline does (hidden dim sized from the full label
    count and forced onto the calibration folds, folds calibrated with a fresh
    ``RandomState(42)``), evaluates on the held-out test set, and returns one
    result row.  Keeping it single-MLP is deliberate: even under the
    ``ensemble_std`` order the *reported* cost measures what the real detector
    computes; the ensemble only chooses which item to vote on next.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    X_list, y_list = _build_train_xy(good_votes, bad_votes, clips_dict, target_category, region_voting)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]
    n_labels = len(good_votes) + len(bad_votes)

    hidden_dim = _auto_hidden_dim(n_labels)
    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        inclusion,
        rng=np.random.RandomState(42),
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
    )
    model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    if safe_thresholds:
        threshold = _blend_safe_threshold(threshold, model, region_aware, sim_clips, X_all_clips, n_labels)

    metrics = _evaluate_on_test(
        model, threshold, clips_dict, test_ids, target_category, inclusion, region_aware=region_aware
    )

    return {
        "seed": seed,
        "dataset": dataset_name,
        "category": target_category,
        "t": t,
        "n_good": len(good_votes),
        "n_bad": len(bad_votes),
        **metrics,
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
    }


def _ensemble_uncertainty_pick(
    remaining: list[tuple[int, str]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    region_voting: bool,
    n_ensemble: int,
) -> int:
    """Index into *remaining* of the item the ensemble is least sure about.

    Trains *n_ensemble* seed-varied MLPs on the current votes, scores every
    un-voted item's whole-image vector with each member, and returns the index
    of the item with the highest member-to-member sigmoid std (maximal
    epistemic disagreement).  Whole-image vectors are used for the selection
    scoring regardless of region awareness - the pick is an active-learning
    heuristic, so a single fast vector per candidate is enough; the eventual
    per-step cost is still measured region-aware in :func:`_score_step`.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    X_list, y_list = _build_train_xy(good_votes, bad_votes, clips_dict, target_category, region_voting)
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]
    hidden_dim = _auto_hidden_dim(len(good_votes) + len(bad_votes))

    cand_embs = np.array([media_embedding(clips_dict[cid]) for cid, _ in remaining])
    X_cand = torch.tensor(cand_embs, dtype=torch.float32)

    member_scores: list[np.ndarray] = []
    for k in range(n_ensemble):
        # Fixed member seeds (42 + k) keep the pick a deterministic function of
        # the votes cast so far, so the whole ``ensemble_std`` walk is
        # reproducible for a given eval seed.
        model = train_model(X, y, input_dim, seed=42 + k, hidden_dim=hidden_dim)
        with torch.no_grad():
            member_scores.append(torch.sigmoid(model(X_cand)).squeeze(1).cpu().numpy())
    std = np.stack(member_scores, axis=0).std(axis=0)
    return int(np.argmax(std))


def _bootstrap_pick(remaining: list[tuple[int, str]], good_votes: dict[int, None], bad_votes: dict[int, None]) -> int:
    """Pick the next vote before the model is trainable (no ensemble yet).

    Until there is at least one Good and one Bad vote the ensemble can't be
    trained, so this steers toward a trainable state fast: if exactly one class
    is present, take the first item of the missing class; otherwise take the
    first remaining item (the seeded shuffle already fixed the order).
    """
    need: str | None = None
    if good_votes and not bad_votes:
        need = "bad"
    elif bad_votes and not good_votes:
        need = "good"
    if need is not None:
        for i, (_, lbl) in enumerate(remaining):
            if lbl == need:
                return i
    return 0


# ------------------------------------------------------------------
# Single (seed, dataset, category) evaluation
# ------------------------------------------------------------------


def simulate_voting_iterations(  # noqa: C901
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    seed: int,
    dataset_name: str = "",
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    region_voting: bool = False,
    vote_order: str = "shuffle",
    n_ensemble: int = 5,
    max_votes: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Simulate voting on *clips_dict* and evaluate at every step.

    Args:
        clips_dict: Pre-loaded media dict (``{id: clip_data}``).
        target_category: Category treated as the positive class.
        seed: Random seed for splitting and vote ordering.
        dataset_name: Label included in result rows.
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias used for simulated voting.
        safe_thresholds: When ``True``, blend the cross-calibration threshold
            with a GMM-based threshold for robustness with small label counts.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        region_voting: When ``True``, each Good vote trains on the region-pooled
            vector of the media's ground-truth box for *target_category* (the
            minimal box covering every annotated instance), instead of the
            whole-image vector - simulating a user who drags a region around the
            object.  Requires a patch embedder: media without a ``patch_grid``
            or without an annotated box fall back to the whole-image vector.
            Scoring is unaffected by this flag - a patch dataset always scores
            region-aware (max-pool over regions), so the only thing this toggles
            is the Good-vote training vector, isolating region voting's effect.
        vote_order: Which strategy orders the votes (see :data:`VOTE_ORDERS`):
            ``"shuffle"`` (default, seeded random), ``"balanced"`` (interleave
            Good/Bad so the running label set stays balanced), or
            ``"ensemble_std"`` (active learning - vote next on the item an MLP
            ensemble disagrees about most).
        n_ensemble: Number of MLP members trained per step to pick the next
            vote under ``vote_order="ensemble_std"``.  Ignored by the other
            orders (default 5).
        max_votes: Optional cap on how many votes are cast.  ``None`` (default)
            walks the entire simulation set.  Handy for the expensive
            ``ensemble_std`` order, which retrains ``n_ensemble`` models per
            step.

    Returns:
        List of row dicts with keys ``seed, dataset, category, t, n_good,
        n_bad, cost, fpr, fnr, elapsed_seconds``. ``n_good``/``n_bad`` report
        the vote counts behind each row so callers can tell apart metrics
        learned from a 1-vs-1 model and a many-vs-many one.  The row schema is
        identical across every ``vote_order`` - selection strategy never leaks
        into per-step evaluation, which always uses the single production MLP.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if vote_order not in VOTE_ORDERS:
        raise ValueError(f"Unknown vote_order {vote_order!r}; choices: {list(VOTE_ORDERS)}")

    rng = np.random.RandomState(seed)
    # Note: no torch.manual_seed() here - train_model handles its own
    # RNG seeding via fork_rng, keeping it thread-safe.
    start_time = time.monotonic()

    sim_ids, test_ids = _split_media_ids(clips_dict, sim_fraction, rng)

    # Ensure the test set has both positive and negative medias.  Routes through
    # ``media_is_positive`` so multi-label (Visual Genome) images - where the
    # target may be a non-primary category - are counted correctly.
    test_pos = [cid for cid in test_ids if media_is_positive(clips_dict[cid], target_category)]
    test_neg = [cid for cid in test_ids if not media_is_positive(clips_dict[cid], target_category)]
    if not test_pos or not test_neg:
        return []

    # A patch dataset exposes ``patch_regions`` per media; such datasets are
    # scored region-aware (max-pool over regions) the same way the live
    # detector scores them, regardless of how the Good votes were assembled.
    region_aware = any(clips_dict[cid].get("patch_regions") for cid in clips_dict)

    # Pre-compute embeddings for safe-threshold GMM scoring.  Restrict to the
    # simulation set so the held-out ``test_ids`` never feed into the GMM that
    # picks the threshold - otherwise the test scores leak into calibration
    # and the reported metrics are biased upward.  Region-aware datasets keep a
    # sim-set snapshot and score it per-step via region max-pool (to match how
    # the test set is scored); single-vector datasets pre-stack whole-image
    # embeddings once.
    sim_clips: dict[int, dict[str, Any]] | None = None
    X_all_clips: Any = None
    if safe_thresholds and region_aware:
        sim_clips = {cid: clips_dict[cid] for cid in sim_ids}
    elif safe_thresholds:
        gmm_clip_embs = np.array([media_embedding(clips_dict[cid]) for cid in sorted(sim_ids)])
        X_all_clips = torch.tensor(gmm_clip_embs, dtype=torch.float32)

    good_votes: dict[int, None] = {}
    bad_votes: dict[int, None] = {}
    rows: list[dict[str, Any]] = []

    # Shared per-step scoring: identical across vote orders, so the reported
    # cost always measures the single production MLP (the ensemble, when used,
    # only influences which item is voted on next).
    def _step(t: int) -> None:
        rows.append(
            _score_step(
                good_votes,
                bad_votes,
                clips_dict,
                target_category,
                region_voting=region_voting,
                region_aware=region_aware,
                inclusion=inclusion,
                safe_thresholds=safe_thresholds,
                sim_clips=sim_clips,
                X_all_clips=X_all_clips,
                calibrate_count=calibrate_count,
                calibration_fraction=calibration_fraction,
                test_ids=test_ids,
                t=t,
                seed=seed,
                dataset_name=dataset_name,
                start_time=start_time,
            )
        )

    if vote_order == "ensemble_std":
        # Adaptive walk: the next vote depends on the current ensemble's
        # uncertainty, so the sequence can't be precomputed.  The seeded
        # shuffle only fixes the starting pool order and the bootstrap /
        # tie-break order before the model is trainable.
        remaining = _make_vote_sequence(sim_ids, clips_dict, target_category, rng)
        cap = len(remaining) if max_votes is None else min(max_votes, len(remaining))
        for t in range(1, cap + 1):
            if not good_votes or not bad_votes:
                idx = _bootstrap_pick(remaining, good_votes, bad_votes)
            else:
                idx = _ensemble_uncertainty_pick(
                    remaining, good_votes, bad_votes, clips_dict, target_category, region_voting, n_ensemble
                )
            cid, label = remaining.pop(idx)
            if label == "good":
                good_votes[cid] = None
            else:
                bad_votes[cid] = None
            if not good_votes or not bad_votes:
                continue
            _step(t)
        return rows

    # Static orders: the whole sequence is fixed up front.
    if vote_order == "balanced":
        vote_seq = _balanced_vote_sequence(sim_ids, clips_dict, target_category, rng)
    else:  # "shuffle"
        vote_seq = _make_vote_sequence(sim_ids, clips_dict, target_category, rng)
    if max_votes is not None:
        vote_seq = vote_seq[:max_votes]

    for t, (cid, label) in enumerate(vote_seq, start=1):
        if label == "good":
            good_votes[cid] = None
        else:
            bad_votes[cid] = None

        # Need at least 1 good and 1 bad to train
        if not good_votes or not bad_votes:
            continue

        _step(t)

    return rows


# ------------------------------------------------------------------
# Full evaluation across seeds x datasets x categories
# ------------------------------------------------------------------


def run_voting_iterations_eval(
    dataset_clips: dict[str, dict[int, dict[str, Any]]],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    region_voting: bool = False,
    vote_order: str = "shuffle",
    n_ensemble: int = 5,
    max_votes: Optional[int] = None,
) -> pd.DataFrame:
    """Run the voting-iterations evaluation over multiple seeds/datasets/categories.

    Args:
        dataset_clips: Mapping of dataset name to a pre-loaded medias dict.
            Each medias dict maps ``int`` media IDs to media data dicts
            (must carry a resolvable embedding in the per-embedder
            ``"embeddings"`` store and a ``"category"`` key).
        seeds: List of random seeds to iterate over.
        categories: Optional mapping of dataset name to list of target
            categories.  If ``None`` or a dataset is missing from the dict,
            all unique categories in that dataset are used.
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias reserved for simulated voting.
        safe_thresholds: When ``True``, blend thresholds with GMM
            (see :func:`simulate_voting_iterations`).
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        region_voting: When ``True``, Good votes train on the ground-truth
            region-pooled vector for patch datasets (see
            :func:`simulate_voting_iterations`).
        vote_order: Vote-ordering strategy - ``"shuffle"``, ``"balanced"``, or
            ``"ensemble_std"`` (see :func:`simulate_voting_iterations`).
        n_ensemble: Ensemble size for ``vote_order="ensemble_std"`` selection.
        max_votes: Optional cap on votes cast per (seed, dataset, category).

    Returns:
        A :class:`~pandas.DataFrame` with columns ``seed, dataset, category,
        t, n_good, n_bad, cost, fpr, fnr, elapsed_seconds``.
    """
    import pandas as pd  # noqa: PLC0415

    all_rows: list[dict[str, Any]] = []

    for ds_name, clips_dict in dataset_clips.items():
        # Determine target categories.  For multi-label datasets each image's
        # ``category`` is only its primary, so fall back to the union of every
        # image's ``categories`` list when present.
        if categories and ds_name in categories:
            target_cats = categories[ds_name]
        else:
            cat_set: set[str] = set()
            for cid in clips_dict:
                media = clips_dict[cid]
                cat_set.update(media.get("categories") or [media["category"]])
            target_cats = sorted(cat_set)

        for seed in seeds:
            for cat in target_cats:
                rows = simulate_voting_iterations(
                    clips_dict,
                    target_category=cat,
                    seed=seed,
                    dataset_name=ds_name,
                    inclusion=inclusion,
                    sim_fraction=sim_fraction,
                    safe_thresholds=safe_thresholds,
                    calibrate_count=calibrate_count,
                    calibration_fraction=calibration_fraction,
                    region_voting=region_voting,
                    vote_order=vote_order,
                    n_ensemble=n_ensemble,
                    max_votes=max_votes,
                )
                all_rows.extend(rows)

    return pd.DataFrame(
        all_rows,
        columns=pd.Index(
            ["seed", "dataset", "category", "t", "n_good", "n_bad", "cost", "fpr", "fnr", "elapsed_seconds"]
        ),
    )


def run_voting_iterations_eval_from_pickles(
    dataset_paths: dict[str, str],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    region_voting: bool = False,
    vote_order: str = "shuffle",
    n_ensemble: int = 5,
    max_votes: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience wrapper that loads datasets from pickle files.

    Args:
        dataset_paths: Mapping of dataset name to pickle file path.
        seeds: List of random seeds.
        categories: Optional category filter (see :func:`run_voting_iterations_eval`).
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias for simulation.
        safe_thresholds: When ``True``, blend thresholds with GMM.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        region_voting: When ``True``, Good votes train on the ground-truth
            region-pooled vector for patch datasets (see
            :func:`simulate_voting_iterations`).
        vote_order: Vote-ordering strategy - ``"shuffle"``, ``"balanced"``, or
            ``"ensemble_std"`` (see :func:`simulate_voting_iterations`).
        n_ensemble: Ensemble size for ``vote_order="ensemble_std"`` selection.
        max_votes: Optional cap on votes cast per (seed, dataset, category).

    Returns:
        A :class:`~pandas.DataFrame` identical to :func:`run_voting_iterations_eval`
        (columns: ``seed, dataset, category, t, n_good, n_bad, cost, fpr, fnr,
        elapsed_seconds``).
    """
    from vtscore.datasets.loader import load_dataset_from_pickle

    dataset_clips: dict[str, dict[int, dict[str, Any]]] = {}
    for name, path in dataset_paths.items():
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(Path(path), medias)
        dataset_clips[name] = medias

    return run_voting_iterations_eval(
        dataset_clips,
        seeds=seeds,
        categories=categories,
        inclusion=inclusion,
        sim_fraction=sim_fraction,
        safe_thresholds=safe_thresholds,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        region_voting=region_voting,
        vote_order=vote_order,
        n_ensemble=n_ensemble,
        max_votes=max_votes,
    )
