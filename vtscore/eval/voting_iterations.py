"""Evaluate learned-sort cost over simulated voting iterations.

For each combination of seed *s*, dataset *d*, and target category *c*:

1. Load the dataset and split medias into **D_sim** (simulation) and
   **D_test** (held-out) using *s* to control the random split.
2. Assign ground-truth labels based on *c*: medias whose ``"category"``
   matches *c* are positive (``good``), others are negative (``bad``).
3. Vote on D_sim one item at a time, choosing *which* item to vote on next by
   reproducing the app's **Autopilot** flow (order seeded by *s*).
4. At each step *t* (once at least one good **and** one bad vote exist),
   train a model on votes so far, find a threshold, score D_test, and record
   the inclusion-weighted cost (``fpr_weight * FPR + fnr_weight * FNR``).

Which item the simulated user votes on at each step is chosen by the
``autopilot`` vote-order strategy (see :mod:`vtscore.eval.al_strategies`): seed
from text sort (or a few random known-good examples), then the standard
Good / Bad / Hard / New phases.  This is the only strategy the eval runs — the
point is to measure how the tool itself would function, not to compare
acquisition heuristics.

The result is a :class:`pandas.DataFrame` with columns
``seed, dataset, category, strategy, t, n_good, n_bad, cost, fpr, fnr``.

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
from vtscore.eval.al_strategies import ALContext, select_next
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


# ------------------------------------------------------------------
# Active-learning acquisition helpers
# ------------------------------------------------------------------


def _score_pool(
    model: torch.nn.Sequential,
    pool_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
) -> dict[int, float]:
    """Return ``{pool_id: sigmoid score}`` for the current model over the pool.

    Scores each pool item by its single whole-image vector - the fast path the
    acquisition strategies rank uncertainty over.  This intentionally uses the
    whole-image embedding even for patch datasets (where *test* scoring
    max-pools over regions): acquisition only needs a monotone uncertainty
    signal to order the pool, and the whole-image score is a cheap, adequate
    proxy for that ordering.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not pool_ids:
        return {}
    embs = np.array([media_embedding(clips_dict[cid]) for cid in pool_ids])
    X = torch.tensor(embs, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        scores = torch.sigmoid(model(X)).squeeze(1).cpu().tolist()
    return dict(zip(pool_ids, scores, strict=True))


def _build_eval_atlas(embeddings: dict[int, np.ndarray], min_node_size: int) -> Any:
    """Build a coverage atlas over *embeddings* for the autopilot New phase.

    Returns ``None`` when there are no vectors.  Uses the same hierarchical
    k-means partition the live dataset builds (see
    :class:`~vtscore.state.coverage_atlas.CoverageAtlas`); *min_node_size* is
    exposed so a caller with a small simulation set can drive the partition
    deeper than the production floor (20) and actually resolve density cells.
    """
    from vtscore.state.coverage_atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

    if not embeddings:
        return None
    return CoverageAtlas(
        embeddings,
        k=3,
        max_depth=auto_max_depth(len(embeddings), k=3, min_node_size=min_node_size),
        min_node_size=min_node_size,
    )


def _train_and_calibrate(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
) -> tuple[torch.nn.Sequential, float, int, int]:
    """Train the step's model and calibrate its threshold from the current votes.

    Returns ``(model, threshold, hidden_dim, n_labels)``.  Good votes region-pool
    their ground-truth box when *region_voting* is on (and the media supports
    it); Bad votes always train on the whole-image vector - matching the live
    detector, where only Yes-votes carry a region.

    The threshold matches the production ``_train_and_score_xy`` /
    ``train_and_threshold`` pipeline exactly so the reported cost measures what
    the live detector computes:

    * ``hidden_dim`` is sized from the *full* label count and forced onto the
      calibration folds, so the fold models share the final model's architecture
      (production passes this into ``cross_calibration_threshold_cached``).
      Letting each fold auto-size to its own smaller train split would train
      narrower fold nets and report a threshold the live pipeline never produces.
    * the fold splits use a fresh ``RandomState(42)`` - the fixed seed
      ``cross_calibration_threshold_cached`` always calibrates with - rather than
      the shared per-seed simulation RNG, so the calibration is byte-for-byte
      what production runs for this vote set.  The eval seed still varies the
      data (which media are voted, in what order, and the held-out test split);
      only the calibration folds are pinned, as they are in production.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for vid in good_votes:
        X_list.append(_good_training_vec(clips_dict[vid], target_category, region_voting))
        y_list.append(1.0)
    for vid in bad_votes:
        X_list.append(media_embedding(clips_dict[vid]))
        y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
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
    return model, threshold, hidden_dim, n_labels


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
    strategy: str = "autopilot",
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[int, float]] = None,
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
        strategy: Vote-order strategy naming *which* pool item the simulated
            user labels next (see :data:`vtscore.eval.al_strategies.STRATEGIES`).
            Only ``"autopilot"`` (the default) exists: it reproduces the app's
            real user flow — seed from text sort (or random known-good examples),
            then the standard Good / Bad / Hard / New phases.
        max_steps: Cap on the number of voting steps (pool items labelled).
            ``None`` (default) votes on the entire simulation set.
        atlas_min_node_size: Minimum leaf population for the coverage atlas the
            autopilot New phase reads (default 20, the production floor).  Lower
            it for small simulation sets so diversity cells actually resolve.
        seed_scores: Optional ``{media_id: similarity}`` text-sort ranking (each
            item's cosine to the typed query).  When provided the autopilot seed
            follows the text sort (top items for the initial goods, bottom items
            for the initial bads); ``None`` (default) means the dataset has no
            text sort, so autopilot seeds from random known-good examples.

    Returns:
        List of row dicts with keys ``seed, dataset, category, strategy, t,
        n_good, n_bad, cost, fpr, fnr, elapsed_seconds``. ``n_good``/``n_bad``
        report the vote counts behind each row so callers can tell apart metrics
        learned from a 1-vs-1 model and a many-vs-many one.
    """
    import numpy as np  # noqa: PLC0415

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

    import torch  # noqa: PLC0415

    # Whole-image embeddings of the simulation pool.  These feed the autopilot
    # selector (the example-sort good centroid and the coverage atlas); the
    # Good-vote *training* vector can still be region-pooled below when
    # ``region_voting`` is on.
    sim_embeddings: dict[int, np.ndarray] = {
        cid: np.asarray(media_embedding(clips_dict[cid]), dtype=np.float32) for cid in sim_ids
    }
    input_dim = int(next(iter(sim_embeddings.values())).shape[0])

    # The autopilot New phase reads a coverage atlas built over the pool; it is
    # labelled in lock-step with the votes below so its coverage advances.
    atlas = _build_eval_atlas(sim_embeddings, atlas_min_node_size) if strategy == "autopilot" else None

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
    labeled: dict[int, float] = {}
    rows: list[dict[str, Any]] = []

    # Voting proceeds one item at a time: the autopilot selector picks the next
    # pool item using the *current* detector (trained at the previous step), the
    # item's ground-truth label is revealed, a fresh model is trained on all
    # votes so far, and the coverage atlas is labelled so its New-phase coverage
    # advances.  Before a trainable model exists the selector runs its seed/bad
    # phases (text sort or example sort), so a cold start still makes real picks.
    pool = sorted(sim_ids)
    # Ground-truth pool labels: autopilot draws its random known-good seed
    # examples from the positives here when no text sort is available.  Cheap to
    # build once up front.
    pool_labels = {cid: (1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0) for cid in sim_ids}
    model: torch.nn.Sequential | None = None
    threshold = 0.5
    pool_scores: dict[int, float] = {}
    n_steps = len(pool) if max_steps is None else min(max_steps, len(pool))

    for t in range(1, n_steps + 1):
        if not pool:
            break
        ctx = ALContext(
            pool_ids=pool,
            embeddings=sim_embeddings,
            labeled=labeled,
            scores=pool_scores,
            model=model,
            threshold=threshold,
            atlas=atlas,
            rng=rng,
            pool_labels=pool_labels,
            seed_scores=seed_scores,
        )
        cid = select_next(strategy, ctx)
        pool.remove(cid)
        is_positive = media_is_positive(clips_dict[cid], target_category)
        if is_positive:
            good_votes[cid] = None
            labeled[cid] = 1.0
        else:
            bad_votes[cid] = None
            labeled[cid] = 0.0
        # Mirror the vote onto the coverage atlas so the New phase's next_sample
        # advances past covered regions (the app labels the atlas the same way).
        if atlas is not None and cid in atlas.vector_to_leaf:
            atlas.label(cid, good=is_positive)

        # Need at least 1 good and 1 bad to train
        if not good_votes or not bad_votes:
            model = None
            continue

        model, threshold, _hidden_dim, n_labels = _train_and_calibrate(
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
        )

        # Apply safe threshold blending if enabled
        if safe_thresholds:
            threshold = _blend_safe_threshold(threshold, model, region_aware, sim_clips, X_all_clips, n_labels)

        # Evaluate on held-out test set
        metrics = _evaluate_on_test(
            model, threshold, clips_dict, test_ids, target_category, inclusion, region_aware=region_aware
        )

        # Score the remaining pool with the fresh model so the next step's
        # autopilot Bad / Hard picks rank it.
        pool_scores = _score_pool(model, pool, clips_dict)

        rows.append(
            {
                "seed": seed,
                "dataset": dataset_name,
                "category": target_category,
                "strategy": strategy,
                "t": t,
                "n_good": len(good_votes),
                "n_bad": len(bad_votes),
                **metrics,
                "elapsed_seconds": round(time.monotonic() - start_time, 3),
            }
        )

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
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
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
        strategies: Vote-order strategies to run (see
            :data:`vtscore.eval.al_strategies.STRATEGIES`).  ``None`` (default)
            runs ``["autopilot"]``, the only strategy; the name is recorded in
            the ``strategy`` result column.
        max_steps: Cap on the number of voting steps per run (see
            :func:`simulate_voting_iterations`).
        atlas_min_node_size: Minimum coverage-atlas leaf population for the
            autopilot New phase (see :func:`simulate_voting_iterations`).
        seed_scores: Optional text-sort rankings keyed
            ``{dataset: {category: {media_id: similarity}}}``.  When a
            (dataset, category) has an entry, the autopilot seed follows that
            text ranking; otherwise it seeds from random known-good examples.

    Returns:
        A :class:`~pandas.DataFrame` with columns ``seed, dataset, category,
        strategy, t, n_good, n_bad, cost, fpr, fnr, elapsed_seconds``.
    """
    import pandas as pd  # noqa: PLC0415

    strategy_list = strategies if strategies is not None else ["autopilot"]
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
                cat_seed_scores = (seed_scores or {}).get(ds_name, {}).get(cat)
                for strategy in strategy_list:
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
                        strategy=strategy,
                        max_steps=max_steps,
                        atlas_min_node_size=atlas_min_node_size,
                        seed_scores=cat_seed_scores,
                    )
                    all_rows.extend(rows)

    return pd.DataFrame(
        all_rows,
        columns=pd.Index(
            [
                "seed",
                "dataset",
                "category",
                "strategy",
                "t",
                "n_good",
                "n_bad",
                "cost",
                "fpr",
                "fnr",
                "elapsed_seconds",
            ]
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
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
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
        strategies: Vote-order strategies to run (see
            :func:`run_voting_iterations_eval`).
        max_steps: Cap on the number of voting steps per run.
        atlas_min_node_size: Minimum coverage-atlas leaf population for the
            autopilot New phase.
        seed_scores: Optional text-sort rankings keyed
            ``{dataset: {category: {media_id: similarity}}}`` (see
            :func:`run_voting_iterations_eval`).

    Returns:
        A :class:`~pandas.DataFrame` identical to :func:`run_voting_iterations_eval`
        (columns: ``seed, dataset, category, strategy, t, n_good, n_bad, cost,
        fpr, fnr, elapsed_seconds``).
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
        strategies=strategies,
        max_steps=max_steps,
        atlas_min_node_size=atlas_min_node_size,
        seed_scores=seed_scores,
    )
