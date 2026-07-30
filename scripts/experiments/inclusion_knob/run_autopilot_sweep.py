"""Selection-bias sweep, driven by the **canonical Autopilot** vote order.

Supersedes the ``toplist`` arm of :mod:`run_selection_sweep`, which hand-rolled
a greedy top-of-sort labeling loop.  That was not what VTSearch does: real
Autopilot votes the top of the ranking only for its first few Goods, takes its
Bads from the *bottom*, and then spends every remaining vote alternating
**Hard** (the item nearest the decision threshold) and **New** (the coverage
atlas's next under-explored region).  Margin-plus-diversity sampling biases
the calibration positives in the opposite direction from top-greedy, so the
earlier arm could not answer the question it was built for.

This stage mirrors the production simulation loop in
:func:`vtscore.eval.voting_iterations.simulate_voting_iterations` and picks
every vote with the repo's own selector
(:func:`vtscore.eval.al_strategies.select_next`, strategy ``autopilot``) over a
real :class:`~vtscore.state.coverage_atlas.CoverageAtlas`, so the vote order is
the app's by construction rather than by imitation.

Policies compared:

* ``autopilot`` - the canonical selector: text-sort seed for the first
  ``_GOOD_TARGET`` Goods (AG News gets a genuine E5 ``"query: ..."`` embedding;
  synthetic arms have no text, so the selector takes its designed
  random-known-good seed path), lowest-scored items for the first
  ``_BAD_TARGET`` Bads, then the Hard/New interleave.  One model fit +
  cross-calibration per vote, exactly as the app retrains per vote.
* ``uniform`` - stratified random votes: the exchangeable reference the
  conformal rule assumes.

Unlike the earlier sweep, votes are drawn from a **simulation half** and every
metric is measured on a held-out **test half** (``SIM_FRACTION``, as the eval
framework does), so removing high-scoring items by voting on them can no
longer depress the evaluation set's own positive quantiles.

Designs scored at each checkpoint: production ``conformal``, the production
safe-``blend``, and an ``oracle`` (the same rule fed the test half's true
scores and labels) that isolates threshold placement from model quality.

Usage::

    python run_autopilot_sweep.py [--quick] [--out CSV]
"""

from __future__ import annotations

import argparse
import itertools

import common

common.setup_env()

import numpy as np  # noqa: E402

SEEDS = range(4)
CHECKPOINTS = (12, 24, 50, 100)
POLICIES = ("uniform", "autopilot")
DESIGNS = ("conformal", "blend", "oracle")
INCLUSIONS = (-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10)
CALIBRATE_COUNT = 2
CALIBRATION_FRACTION = 0.5
VOTE_POS_FRACTION = 1 / 3  # uniform arm only; autopilot's ratio is emergent
SIM_FRACTION = 0.5  # half the items votable, half held out for metrics
ATLAS_MIN_NODE_SIZE = 20  # production floor; sim halves here are >= 1200 items
#: Text a user would plausibly type per AG News category, embedded with E5's
#: ``"query: "`` prefix to seed the Autopilot text sort (mirrors
#: :func:`vtscore.eval.seed_scores.build_seed_scores`).
AGNEWS_QUERIES = {
    "Business": "business and finance news",
    "Sci/Tech": "science and technology news",
    "Sports": "sports news",
    "World": "world news and international affairs",
}


def _load_arms() -> dict[str, tuple[np.ndarray, np.ndarray] | str]:
    """Arm name -> (X, y_binary) for AG News; synthetic arms resolve per-seed."""
    arms: dict = {}
    npz = common.CACHE / "agnews_e5.npz"
    if npz.exists():
        ag = np.load(npz, allow_pickle=True)
        X, y, cats = ag["X"], ag["y"], [str(c) for c in ag["categories"]]
        for ci, cat in enumerate(cats):
            arms[f"agnews:{cat}"] = (X, (y == ci).astype(np.int8))
    else:
        common.log(f"WARNING: {npz} missing - run prepare_agnews.py first; sweeping synthetic arms only")
    for level in ("easy", "medium", "hard"):
        arms[f"synth:{level}"] = level
    return arms


#: Per-arm text-sort ranking cache: the query embedding depends only on the
#: arm's category, so every seed reuses it instead of reloading E5.
_SEED_SCORE_CACHE: dict[str, dict[int, float]] = {}


def _agnews_seed_scores(X: np.ndarray, arm: str) -> dict[int, float] | None:
    """Cosine of every item to a real E5-embedded text query, or ``None``.

    Only AG News arms get a text sort: they are a text dataset, so a typed
    query is exactly how a user seeds Autopilot.  Synthetic arms have no text,
    which is the case the selector's random-known-good seed path exists for.
    """
    if not arm.startswith("agnews:"):
        return None
    if arm in _SEED_SCORE_CACHE:
        return _SEED_SCORE_CACHE[arm]
    category = arm.split(":", 1)[1]
    from sentence_transformers import SentenceTransformer

    from vtscore.config import E5_MODEL_ID

    model = SentenceTransformer(E5_MODEL_ID)
    q = model.encode(f"query: {AGNEWS_QUERIES[category]}", normalize_embeddings=True)
    sims = X @ np.asarray(q, dtype=np.float32)
    _SEED_SCORE_CACHE[arm] = {int(i): float(s) for i, s in enumerate(sims)}
    return _SEED_SCORE_CACHE[arm]


def _split_sim_test(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified sim/test halves, so both carry positives at pool prevalence."""
    rng = np.random.default_rng(1000 + seed)
    sim, test = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        cut = int(round(len(idx) * SIM_FRACTION))
        sim.append(idx[:cut])
        test.append(idx[cut:])
    return np.concatenate(sim), np.concatenate(test)


def _votes_uniform(y: np.ndarray, sim_ids: np.ndarray, n_votes: int, seed: int) -> list[int]:
    """Stratified random votes from the simulation half: ~1/3 positive."""
    rng = np.random.default_rng(seed)
    n_pos = max(2, round(n_votes * VOTE_POS_FRACTION))
    pos = sim_ids[y[sim_ids] == 1]
    neg = sim_ids[y[sim_ids] == 0]
    return [
        int(i)
        for i in np.concatenate(
            [
                rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
                rng.choice(neg, size=min(n_votes - n_pos, len(neg)), replace=False),
            ]
        )
    ]


def _threshold_for_votes(
    X: np.ndarray,
    y: np.ndarray,
    votes: list[int],
    inclusion: int,
    hidden_dim: int,
) -> float:
    """Production cross-calibration threshold over *votes* (fresh RandomState(42))."""
    from vtscore.training.thresholds import compute_fold_orderings, threshold_from_fold_orderings

    orderings, fallback = compute_fold_orderings(
        list(np.asarray(X[votes], dtype=np.float32)),
        [float(v) for v in y[votes]],
        X.shape[1],
        rng=np.random.RandomState(42),
        calibrate_count=CALIBRATE_COUNT,
        calibration_fraction=CALIBRATION_FRACTION,
        hidden_dim=hidden_dim,
    )
    if fallback is not None:
        return fallback
    return threshold_from_fold_orderings(orderings, inclusion)


def _votes_autopilot(
    X: np.ndarray,
    y: np.ndarray,
    sim_ids: np.ndarray,
    seed: int,
    seed_scores: dict[int, float] | None,
    max_votes: int,
    checkpoints: tuple[int, ...],
) -> dict[int, list[int]]:
    """Run the canonical Autopilot vote order; snapshot the vote set per checkpoint.

    Mirrors :func:`vtscore.eval.voting_iterations.simulate_voting_iterations`'s
    loop: the selector picks from the *current* detector's scores, the ground
    truth is revealed, the atlas is labelled so the New phase advances, and a
    fresh model + production threshold are computed for the next pick.  One run
    to ``max_votes`` yields every checkpoint, so the trajectory is shared
    (as a real session's is) instead of re-simulated per vote count.
    """
    import torch

    from vtscore.eval.al_strategies import ALContext, select_next
    from vtscore.eval.voting_iterations import _build_eval_atlas
    from vtscore.training.mlp import _auto_hidden_dim, train_model
    from vtscore.utils.scores import sigmoid_to_finite_array

    rng = np.random.RandomState(seed)
    sim_embeddings = {int(i): X[i] for i in sim_ids}
    pool_labels = {int(i): float(y[i]) for i in sim_ids}
    atlas = _build_eval_atlas(sim_embeddings, ATLAS_MIN_NODE_SIZE)

    pool = sorted(int(i) for i in sim_ids)
    votes: list[int] = []
    labeled: dict[int, float] = {}
    pool_scores: dict[int, float] = {}
    model = None
    threshold = 0.5
    snapshots: dict[int, list[int]] = {}

    while len(votes) < max_votes and pool:
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
        cid = select_next("autopilot", ctx)
        pool.remove(cid)
        is_positive = pool_labels[cid] == 1.0
        votes.append(cid)
        labeled[cid] = 1.0 if is_positive else 0.0
        if atlas is not None and cid in atlas.vector_to_leaf:
            atlas.label(cid, good=is_positive)

        n_good = sum(1 for v in labeled.values() if v == 1.0)
        if n_good and n_good < len(labeled):
            hidden_dim = _auto_hidden_dim(len(votes))
            X_t = torch.from_numpy(np.ascontiguousarray(X[votes], dtype=np.float32))
            y_t = torch.tensor([labeled[i] for i in votes], dtype=torch.float32).unsqueeze(1)
            model = train_model(X_t, y_t, X.shape[1], hidden_dim=hidden_dim)
            # The Hard phase measures |p - threshold|, so the threshold driving
            # the next pick must be the production one at the user's inclusion
            # (0 = the default the app ships).
            threshold = _threshold_for_votes(X, y, votes, 0, hidden_dim)
            with torch.no_grad():
                X_pool = torch.from_numpy(np.ascontiguousarray(X[pool], dtype=np.float32))
                X_pool = X_pool.to(next(model.parameters()).device)
                scores = sigmoid_to_finite_array(model(X_pool)).astype(np.float64)
            pool_scores = {int(i): float(s) for i, s in zip(pool, scores, strict=True)}
        if len(votes) in checkpoints:
            snapshots[len(votes)] = list(votes)
    return snapshots


def _composition_stats(
    orderings: list[tuple[list[float], list[float]]],
    test_scores: np.ndarray,
    test_truth: np.ndarray,
    y_votes: np.ndarray,
) -> dict[str, float]:
    """Calibration-vs-population diagnostics: where the k=0 cut is read from."""
    cal_pos = np.array([s for ss, ll in orderings for s, lb in zip(ss, ll, strict=True) if lb == 1.0])
    test_pos = test_scores[test_truth == 1]
    return {
        "vote_pos_frac": float(np.mean(y_votes == 1)),
        "n_cal_pos": int(len(cal_pos)),
        "cal_pos_q25": float(np.quantile(cal_pos, 0.25)) if len(cal_pos) else float("nan"),
        "test_pos_q25": float(np.quantile(test_pos, 0.25)) if len(test_pos) else float("nan"),
        "cal_pos_min": float(cal_pos.min()) if len(cal_pos) else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the autopilot selection-bias sweep.")
    parser.add_argument("--quick", action="store_true", help="1 seed, checkpoints up to 24 (smoke test)")
    parser.add_argument("--out", default=str(common.RESULTS / "autopilot_sweep.csv"))
    args = parser.parse_args(argv)

    import pandas as pd
    import torch

    import knobs
    import synthetic
    from vtscore.embedding.loader import ensure_torch_configured
    from vtscore.training.mlp import _auto_hidden_dim, train_model
    from vtscore.training.thresholds import (
        CONFORMAL_BASE_BUDGET,
        calculate_safe_threshold,
        compute_fold_orderings,
        conformal_threshold,
        threshold_from_fold_orderings,
    )
    from vtscore.utils.scores import sigmoid_to_finite_array

    ensure_torch_configured()

    seeds = [0] if args.quick else list(SEEDS)
    checkpoints = (12, 24) if args.quick else CHECKPOINTS

    arms = _load_arms()
    rows: list[dict] = []
    cells = list(itertools.product(sorted(arms), seeds))
    for ci, (arm, seed) in enumerate(cells):
        spec = arms[arm]
        X, y = synthetic.make_synthetic(spec, seed) if isinstance(spec, str) else spec
        sim_ids, test_ids = _split_sim_test(y, seed)
        seed_scores = _agnews_seed_scores(X, arm)
        test_truth = y[test_ids].astype(np.int8)

        # One Autopilot trajectory per (arm, seed) yields every checkpoint.
        auto_snapshots = _votes_autopilot(X, y, sim_ids, seed, seed_scores, max(checkpoints), checkpoints)

        for policy, n_votes in itertools.product(POLICIES, checkpoints):
            votes = (
                _votes_uniform(y, sim_ids, n_votes, seed) if policy == "uniform" else auto_snapshots.get(n_votes, [])
            )
            y_votes = y[votes].astype(np.float64) if votes else np.array([])
            if len(votes) < 4 or min(np.sum(y_votes == 1), np.sum(y_votes == 0)) < 2:
                common.log(f"  {arm} seed={seed} {policy} n={n_votes}: SKIPPED (not calibratable)")
                continue

            hidden_dim = _auto_hidden_dim(len(votes))
            X_t = torch.from_numpy(np.ascontiguousarray(X[votes], dtype=np.float32))
            y_t = torch.tensor(y_votes, dtype=torch.float32).unsqueeze(1)
            model = train_model(X_t, y_t, X.shape[1], hidden_dim=hidden_dim)
            with torch.no_grad():
                X_test = torch.from_numpy(np.ascontiguousarray(X[test_ids], dtype=np.float32))
                X_test = X_test.to(next(model.parameters()).device)
                test_scores = sigmoid_to_finite_array(model(X_test)).astype(np.float64)

            orderings, fallback = compute_fold_orderings(
                list(np.asarray(X[votes], dtype=np.float32)),
                [float(v) for v in y_votes],
                X.shape[1],
                rng=np.random.RandomState(42),
                calibrate_count=CALIBRATE_COUNT,
                calibration_fraction=CALIBRATION_FRACTION,
                hidden_dim=hidden_dim,
            )
            comp = (
                _composition_stats(orderings, test_scores, test_truth, y_votes)
                if fallback is None
                else {"vote_pos_frac": float(np.mean(y_votes == 1))}
            )

            for design in DESIGNS:
                for k in INCLUSIONS:
                    if design == "oracle":
                        threshold = conformal_threshold(test_scores.tolist(), test_truth.astype(np.float64).tolist(), k)
                    elif fallback is not None:
                        threshold = fallback
                    else:
                        threshold = threshold_from_fold_orderings(orderings, k)
                        if design == "blend":
                            threshold = calculate_safe_threshold(threshold, test_scores.tolist(), len(votes))
                    m = knobs.pool_metrics(test_scores, test_truth, threshold)
                    alpha = min(1.0, CONFORMAL_BASE_BUDGET * 2.0**-k)
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "n_votes": n_votes,
                            "policy": policy,
                            "design": design,
                            "inclusion": k,
                            "threshold": threshold,
                            "alpha_cap": alpha,
                            "fnr_excess": max(0.0, m["fnr"] - alpha) if np.isfinite(m["fnr"]) else float("nan"),
                            "seed_mode": "text" if seed_scores is not None else "known_good",
                            **m,
                            **comp,
                        }
                    )
            common.log(
                f"[{ci + 1}/{len(cells)}] {arm} seed={seed} {policy} n={n_votes}: "
                f"pos_frac={comp['vote_pos_frac']:.2f} "
                f"cal_q25={comp.get('cal_pos_q25', float('nan')):.3f} "
                f"test_q25={comp.get('test_pos_q25', float('nan')):.3f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    common.log(f"wrote {args.out}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
