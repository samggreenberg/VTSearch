"""Adversarial bound: the conformal budget under purely exploitative labeling.

**Superseded for production estimates by :mod:`run_autopilot_sweep`.**  This
stage's ``toplist`` policy is *not* VTSearch's workflow - it greedily votes the
top of the sort for every vote past the seed phase, whereas real Autopilot votes
the top only for its first few Goods, takes its Bads from the *bottom*, and then
alternates Hard (nearest the threshold) and New (atlas diversity).  Margin
sampling biases the calibration positives in the opposite direction, so this arm
overstates - and mis-signs - the production effect.  It is retained because it
is a fair model of a user manually reviewing a learned-sort result list
top-down, and a useful worst case.  See
``docs/experiments/2026-07-27-inclusion-knob/SELECTION-BIAS.md``.

Grid: arms (AG News categories if cached, plus 3 synthetic separability
levels) x seeds x vote counts x **vote-selection policies**:

* ``uniform`` - stratified random votes (the exchangeable baseline, as in
  ``run_sweep.py``).
* ``toplist`` - purely exploitative labeling: a cosine text-query stand-in seeds
  the first votes from the top of its ranking, then each round trains the
  production MLP on the votes so far and labels the top ``TOPLIST_BATCH``
  unvoted items of its sort.

For every cell the *thresholding* path is bit-for-bit production
(``compute_fold_orderings`` + ``threshold_from_fold_orderings``); the only
manipulated variable is which items got labeled.  Two references bracket each
conformal threshold:

* ``oracle`` - the same conformal rule fed the *entire* ground-truth pool as
  calibration: what the rule would do with a perfectly representative,
  effectively infinite calibration set.  conformal-vs-oracle gap = selection
  bias + finite-sample noise; the uniform policy isolates the noise term.
* ``blend`` - production ``calculate_safe_threshold`` (GMM on the full pool
  score distribution, ramped in below 20 labels): the one production input
  immune to labeling bias, measured here as a candidate mitigation.

Emits one CSV row per (cell, design, inclusion) with the confusion metrics,
the ``alpha(k)`` budget, ``fnr_excess = max(0, fnr - alpha)``, and per-cell
composition stats (vote class ratio, calibration-vs-pool positive quantiles).

Usage::

    python run_selection_sweep.py [--quick] [--out CSV]
"""

from __future__ import annotations

import argparse
import itertools

import common

common.setup_env()

import numpy as np  # noqa: E402

SEEDS = range(4)
N_VOTES = (12, 24, 50, 100)
POLICIES = ("uniform", "toplist")
DESIGNS = ("conformal", "blend", "oracle")
INCLUSIONS = (-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10)
CALIBRATE_COUNT = 2
CALIBRATION_FRACTION = 0.5
VOTE_POS_FRACTION = 1 / 3  # uniform arm only; toplist's ratio is emergent
TOPLIST_BATCH = 8  # votes per re-sort round, a "handful then re-sort" session
TOPLIST_QUERY_EXEMPLARS = 3  # positives averaged into the cosine query stand-in


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


def _sample_votes_uniform(y: np.ndarray, n_votes: int, seed: int) -> np.ndarray:
    """Stratified random vote sample: ~1/3 positive, rest negative."""
    rng = np.random.default_rng(seed)
    n_pos = max(2, round(n_votes * VOTE_POS_FRACTION))
    n_neg = n_votes - n_pos
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    return np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ]
    )


def _sample_votes_toplist(X: np.ndarray, y: np.ndarray, n_votes: int, seed: int) -> np.ndarray:
    """Simulate a production labeling session: cosine seed, then top-of-sort rounds.

    Round 0 walks down a cosine ranking (query = normalized mean of a few
    random positive exemplars, the text-query stand-in) labeling items until
    both classes have >= 2 votes - the minimum ``compute_fold_orderings``
    needs - just as a real user keeps voting until the sort is trainable.
    Every later round trains the production MLP on the votes so far and
    labels the ``TOPLIST_BATCH`` highest-scoring unvoted items.
    """
    import torch

    from vtscore.training.mlp import _auto_hidden_dim, train_model
    from vtscore.utils.scores import sigmoid_to_finite_array

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    exemplars = rng.choice(pos_idx, size=TOPLIST_QUERY_EXEMPLARS, replace=False)
    query = X[exemplars].mean(axis=0)
    query /= np.linalg.norm(query)
    cosine_order = np.argsort(-(X @ query))

    votes: list[int] = []
    voted = np.zeros(len(y), dtype=bool)

    def _n_class(label: int) -> int:
        return sum(1 for i in votes if y[i] == label)

    # Seed round: label down the cosine list until trainable (>= 2 per class).
    # A sharp query on an easy task tops the ranking with pure matches; a real
    # user still needs Bad votes before training is possible, so they scroll
    # past surplus matches to find non-matches.  Model that by reserving the
    # last slots for whichever class is still missing: an item is skipped
    # (scrolled past, not voted) when voting it would spend a reserved slot.
    for i in cosine_order:
        if len(votes) >= n_votes or (_n_class(1) >= 2 and _n_class(0) >= 2):
            break
        slots_left = n_votes - len(votes)
        reserved = max(0, 2 - _n_class(0)) if y[i] == 1 else max(0, 2 - _n_class(1))
        if slots_left <= reserved:
            continue
        votes.append(int(i))
        voted[i] = True

    # Online rounds: production model on current votes -> label its top items.
    while len(votes) < n_votes and _n_class(1) >= 2 and _n_class(0) >= 2:
        X_t = torch.from_numpy(np.ascontiguousarray(X[votes], dtype=np.float32))
        y_t = torch.tensor(y[votes], dtype=torch.float32).unsqueeze(1)
        model = train_model(X_t, y_t, X.shape[1], hidden_dim=_auto_hidden_dim(len(votes)))
        with torch.no_grad():
            X_all = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            X_all = X_all.to(next(model.parameters()).device)
            scores = sigmoid_to_finite_array(model(X_all)).astype(np.float64)
        scores[voted] = -np.inf
        batch = np.argsort(-scores)[: min(TOPLIST_BATCH, n_votes - len(votes))]
        for i in batch:
            votes.append(int(i))
            voted[i] = True
    return np.asarray(votes)


def _composition_stats(
    orderings: list[tuple[list[float], list[float]]],
    pool_scores: np.ndarray,
    pool_truth: np.ndarray,
    y_votes: np.ndarray,
) -> dict[str, float]:
    """Per-cell selection-bias diagnostics.

    ``cal_pos_q25`` is the value the k=0 conformal cut is read from;
    ``pool_pos_q25`` is where a perfectly representative calibration set
    would put it.  Their gap is the threshold inflation selection bias buys.
    """
    cal_pos = np.array([s for ss, ll in orderings for s, lb in zip(ss, ll, strict=True) if lb == 1.0])
    pool_pos = pool_scores[pool_truth == 1]
    return {
        "vote_pos_frac": float(np.mean(y_votes == 1)),
        "n_cal_pos": int(len(cal_pos)),
        "cal_pos_q25": float(np.quantile(cal_pos, 0.25)) if len(cal_pos) else float("nan"),
        "pool_pos_q25": float(np.quantile(pool_pos, 0.25)) if len(pool_pos) else float("nan"),
        "cal_pos_min": float(cal_pos.min()) if len(cal_pos) else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the vote-selection-bias sweep.")
    parser.add_argument("--quick", action="store_true", help="1 seed, 2 vote counts (smoke test)")
    parser.add_argument("--out", default=str(common.RESULTS / "selection_sweep.csv"))
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
    vote_counts = (12, 50) if args.quick else N_VOTES

    arms = _load_arms()
    rows: list[dict] = []
    cells = list(itertools.product(sorted(arms), seeds, vote_counts, POLICIES))
    for i, (arm, seed, n_votes, policy) in enumerate(cells):
        spec = arms[arm]
        if isinstance(spec, str):
            X, y = synthetic.make_synthetic(spec, seed)
        else:
            X, y = spec

        if policy == "uniform":
            vote_idx = _sample_votes_uniform(y, n_votes, seed)
        else:
            vote_idx = _sample_votes_toplist(X, y, n_votes, seed)
        pool_idx = np.setdiff1d(np.arange(len(y)), vote_idx)
        y_votes = y[vote_idx].astype(np.float64)
        if min(np.sum(y_votes == 1), np.sum(y_votes == 0)) < 2:
            common.log(
                f"[{i + 1}/{len(cells)}] {arm} seed={seed} votes={n_votes} {policy}: SKIPPED (single-class votes)"
            )
            continue
        hidden_dim = _auto_hidden_dim(len(vote_idx))

        X_t = torch.from_numpy(np.ascontiguousarray(X[vote_idx], dtype=np.float32))
        y_t = torch.tensor(y_votes, dtype=torch.float32).unsqueeze(1)
        model = train_model(X_t, y_t, X.shape[1], hidden_dim=hidden_dim)
        with torch.no_grad():
            X_pool = torch.from_numpy(np.ascontiguousarray(X[pool_idx], dtype=np.float32))
            X_pool = X_pool.to(next(model.parameters()).device)
            pool_scores = sigmoid_to_finite_array(model(X_pool)).astype(np.float64)
        pool_truth = y[pool_idx].astype(np.int8)

        orderings, fallback = compute_fold_orderings(
            list(np.asarray(X[vote_idx], dtype=np.float32)),
            [float(v) for v in y_votes],
            X.shape[1],
            rng=np.random.RandomState(42),
            calibrate_count=CALIBRATE_COUNT,
            calibration_fraction=CALIBRATION_FRACTION,
            hidden_dim=hidden_dim,
        )
        comp = (
            _composition_stats(orderings, pool_scores, pool_truth, y_votes)
            if fallback is None
            else {"vote_pos_frac": float(np.mean(y_votes == 1))}
        )

        for design in DESIGNS:
            for k in INCLUSIONS:
                if design == "oracle":
                    threshold = conformal_threshold(pool_scores.tolist(), pool_truth.astype(np.float64).tolist(), k)
                elif fallback is not None:
                    threshold = fallback
                else:
                    threshold = threshold_from_fold_orderings(orderings, k)
                    if design == "blend":
                        threshold = calculate_safe_threshold(threshold, pool_scores.tolist(), len(vote_idx))
                m = knobs.pool_metrics(pool_scores, pool_truth, threshold)
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
                        **m,
                        **comp,
                    }
                )
        common.log(
            f"[{i + 1}/{len(cells)}] {arm} seed={seed} votes={n_votes} {policy}: "
            f"pos_frac={comp['vote_pos_frac']:.2f} "
            f"cal_q25={comp.get('cal_pos_q25', float('nan')):.3f} pool_q25={comp.get('pool_pos_q25', float('nan')):.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    common.log(f"wrote {args.out}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
