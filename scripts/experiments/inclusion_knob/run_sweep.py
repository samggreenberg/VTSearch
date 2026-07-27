"""Stage 1: the inclusion-knob sweep.

Grid: arms (4 AG News one-vs-rest categories x real E5 embeddings, plus 3
synthetic separability levels) x seeds x vote counts x treatments (raw /
label-smoothed) -> for each cell train the production-shaped final model and
calibration folds once, then evaluate every knob design at every inclusion
value.  Emits one CSV row per (cell, design, inclusion).

Production-faithful fixed choices: ``hidden_dim = _auto_hidden_dim(n_votes)``
for both fold and final models (as ``vtscore/detectors/training.py`` does),
``calibration_fraction = 0.5``, safe-thresholds off (the issue is about the
cross-calibration path).  ``calibrate_count = 2`` matches the MLP-vs-SVM
study's pre-registered value.

Usage::

    python run_sweep.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools

import common

common.setup_env()

import numpy as np  # noqa: E402

SEEDS = range(4)
N_VOTES = (12, 24, 50, 100)
TREATMENTS = {"raw": 0.0, "smooth": 0.05}
DESIGNS = ("argmin", "bayes", "bayes_temp", "conformal")
INCLUSIONS = (-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10)
CALIBRATE_COUNT = 2
CALIBRATION_FRACTION = 0.5
VOTE_POS_FRACTION = 1 / 3


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
        arms[f"synth:{level}"] = level  # generated per-seed in the loop
    return arms


def _sample_votes(y: np.ndarray, n_votes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random vote sample: ~1/3 positive, rest negative."""
    rng = np.random.default_rng(seed)
    n_pos = max(2, round(n_votes * VOTE_POS_FRACTION))
    n_neg = n_votes - n_pos
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    votes = np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ]
    )
    pool = np.setdiff1d(np.arange(len(y)), votes)
    return votes, pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the inclusion-knob sweep.")
    parser.add_argument("--quick", action="store_true", help="1 seed, 2 vote counts (smoke test)")
    parser.add_argument("--out", default=str(common.RESULTS / "sweep.csv"))
    args = parser.parse_args(argv)

    import pandas as pd
    import torch

    import knobs
    import synthetic
    from vtscore.embedding.loader import ensure_torch_configured
    from vtscore.training.mlp import _auto_hidden_dim
    from vtscore.utils.scores import sigmoid_to_finite_array

    ensure_torch_configured()

    seeds = [0] if args.quick else list(SEEDS)
    vote_counts = (12, 50) if args.quick else N_VOTES

    arms = _load_arms()
    rows: list[dict] = []
    cells = list(itertools.product(sorted(arms), seeds, vote_counts, TREATMENTS.items()))
    for i, (arm, seed, n_votes, (treatment, eps)) in enumerate(cells):
        spec = arms[arm]
        if isinstance(spec, str):
            X, y = synthetic.make_synthetic(spec, seed)
        else:
            X, y = spec
        vote_idx, pool_idx = _sample_votes(y, n_votes, seed)
        hidden_dim = _auto_hidden_dim(n_votes)

        model = knobs.train_final_model(X[vote_idx], y[vote_idx].astype(np.float64), hidden_dim, eps)
        with torch.no_grad():
            X_pool = torch.from_numpy(np.ascontiguousarray(X[pool_idx], dtype=np.float32))
            X_pool = X_pool.to(next(model.parameters()).device)
            pool_scores = sigmoid_to_finite_array(model(X_pool)).astype(np.float64)
        pool_truth = y[pool_idx].astype(np.int8)

        orderings, fallback = knobs.fold_orderings_for_treatment(
            X[vote_idx],
            y[vote_idx].astype(np.float64),
            hidden_dim,
            eps,
            CALIBRATE_COUNT,
            CALIBRATION_FRACTION,
        )
        temperature = knobs.fit_temperature(orderings) if fallback is None else 1.0
        sat = knobs.saturation_stats(pool_scores)
        cal_sat = knobs.saturation_stats(np.array([s for ss, _ in orderings for s in ss])) if orderings else sat

        for design in DESIGNS:
            for k in INCLUSIONS:
                if fallback is not None:
                    threshold = fallback
                else:
                    threshold = knobs.knob_threshold(design, orderings, temperature, k)
                m = knobs.pool_metrics(pool_scores, pool_truth, threshold)
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "n_votes": n_votes,
                        "treatment": treatment,
                        "design": design,
                        "inclusion": k,
                        "threshold": threshold,
                        "temperature": temperature,
                        **m,
                        **sat,
                        "cal_mean_abs_logit": cal_sat["sat_mean_abs_logit"],
                    }
                )
        common.log(
            f"[{i + 1}/{len(cells)}] {arm} seed={seed} votes={n_votes} {treatment}: "
            f"T={temperature:.2f} mid-mass={sat['sat_frac_mid']:.3f} extreme={sat['sat_frac_extreme']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    common.log(f"wrote {args.out}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
