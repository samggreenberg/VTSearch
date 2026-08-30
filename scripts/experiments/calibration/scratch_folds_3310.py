"""Laptop-scale fold-count bench for issue #3310: does K>2 help, and where?

Issue #3310 asks whether the old "more calibration folds is WORSE" result
(#2897) can possibly be right.  This bench separates the two things that
result confounded — the *fold count* and the *combine rule* — on a synthetic
2-class Gaussian embedding problem where the oracle cost is measurable, using
the production code paths end to end: liblinear SVM fold fits via
:func:`vtscore.training.thresholds.calibration_folds`, the pooled conformal
rule, the ``tmean`` challenger (#3115), and the shipped fold-anchored rule.

The theory it checks (see ``docs/experiments/2026-08-28-calibration-fold-count-3310/PLAN.md``):

* ``calibrate_count`` draws **independent repeated splits** at a fixed per-fold
  size, so per-fold statistics are i.i.d. draws and raising K adds draws
  without changing any draw's quality.
* Rules that **average** per-fold statistics (``tmean``; the shipped anchored
  rule's ``qmean``) therefore concentrate toward a K-independent target:
  more folds can reduce variance but cannot move the mean.  Expected regret
  should fall (or stay flat) with K, saturating ~1/K, with the gain largest
  where per-fold noise is largest — small labelsets.
* The **pooled** conformal rule is different in kind: its pool is a mixture of
  K fold models' score distributions, and the conformal rule reads extreme
  order statistics of that pool (``gap_mid`` reads ``min(pos)``), so its
  *target moves with K*.  More folds give a better estimate of a
  K-dependently-worse target — which is how "bigger is worse" (#2897, binary)
  happens without any paradox.

Nested-fold slicing (the #2897 trick): Kmax folds are trained once per
replicate and every smaller K reads a prefix, so all K are paired within the
replicate.  The per-K anchored cut is assembled by fitting each fold's
anchored mixture once and concatenating — verified in-session to match
:func:`fold_anchored_gmm_threshold` bit-for-bit.

Run: ``python scripts/experiments/calibration/scratch_folds_3310.py``
(CPU, ~20 min at the default 40 reps; ``--reps 8`` for a smoke pass).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vtscore.training.mlp import LINEAR_SVM_HEAD, train_model  # noqa: E402
from vtscore.training.thresholds import (  # noqa: E402
    calibration_folds,
    combined_fold_conformal_threshold,
    fit_fold_anchored_cut,
    fold_anchored_gmm_threshold,
    threshold_from_fold_orderings,
)
from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: E402

DIM = 64
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
KMAX = max(K_GRID)
N_GRID = [8, 12, 16, 24, 40, 80]  # labelset sizes (votes)
HAYSTACK_N = 3000
TEST_N = 20000
PREVALENCE = 0.10
SEPARATIONS = (1.5, 3.0)  # class-mean separation: overlapping vs clean
RULES = ("pooled", "tmean", "anchored")


def make_env(rng: np.random.Generator, sep: float) -> tuple[np.ndarray, np.ndarray]:
    """Class means separated by *sep* along a random direction; unit covariance."""
    direction = rng.standard_normal(DIM)
    direction /= np.linalg.norm(direction)
    mu_pos = direction * (sep / 2.0)
    return mu_pos, -mu_pos


def draw(rng: np.random.Generator, mu: np.ndarray, n: int) -> np.ndarray:
    return (rng.standard_normal((n, DIM)) + mu).astype(np.float32)


def score(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return np.asarray(
            sigmoid_to_finite_scores(model(torch.tensor(x, dtype=torch.float32))),
            dtype=np.float64,
        )


def cost_at(test_scores: np.ndarray, test_labels: np.ndarray, thr: float) -> float:
    """Inclusion-0 rate cost, FPR + FNR (weights (1, 1))."""
    pos = test_labels == 1.0
    fnr = float(np.mean(test_scores[pos] < thr))
    fpr = float(np.mean(test_scores[~pos] >= thr))
    return fpr + fnr


def oracle_cost(test_scores: np.ndarray, test_labels: np.ndarray) -> float:
    """Minimum rate cost over every cut of *test_scores* (the paired reference)."""
    order = np.argsort(test_scores)
    y = test_labels[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    cum_pos = np.cumsum(y)
    fn = cum_pos / n_pos
    fp = (n_neg - (np.arange(1, len(y) + 1) - cum_pos)) / n_neg
    # A cut below every score has FNR=0, FPR=1 -> cost 1.0.
    return min(float((fn + fp).min()), 1.0)


def one_rep(rng: np.random.Generator, sep: float, n_votes: int) -> list[dict] | None:
    """All (K, rule) regrets for one labelset draw; ``None`` if calibration fell back."""
    mu_pos, mu_neg = make_env(rng, sep)
    n_good = max(2, n_votes // 3)
    n_bad = max(2, n_votes - n_good)
    x_list = list(draw(rng, mu_pos, n_good)) + list(draw(rng, mu_neg, n_bad))
    y_list = [1.0] * n_good + [0.0] * n_bad

    n_hay_pos = int(HAYSTACK_N * PREVALENCE)
    haystack = np.concatenate([draw(rng, mu_pos, n_hay_pos), draw(rng, mu_neg, HAYSTACK_N - n_hay_pos)])
    n_test_pos = int(TEST_N * PREVALENCE)
    test_x = np.concatenate([draw(rng, mu_pos, n_test_pos), draw(rng, mu_neg, TEST_N - n_test_pos)])
    test_labels = np.concatenate([np.ones(n_test_pos), np.zeros(TEST_N - n_test_pos)])

    # Final model on all votes (production head) — the scale every cut is applied on.
    x_train = torch.tensor(np.array(x_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    final_model = train_model(x_train, y_train, DIM, hidden_dim=LINEAR_SVM_HEAD)
    final_hay = score(final_model, haystack)
    test_scores = score(final_model, test_x)
    o_cost = oracle_cost(test_scores, test_labels)

    t0 = time.monotonic()
    folds = calibration_folds(
        x_list, y_list, DIM, calibrate_count=KMAX, calibration_fraction=0.5, hidden_dim=LINEAR_SVM_HEAD
    )
    fold_wall = time.monotonic() - t0
    if folds.fallback is not None:
        return None
    fold_hays = [score(m, haystack) for m in folds.models]

    # Fit each fold's anchored mixture ONCE (folds are independent), then build
    # each K's cut by slicing — matches fold_anchored_gmm_threshold exactly.
    per_fold_cuts = [fit_fold_anchored_cut([fold_hays[i]], [folds.orderings[i]], final_hay) for i in range(KMAX)]

    rows = []
    for k in K_GRID:
        prefix = folds.orderings[:k]
        thr_pooled = threshold_from_fold_orderings(prefix, 0)
        thr_tmean, _prov = combined_fold_conformal_threshold(prefix, 0, combine="tmean")
        contributing = [c for c in per_fold_cuts[:k] if c is not None]
        if contributing:
            combined = dataclasses.replace(
                contributing[0],
                fits=tuple(f for c in contributing for f in c.fits),
                fold_haystacks=tuple(h for c in contributing for h in c.fold_haystacks),
                n_anchored=sum(c.n_anchored for c in contributing),
            )
            thr_anch = combined.threshold_at(0)
        else:
            thr_anch, _prov = fold_anchored_gmm_threshold(fold_hays[:k], prefix, final_hay, 0)
        rows.append(
            {
                "k": k,
                "pooled": cost_at(test_scores, test_labels, thr_pooled) - o_cost,
                "tmean": cost_at(test_scores, test_labels, thr_tmean) - o_cost,
                "anchored": cost_at(test_scores, test_labels, thr_anch) - o_cost,
                "thr_pooled": thr_pooled,
                "thr_anch": thr_anch,
                "fold_seconds_per": fold_wall / KMAX,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=40, help="replicate labelset draws per (separation, n)")
    args = parser.parse_args()

    results: dict[tuple, list[float]] = {}
    t_start = time.monotonic()
    for sep in SEPARATIONS:
        for n in N_GRID:
            for rep in range(args.reps):
                rng = np.random.default_rng(1000 * n + rep + int(sep * 7))
                rows = one_rep(rng, sep, n)
                if rows is None:
                    continue
                for row in rows:
                    for rule in RULES:
                        results.setdefault((sep, n, row["k"], rule), []).append(row[rule])
                    results.setdefault((sep, n, row["k"], "_thr_pooled"), []).append(row["thr_pooled"])
                    results.setdefault((sep, n, row["k"], "_thr_anch"), []).append(row["thr_anch"])
                    results.setdefault(("timing",), []).append(row["fold_seconds_per"])
            print(f"sep={sep} n={n} done ({time.monotonic() - t_start:.0f}s)", flush=True)

    print("\n=== mean regret (cost - oracle) by K; paired delta vs K=2 in brackets ===")
    for sep in SEPARATIONS:
        print(f"\n--- separation {sep} ({'overlapping' if sep < 2 else 'clean'}) ---")
        for rule in RULES:
            print(f"  rule={rule}")
            for n in N_GRID:
                base = np.array(results.get((sep, n, 2, rule), []))
                cells = []
                for k in K_GRID:
                    v = np.array(results.get((sep, n, k, rule), []))
                    if len(v) == 0:
                        cells.append("      -      ")
                        continue
                    d = float(np.mean(v - base)) if len(v) == len(base) else float("nan")
                    cells.append(f"{np.mean(v):+.4f}[{d:+.4f}]")
                print(f"    n={n:3d}: " + "  ".join(f"K{k}:{c}" for k, c in zip(K_GRID, cells, strict=True)))

    print("\n=== mean threshold by K (drift diagnostic) ===")
    for label, key in (("pooled", "_thr_pooled"), ("anchored", "_thr_anch")):
        print(f"  rule={label}, sep=3.0")
        for n in N_GRID:
            vals = [float(np.mean(results[(3.0, n, k, key)])) for k in K_GRID]
            print(f"    n={n:3d}: " + "  ".join(f"K{k}:{v:.3f}" for k, v in zip(K_GRID, vals, strict=True)))

    timing = np.array(results.get(("timing",), [0.0]))
    print(f"\n=== per-fold fit wall clock: mean {timing.mean() * 1000:.1f} ms (total cost linear in K) ===")


if __name__ == "__main__":
    main()
