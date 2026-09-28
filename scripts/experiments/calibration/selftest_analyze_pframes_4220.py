#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_pframes_4220.py``.

Builds precision frames from two Gaussians with a known posterior (negatives
N(0,1), positives N(3,1)), with votes picked by score only, and checks the
analyzer recovers what the construction guarantees:

* on a corpus with the voted pool's prevalence, the point estimate's cut
  achieves precision near X (median within 0.1);
* the bootstrap lower bound violates the floor less often than the point
  estimate does;
* when the corpus is poorer than the pool (the 5% arm's ``shifted``
  scenario), the uncorrected estimate over-promises, and EM re-weighting to
  the corpus prior over-promises less;
* votes without a single positive fit nothing and promise nothing;
* the shipped reference cut is read at exactly the precision its threshold gives.

    python selftest_analyze_pframes_4220.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_pframes_4220 as A  # noqa: E402

N_POOL, N_TEST = 6000, 6000
THRESHOLD = 1.5


def _draw(rng, n, prev):
    y = (rng.random(n) < prev).astype(np.uint8)
    s = np.where(y == 1, rng.normal(3.0, 1.0, n), rng.normal(0.0, 1.0, n)).astype(np.float32)
    return s, y


def _votes(rng, s, y, n):
    """Score-picked: mostly the top 3%, the rest anywhere - never by label."""
    top = np.argsort(-s)[: max(int(0.03 * len(s)), n)]
    w = np.full(len(s), 0.3 / len(s))
    w[top] += 0.7 / len(top)
    idx = rng.choice(len(s), size=n, replace=False, p=w / w.sum())
    return s[idx], y[idx]


def build_cell(
    cells: Path, idx: int, seed: int, pool_prev: float, test_prev: float, n_votes: int, no_pos: bool = False
):
    rng = np.random.default_rng(seed)
    pool_s, pool_y = _draw(rng, N_POOL, pool_prev)
    test_s, test_y = _draw(rng, N_TEST, test_prev)
    vs, vy = _votes(rng, pool_s, pool_y, n_votes)
    if no_pos:
        vy = np.zeros_like(vy)
    half = n_votes // 2
    frame = {
        "t": np.int32(150),
        "threshold": np.float32(THRESHOLD),
        "test_scores": test_s,
        "test_labels": test_y,
        "pool_scores": pool_s,
        "vote_scores": vs,
        "vote_labels": vy,
        # Folds on the final model's own scale: fold-rank and fold-raw agree here.
        "fold_cal_scores": vs,
        "fold_cal_labels": vy,
        "fold_cal_fold": np.r_[np.zeros(half), np.ones(n_votes - half)].astype(np.uint8),
        "fold_hay_scores": np.r_[pool_s, pool_s],
        "fold_hay_fold": np.r_[np.zeros(N_POOL), np.ones(N_POOL)].astype(np.uint8),
        "style": np.array("whole_image"),
    }
    np.savez_compressed(cells / f"task_{idx:04d}__pframes.npz", **{f"t150/{k}": v for k, v in frame.items()})
    pd.DataFrame([{"dataset": "planted", "category": f"cls{idx}@large", "seed": seed, "t": 150}]).to_csv(
        cells / f"task_{idx:04d}.csv", index=False
    )


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []
    A.BOOTS = 20
    root = Path(tempfile.mkdtemp(prefix="selftest-pframes-4220-"))
    same = root / "natural" / "cells"
    rich = root / "h0.05" / "cells"
    for d in (same, rich):
        d.mkdir(parents=True)
    for i in range(12):
        build_cell(same, i, seed=i, pool_prev=0.05, test_prev=0.05, n_votes=300)
        build_cell(rich, i, seed=100 + i, pool_prev=0.05, test_prev=0.01, n_votes=300)
    build_cell(same, 99, seed=99, pool_prev=0.05, test_prev=0.05, n_votes=150, no_pos=True)
    out = root / "analysis"
    rc = A.main(["--arm", f"natural={root / 'natural'}", "--arm", f"h0.05={root / 'h0.05'}", "--out", str(out)])
    check(rc == 0, "the analyzer runs on planted frames", failures)
    df = pd.read_csv(out / "estimator_rows.csv.gz")
    refs = pd.read_csv(out / "reference_rows.csv.gz")

    live = df[(df["category"] != "cls99@large")]
    sel = live[(live.arm == "natural") & (live.estimator == "insample") & (live.fit == "logistic") & (live.X == 0.5)]
    point = sel[sel.reading == "point"]
    lcb = sel[sel.reading == "lcb"]
    check(
        abs(point["achieved"].median() - 0.5) < 0.1,
        f"point estimate lands near X=0.5 (median {point['achieved'].median():.2f})",
        failures,
    )
    check(
        lcb["violated"].mean() < point["violated"].mean(),
        "the lower bound violates less than the point estimate",
        failures,
    )

    sh = live[
        (live.arm == "h0.05")
        & (live.scenario == "shifted")
        & (live.estimator == "insample")
        & (live.fit == "logistic")
        & (live.X == 0.5)
    ]
    raw_v = sh[sh.reading == "point"]["violated"].mean()
    em_v = sh[sh.reading == "point+em"]["violated"].mean()
    check(raw_v > 0.5, f"a richer pool than corpus over-promises without correction ({raw_v:.2f})", failures)
    check(em_v < raw_v, f"EM to the corpus prior over-promises less ({em_v:.2f} < {raw_v:.2f})", failures)

    nopos = df[df["category"] == "cls99@large"]
    check(
        len(nopos) > 0 and not nopos["fitted"].any() and nopos["empty"].all(),
        "votes without a positive promise nothing",
        failures,
    )

    shipped = refs[(refs.cut == "shipped") & (refs.arm == "natural") & (refs.category == "cls0@large")].iloc[0]
    z = np.load(same / "task_0000__pframes.npz")
    s, y = z["t150/test_scores"], z["t150/test_labels"]
    want = float(y[s >= THRESHOLD].mean())
    # The row files carry five significant figures, so "exact" is to that.
    check(abs(shipped["achieved"] - want) < 1e-4, "the shipped reference cut reads its exact precision", failures)

    print()
    print("SELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
