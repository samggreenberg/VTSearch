#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_floor_candidate_4267.py``.

Checks what the construction of the schedule and of planted rank frames fixes:

* the schedule's (K, R, m) at the five presets is (128, 3, 5), (64, 2, 5),
  (32, 1, 5), (32, 1, 11) and (32, 1, 29), and at R = 1 it is the ruled
  m(X) = max(5, ceil(ln alpha / ln X));
* m is minimal: m all-positive audits reach X at level alpha / R, and when
  m > 5, m - 1 do not;
* **validity:** on frames whose every top k sits just below X, the schedule
  promises (and so breaks) at most alpha of the time at every floor, within
  Monte Carlo error, while the same audit at level 0.5 breaks far more;
* an all-positive top 128 is promised whole at 10% on every draw, for 5 votes;
* a top 32 that is all positive behind a top 128 at 25% is always promised at
  10%, never broken, and at one of 128, 64 or 32;
* **the likely range (do your best):** a census's range is its exact precision;
  on a planted top 32 at 50% precision, the one-round range covers the truth at
  least 1 - 2 alpha of the time; a check confirms X iff its range's lower end
  clears X (``best_attempt_rows`` asserts it); and a short check still returns a
  set, the top 32, so the line is never empty;
* the whole pipeline runs on planted files and is deterministic under its seed.

    python selftest_analyze_floor_candidate_4267.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import analyze_floor_candidate_4267 as F
import analyze_random_verification as A

N_CORPUS = 2000


def frame_row(ranks, i: int) -> dict:
    ranks = sorted(int(r) for r in ranks)
    return {
        "world": "0.44%",
        "category": f"cls{i % 7}@{('large', 'medium', 'small')[i % 3]}",
        "band": ("large", "medium", "small")[i % 3],
        "seed": i // 7,
        "t": (25, 50, 100, 150)[i % 4],
        "n_corpus": N_CORPUS,
        "n_pos": len(ranks),
        "pos_ranks": " ".join(map(str, ranks)),
    }


def frames_of(rows: list[dict]) -> A.Frames:
    return A.parse_frames(pd.DataFrame(rows), "0.44%")


def spaced(density: float, upto: int) -> list[int]:
    """Positives at a steady density: every top k holds floor(density * k)."""
    return [r for r in range(upto) if np.floor(density * (r + 1)) > np.floor(density * r)]


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        failures.append(what)


def test_schedule(failures: list[str]) -> None:
    want = {0.10: (128, 3, 5), 0.25: (64, 2, 5), 0.50: (32, 1, 5), 0.75: (32, 1, 11), 0.90: (32, 1, 29)}
    got = {x: F.schedule_for(x) for x in want}
    check(got == want, f"the presets' schedule is {want} (got {got})", failures)
    ruled = all(F.schedule_for(x)[2] == max(5, A.min_all_positive(x, F.ALPHA)) for x in (0.5, 0.6, 0.75, 0.9, 0.95))
    check(ruled, "at X >= 50% the schedule is the ruled one-round check, m(X) = max(5, ceil(ln a / ln X))", failures)
    bad = []
    for x in np.round(np.arange(0.02, 0.99, 0.01), 2):
        k, rounds, m = F.schedule_for(float(x))
        level = F.ALPHA / rounds
        if A.lower_bound(np.array(m), np.array(m), level) < x - A.EPS:
            bad.append((float(x), "m too small"))
        if m > 5 and A.lower_bound(np.array(m - 1), np.array(m - 1), level) >= x - A.EPS:
            bad.append((float(x), "m not minimal"))
        if k >> (rounds - 1) != F.BASE:
            bad.append((float(x), "does not end at 32"))
    check(
        not bad,
        f"m is the fewest all-positive picks at every X in 2..98%, and every schedule ends at 32 {bad}",
        failures,
    )


def test_validity(failures: list[str]) -> None:
    draws = 200
    for x in (0.10, 0.25, 0.50):
        density = x - 0.01
        fr = frames_of([frame_row(spaced(density, 1200), i) for i in range(40)])
        worst = max(float(fr.hits(np.full(fr.n, k))[0] / k) for k in range(1, 1200))
        rule, m = F.schedule_rule(x)
        out = A.simulate(fr, rule, m, x, F.ALPHA, draws, seed=7)
        rate = float((out.k > 0).mean())  # every promise on these frames is broken
        se = np.sqrt(F.ALPHA * (1 - F.ALPHA) / (fr.n * draws))
        check(
            worst < x and rate <= F.ALPHA + 3 * se,
            f"X = {x:.0%}: frames below X at every k (best {worst:.3f}); the schedule breaks {rate:.4f} <= alpha",
            failures,
        )
        k0 = np.full(fr.n, rule.start)
        loose = A.simulate_rounds(fr, k0, rule.rounds, m, x, 0.5 * rule.rounds, np.random.default_rng(3), draws)
        check(
            float((loose.k > 0).mean()) > 3 * F.ALPHA,
            f"X = {x:.0%}: the same audit at level 0.5 breaks {float((loose.k > 0).mean()):.2f}: the check has teeth",
            failures,
        )


def test_planted_promises(failures: list[str]) -> None:
    rule, m = F.schedule_rule(0.10)
    whole = frames_of([frame_row(list(range(128)) + [1500, 1600], i) for i in range(20)])
    out = A.simulate(whole, rule, m, 0.10, F.ALPHA, 50, seed=11)
    check(
        bool((out.k == 128).all()) and bool((out.votes == 5).all()),
        "an all-positive top 128 is promised whole at 10% on every draw, for 5 votes",
        failures,
    )
    behind = frames_of([frame_row(list(range(32)) + [1500, 1600], i) for i in range(20)])
    out = A.simulate(behind, rule, m, 0.10, F.ALPHA, 200, seed=12)
    ks = set(np.unique(out.k).tolist())
    hits = behind.hits(out.k)
    never_broken = bool((hits >= 0.10 * out.k - A.EPS).all())
    check(
        ks <= {32, 64, 128} and bool((out.k > 0).all()) and never_broken,
        f"a top 32 behind a top 128 at 25% is always promised at 10%, never broken, at {sorted(ks)}",
        failures,
    )


def test_range(failures: list[str]) -> None:
    lo, hi = F.likely_range(np.array([7, 3]), np.array([16, 5]), np.array([16, 32]), 0.05)
    check(
        lo[0] == hi[0] == 7 / 16 and 0 < lo[1] < 3 / 5 < hi[1] < 1,
        f"a census's range is its exact precision (7/16), a sample's straddles its hit rate: {lo}, {hi}",
        failures,
    )
    half = frames_of([frame_row(list(range(0, 32, 2)) + [1500], i) for i in range(40)])
    rule, m = F.schedule_rule(0.5)
    out = A.simulate(half, rule, m, 0.5, F.ALPHA, 300, seed=13)
    k, n, s_, _ = F.final_round(out)
    lo, hi = F.likely_range(s_, n, k, F.range_tail(1))
    truth = half.hits(k) / k
    cover = float(((lo <= truth + A.EPS) & (truth <= hi + A.EPS)).mean())
    se = np.sqrt(2 * F.ALPHA * (1 - 2 * F.ALPHA) / out.k.size)
    check(
        cover >= 1 - 2 * F.ALPHA - 3 * se,
        f"on a top 32 at 50%, the one-round range covers the truth {cover:.3f} >= {1 - 2 * F.ALPHA:.2f}",
        failures,
    )
    rng = np.random.default_rng(9)
    mixed = frames_of([frame_row(rng.choice(300, size=int(rng.integers(3, 60)), replace=False), i) for i in range(60)])
    rows = []
    try:
        for x in F.FLOORS:
            rows += F.best_attempt_rows(mixed, x, draws=20, seed=F.SEED)
        agree = True
    except AssertionError:
        agree = False
    returned = min(r["returned"] for r in rows) if rows else 0
    check(
        agree and returned >= 32 - A.EPS,
        f"on 60 mixed frames at every floor a check confirms X iff its range clears X, and returns >= 32 ({returned})",
        failures,
    )


def test_pipeline(failures: list[str]) -> None:
    rng = np.random.default_rng(5)
    rows = []
    for i in range(30):
        top = rng.choice(200, size=int(rng.integers(5, 40)), replace=False)
        rows.append(frame_row(top, i))
    frames = {"0.44%": frames_of(rows)}
    a = F.run(frames, draws=4, seed=F.SEED, grid=False)
    b = F.run(frames, draws=4, seed=F.SEED, grid=False)
    cols = ["promised", "broken_of_made", "recall_share", "votes_mean", "returned"]
    same = a[cols].fillna(-1).equals(b[cols].fillna(-1))
    sched = a[(a.slice == "all") & (a.part == "schedule") & (a.rule != "read:32")]
    check(
        same and len(sched) == len(F.FLOORS) and sched.rule.tolist()[0] == "schedule:128>32",
        "the pipeline runs on planted frames, one schedule row per floor, deterministically",
        failures,
    )


def main() -> int:
    failures: list[str] = []
    test_schedule(failures)
    test_validity(failures)
    test_planted_promises(failures)
    test_range(failures)
    test_pipeline(failures)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
