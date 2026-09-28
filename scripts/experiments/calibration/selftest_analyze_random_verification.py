#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_random_verification.py``.

Builds rank frames whose top-k precision is known by construction and checks the
analyzer recovers what the construction guarantees:

* the oracle cut (largest top k with precision >= X) and its recall match a
  brute-force scan of every k, on random frames and at every floor;
* **validity:** on frames whose every top k sits just below X (so every promise
  is broken), each audited rule promises - and so breaks - at most alpha of the
  time, within Monte Carlo error; a deliberately loose rule (the same audit at
  level 0.5) breaks far more often, so the check has teeth;
* on those same frames every promise made is broken (broken of made = 1), while
  a mix with good frames brings it under alpha: the two rates differ;
* a top 32 that is all positive is promised on every draw, at the exact recall,
  oracle, votes and unaudited recall the construction fixes;
* a census (k0 <= m) is exact: a top 16 holding 8 positives is always promised at
  X = 0.5, and one holding 7 never is;
* shrinking from 128 finds the all-positive top 32 behind a top 128 at 25%;
* the stratified rule promises exactly the all-positive top 16 and books which
  audits fall inside it;
* the stored baselines are read at their exact precision;
* the whole pipeline runs on planted files and is deterministic under its seed.

    python selftest_analyze_random_verification.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_random_verification as A

N_CORPUS = 2000


def frame_row(ranks, i: int, n_corpus: int = N_CORPUS, cons: int = 0, ship: int = 0) -> dict:
    ranks = sorted(int(r) for r in ranks)
    row = {
        "world": "0.44%",
        "category": f"cls{i % 7}@{('large', 'medium', 'small')[i % 3]}",
        "band": ("large", "medium", "small")[i % 3],
        "seed": i // 7,
        "t": (25, 50, 100, 150)[i % 4],
        "n_corpus": n_corpus,
        "n_pos": len(ranks),
        "pos_ranks": " ".join(map(str, ranks)),
        "n_votes": 100,
        "n_vote_pos": 3,
        "n_cal_pos": 2,
    }
    for x in A.FLOORS:
        tag = f"x{round(x * 100)}"
        for name, k in (("shipped", ship), ("consistent", cons)):
            row[f"{name}_status_{tag}"] = "promised" if k else "insufficient_evidence"
            row[f"{name}_k_{tag}"] = k
    return row


def frames_of(rows: list[dict]) -> A.Frames:
    return A.parse_frames(pd.DataFrame(rows), "0.44%")


def spaced(density: float, upto: int) -> list[int]:
    """Positives at a steady density: every top k holds floor(density * k)."""
    return [r for r in range(upto) if np.floor(density * (r + 1)) > np.floor(density * r)]


def brute_oracle(ranks: list[int], n_pos: int, x: float, n_corpus: int) -> tuple[int, float]:
    y = np.zeros(n_corpus, dtype=int)
    y[ranks] = 1
    prec = np.cumsum(y) / np.arange(1, n_corpus + 1)
    ok = np.flatnonzero(prec >= x - 1e-9)
    if not len(ok):
        return 0, 0.0
    k = int(ok.max()) + 1
    return k, float(y[:k].sum() / n_pos)


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        failures.append(what)


def rule(name: str) -> A.Rule:
    return next(r for r in A.RULES if r.name == name)


def test_oracle(failures: list[str]) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        n_pos = int(rng.integers(1, 40))
        # Mostly near the top, some deep: the shape a detector's ranking has.
        top = rng.choice(80, size=min(n_pos, 80), replace=False)
        deep = rng.choice(np.arange(80, N_CORPUS), size=n_pos - len(top), replace=False)
        rows.append(frame_row(np.r_[top, deep], i))
    fr = frames_of(rows)
    bad = 0
    for x in A.FLOORS:
        k = A.oracle_k(fr, x)
        rec = A.recall_of(fr, k)
        for i, row in enumerate(rows):
            ranks = list(map(int, row["pos_ranks"].split()))
            bk, brec = brute_oracle(ranks, len(ranks), x, N_CORPUS)
            # The oracle cut may stop short of trailing negatives; recall is what must agree.
            bad += (abs(rec[i] - brec) > 1e-12) or (k[i] > bk) or (fr.hits(k)[i] != fr.hits(np.array([bk] * fr.n))[i])
    check(bad == 0, "the oracle cut and its recall match a brute-force scan (60 frames x 3 floors)", failures)


def test_validity(failures: list[str]) -> None:
    x, alpha, draws = 0.5, 0.05, 200
    below = [frame_row(spaced(0.49, 400), i, cons=128) for i in range(40)]
    fr = frames_of(below)
    worst = max(float(fr.hits(np.full(fr.n, k))[0] / k) for k in range(1, 400))
    check(worst < x, f"planted frames sit below X at every k (best top-k precision {worst:.3f})", failures)
    n = fr.n * draws
    for r in A.RULES:
        out = A.simulate(fr, r, 20, x, alpha, draws, seed=7)
        rate = float((out.k > 0).mean())  # every promise here is a broken one
        se = np.sqrt(alpha * (1 - alpha) / n)
        check(rate <= alpha + 3 * se, f"{r.name}: breaks {rate:.4f} of frames <= alpha={alpha}", failures)
    loose = A.simulate_rounds(fr, np.full(fr.n, 32), 1, 20, x, 0.5, np.random.default_rng(3), draws)
    check(
        float((loose.k > 0).mean()) > 3 * alpha,
        f"a loose audit (level 0.5) breaks {float((loose.k > 0).mean()):.2f} of frames: the check has teeth",
        failures,
    )

    good = [frame_row(list(range(32)) + list(range(1000, 1008)), 100 + i, cons=32) for i in range(40)]
    mix = frames_of(below + good)
    rows = A.summarise(
        mix,
        A.simulate(mix, rule("a:top32"), 10, x, alpha, draws, seed=11),
        x,
        A.recall_of(mix, A.oracle_k(mix, x)),
        np.zeros(mix.n),
    )
    alone = A.summarise(
        fr, A.simulate(fr, rule("a:top32"), 10, x, alpha, draws, seed=11), x, np.zeros(fr.n), np.zeros(fr.n)
    )
    made_alone = alone[0]["broken_of_made"]
    check(
        np.isnan(made_alone) or made_alone == 1.0,
        "on below-X frames alone, every promise made is broken (broken of made = 1)",
        failures,
    )
    check(
        rows[0]["broken_of_made"] <= alpha,
        f"mixed with good frames, broken of made falls to {rows[0]['broken_of_made']:.4f} <= alpha",
        failures,
    )


def test_known_good(failures: list[str]) -> None:
    # Top 32 all positive, 8 more deep: precision 1 to k=32, exactly 0.5 at k=64.
    rows = [frame_row(list(range(32)) + list(range(1000, 1008)), i) for i in range(10)]
    fr = frames_of(rows)
    out = A.simulate(fr, rule("a:top32"), 10, 0.5, 0.05, 20, seed=5)
    check(bool((out.k == 32).all()), "an all-positive top 32 is promised on every draw", failures)
    s = A.summarise(fr, out, 0.5, A.recall_of(fr, A.oracle_k(fr, 0.5)), np.zeros(fr.n))[0]
    check(abs(s["recall"] - 0.8) < 1e-12, f"its recall is exactly 32/40 (got {s['recall']})", failures)
    check(abs(s["oracle"] - 0.8) < 1e-12, "the oracle (k = 64 at exactly 0.5) recalls 32/40", failures)
    check(abs(s["recall_share"] - 1.0) < 1e-12, "recall / oracle is 1", failures)
    check(s["votes_mean"] == 10 and s["broken_of_all"] == 0, "it costs m = 10 votes and never breaks", failures)
    check(abs(s["unaudited_recall"] - 22 / 40) < 1e-12, "the unaudited recall is (32 - 10) / 40", failures)
    check(s["census_of_made"] == 0, "a 10-item audit of 32 is not a census", failures)


def test_census(failures: list[str]) -> None:
    eight = frames_of([frame_row(list(range(0, 16, 2)) + [500], i) for i in range(10)])
    seven = frames_of([frame_row(list(range(0, 14, 2)) + [500, 501], i) for i in range(10)])
    a = A.simulate(eight, rule("a:top16"), 20, 0.5, 0.05, 20, seed=1)
    b = A.simulate(seven, rule("a:top16"), 20, 0.5, 0.05, 20, seed=1)
    check(
        bool((a.k == 16).all()) and bool((a.votes == 16).all()),
        "census: 8/16 is promised at X = 0.5, for 16 votes",
        failures,
    )
    check(bool((b.k == 0).all()), "census: 7/16 is never promised at X = 0.5", failures)


def test_shrinking(failures: list[str]) -> None:
    # 128 at 25%, 64 at exactly 50%, 32 at 100%.
    fr = frames_of([frame_row(list(range(32)) + list(range(1000, 1010)), i) for i in range(30)])
    out = A.simulate(fr, rule("b:top128"), 20, 0.5, 0.05, 50, seed=9)
    k = out.k
    check(
        float((k == 32).mean()) >= 0.9,
        f"shrinking lands on the all-positive top 32 ({float((k == 32).mean()):.2f})",
        failures,
    )
    check(not bool((k == 128).any()), "shrinking never promises the 25% top 128", failures)
    check(bool((out.votes <= 20 * 6).all()), "it spends at most R x m votes", failures)
    # Votes: 20 on top 128, then top 64 (keeping ~10 known), then a census of 32.
    check(35 <= float(out.votes.mean()) <= 65, f"and about 40-60 votes on the way ({out.votes.mean():.1f})", failures)


def test_stratified(failures: list[str]) -> None:
    # Top 16 all positive, then none until 1000. At m = 40 the bins above 16 are
    # censused (4 + 4 + 8), so the top 16 is exact and the top 32 is exactly 50%.
    fr = frames_of([frame_row(list(range(16)) + list(range(1000, 1004)), i) for i in range(10)])
    for name in ("c:union", "c:seq"):
        for x, want_k, want_seen in ((0.6, 16, 16), (0.5, 32, 24)):
            out = A.simulate(fr, rule(name), 40, x, 0.05, 20, seed=2)
            check(bool((out.k == want_k).all()), f"{name}: at X = {x:g} promises exactly the top {want_k}", failures)
            check(
                bool((out.seen == want_seen).all()) and bool((out.seen_pos == 16).all()),
                f"{name}: books {want_seen} audits (16 positive) inside it",
                failures,
            )
        check(bool((out.votes == 40).all()), f"{name}: spends its m = 40", failures)
    check(A.allocate(10, [4, 4, 8, 16, 32, 64]) == [2, 2, 2, 2, 1, 1], "round-robin allocation from the top", failures)
    check(A.allocate(40, [4, 4, 8, 16, 32, 64]) == [4, 4, 8, 8, 8, 8], "allocation is capped at a bin's size", failures)


def test_baselines(failures: list[str]) -> None:
    # Top 10 positive: consistent returns 40 (25%), shipped returns 16 (62.5%).
    fr = frames_of([frame_row(list(range(10)) + [900], i, cons=40, ship=16) for i in range(6)])
    orec = A.recall_of(fr, A.oracle_k(fr, 0.5))
    for name, want_broken in (("consistent", 1.0), ("shipped", 0.0)):
        k = A.baseline_k(fr, name, 0.5)[:, None]
        zero = np.zeros_like(k)
        s = A.summarise(fr, A.Outcome(k, zero, zero, zero, []), 0.5, orec, np.zeros(fr.n))[0]
        check(s["broken_of_made"] == want_broken, f"the stored {name} cut reads broken = {want_broken:g}", failures)


def test_pipeline(failures: list[str]) -> None:
    rng = np.random.default_rng(4)
    rows = []
    for i in range(24):
        n_top = int(rng.integers(0, 40))
        ranks = sorted(set(rng.choice(64, size=n_top, replace=False).tolist()) | {1500 + i})
        rows.append(frame_row(ranks, i, cons=int(rng.choice([0, 48])), ship=int(rng.choice([0, 8]))))
    rows.append({**frame_row(list(range(30)) + [1999], 999, cons=60), "category": "bird@large", "seed": 0, "t": 150})
    root = Path(tempfile.mkdtemp(prefix="selftest-rv-4257-"))
    pd.DataFrame(rows).to_csv(root / A.WORLDS["0.44%"], index=False)
    outs = []
    for run in ("one", "two"):
        out = root / run
        rc = A.main(["--frames", str(root), "--out", str(out), "--draws", "4", "--no-figures"])
        check(rc == 0, f"the analyzer runs end to end on planted files ({run})", failures)
        outs.append(out)
    for f in (
        "summary.csv",
        "summary_by_t.csv",
        "summary_by_band.csv",
        "examples.csv",
        "min_audit.csv",
        "provenance.json",
    ):
        check((outs[0] / f).exists(), f"it writes {f}", failures)
    a, b = (pd.read_csv(o / "summary.csv") for o in outs)
    check(a.equals(b), "two runs under the same seed agree exactly", failures)
    ex = pd.read_csv(outs[0] / "examples.csv")
    check("bird@large" in set(ex["cell"]), "a named cell reaches the literal examples", failures)
    m = pd.read_csv(outs[0] / "min_audit.csv")
    got = m.set_index(["X", "alpha"])["min_all_positive_audits"]
    check(
        got[(0.5, 0.05)] == 5 and got[(0.75, 0.05)] == 11 and got[(0.75, 0.1)] == 9,
        "minimum all-positive audits: 5, 11, 9",
        failures,
    )


def main() -> int:
    failures: list[str] = []
    test_oracle(failures)
    test_validity(failures)
    test_known_good(failures)
    test_census(failures)
    test_shrinking(failures)
    test_stratified(failures)
    test_baselines(failures)
    test_pipeline(failures)
    print()
    print("SELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
