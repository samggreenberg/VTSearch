#!/usr/bin/env python3
"""The spot check's candidate at low floors (#4267 follow-up): how big, and how many picks?

The owner's #4267 ruling earns the precision floor's promise with one **spot
check** of the top 32 unvoted items (#4257's ``a:top32``). Its candidate is fixed,
so every floor returns the same 32 items when promised: a lower floor only makes
the promise easier to earn. With a 10% preset added (owner, 2026-09-29, "a more
exhaustive option for users willing to dig through results"), the owner ruled
that **the candidate grows as the floor falls**. This prices how far, from the
#4224 rank frames, reusing ``analyze_random_verification.py``'s exact audit
simulation (hypergeometric draws, one-sided Clopper-Pearson bound, Bonferroni
over rounds).

Rules, at X in ``FLOORS`` (``grid``):

* ``a:topK`` - one round: audit m items uniformly from the top K, promise it or
  nothing. ``a:top32`` at m(X) is the rule as first ruled.
* ``b:topK>32`` - shrinking: audit m fresh items a round from the top K, halve K
  on failure down to 32, each round at alpha / R.

and the **schedule** the owner picked from this grid (``schedule``):

* the candidate starts at K(X) = 32 * 2**max(0, floor(log2(0.5 / X))), so the top
  128 at 10%, the top 64 at 25%, and the top 32 at 50% and above;
* it halves on a failed round down to 32: R = log2(K / 32) + 1 rounds;
* each round audits m(X) = max(5, ceil(ln(alpha / R) / ln X)) fresh items, the
  fewest that can reach X at level alpha / R when all are right. At R = 1 this
  is the ruled m(X): 5 up to X = 54.9%, 11 at 75%, 29 at 90%.

Metrics are ``analyze_random_verification.summarise``'s, plus ``returned``: the
mean size of a promised set (``summary.csv``).

**Do your best** (owner, 2026-09-29, ``best_attempt.csv``). The promise is not
make-or-break: the line always keeps the set the check ended on (the promised set,
or the top 32 after a short check), and the control shows how close it got as a
**likely range**: a Clopper-Pearson interval from the labels inside that set, each
tail at the level the check's rounds are tested at (``range_tail``), and exact for a
census. This measures how often that range contains the
set's true precision (``coverage``, split by a confirmed and a short check and by
the round a check ended in), how close a short check gets, and how good the
unchecked starting candidate is on its own (what a headless run returns).

    python analyze_floor_candidate_4267.py [--frames DIR] [--out DIR] [--draws 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_random_verification as A

OUT_DIR = A.REPO / "docs" / "experiments" / "2026-09-29-floor-candidate-4267"
FLOORS = (0.10, 0.25, 0.50, 0.75, 0.90)
ALPHA = 0.05
SEED = 4267
#: The grid the schedule was chosen from (one round, and shrinking to 32).
ONE_ROUND = ((32, (5, 10, 20, 40)), (64, (5, 10, 20, 40)), (128, (5, 10, 20, 40)))
SHRINKING = ((64, (5, 10, 20)), (128, (5, 10, 20)), (256, (5, 10, 20)), (512, (5, 10, 20)))
BASE = 32


def schedule_for(x: float, alpha: float = ALPHA) -> tuple[int, int, int]:
    """(K, R, m) for floor ``x``: the starting candidate, the rounds, and the picks a round."""
    doublings = max(0, math.floor(math.log2(0.5 / x) + A.EPS))
    k = BASE * 2**doublings
    rounds = doublings + 1
    m = max(5, math.ceil(math.log(alpha / rounds) / math.log(x) - A.EPS))
    return k, rounds, m


def schedule_rule(x: float) -> tuple[A.Rule, int]:
    k, rounds, m = schedule_for(x)
    kind = "a" if rounds == 1 else "b"
    return A.Rule(f"schedule:{k}>{BASE}" if rounds > 1 else f"schedule:{k}", kind, k, rounds=rounds), m


def grid_rules() -> list[tuple[A.Rule, int]]:
    out = [(A.Rule(f"a:top{k}", "a", k), m) for k, ms in ONE_ROUND for m in ms]
    for k, ms in SHRINKING:
        rounds = int(math.log2(k // BASE)) + 1
        out += [(A.Rule(f"b:top{k}>{BASE}", "b", k, rounds=rounds), m) for m in ms]
    return out


def rows_for(fr: A.Frames, x: float, rules: list[tuple[A.Rule, int]], draws: int, seed: int, part: str) -> list[dict]:
    ok = A.oracle_k(fr, x)
    orec = A.recall_of(fr, ok)
    refs = {"read32": A.recall_of(fr, A.census_k(fr, 32, x))}
    masks = dict(A.slices(fr))

    def extra(r: dict, k: np.ndarray) -> dict:
        """Per slice: the mean size of a promised set, and the oracle's median cut."""
        sel = masks[r["slice"]]
        promised = k[sel][k[sel] > 0]
        return {
            "returned": float(promised.mean()) if promised.size else 0.0,
            "oracle_k_median": float(np.median(ok[sel])),
        }

    rows = []
    for rule, m in rules:
        out = A.simulate(fr, rule, m, x, ALPHA, draws, seed)
        for r in A.summarise(fr, out, x, orec, refs):
            head = {"part": part, "world": fr.world, "X": x, "rule": rule.name, "K": int(rule.start)}
            rounds = rule.rounds if rule.kind == "b" else 1
            rows.append({**head, "rounds": rounds, "m": m, **r, **extra(r, out.k)})
    # Reading the top 32 yourself, the reference the report compares against.
    k = A.census_k(fr, 32, x)[:, None]
    ref = A.Outcome(k, np.full_like(k, 32), k, fr.hits(k), [])
    for r in A.summarise(fr, ref, x, orec, refs):
        head = {"part": part, "world": fr.world, "X": x, "rule": "read:32", "K": 32}
        rows.append({**head, "rounds": 0, "m": 32, **r, **extra(r, k)})
    return rows


def range_tail(rounds: int, alpha: float = ALPHA) -> float:
    """Each tail of the "likely range": the level every round of the check is tested at.

    So the range's lower end is exactly the bound the check tested, and a check
    confirms X iff that lower end is >= X. At one round it is a two-sided 90%
    Clopper-Pearson interval. The narrower 90% range at every round count covers
    too rarely after an early pass: at 10%, 0.44%, a check confirmed in its first
    round showed a range above the truth 11% of the time (winner's curse), against
    4.0% with this tail.
    """
    return alpha / rounds


def likely_range(s: np.ndarray, n: np.ndarray, k: np.ndarray, tail: float) -> tuple[np.ndarray, np.ndarray]:
    """The range shown for a set of ``k`` whose ``n`` uniform labels hold ``s`` right; exact for a census."""
    lo = A.lower_bound(s, n, tail)
    hi = 1.0 - A.lower_bound(n - s, n, tail)
    exact = s / np.maximum(k, 1)
    census = n >= k
    return np.where(census, exact, lo), np.where(census, exact, hi)


def final_round(out: A.Outcome) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per frame x draw: the set the check ended on, its labels (n, s), and the round (1-based)."""
    shape = out.k.shape
    k_fin, n_fin, s_fin = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    r_fin = np.zeros(shape, dtype=np.int64)
    for r, step in enumerate(out.trace):
        live = step["live"]
        k_fin = np.where(live, np.broadcast_to(step["k"][:, None], shape), k_fin)
        n_fin = np.where(live, step["n"], n_fin)
        s_fin = np.where(live, step["s"], s_fin)
        r_fin = np.where(live, r + 1, r_fin)
    return k_fin, n_fin, s_fin, r_fin


def best_attempt_rows(fr: A.Frames, x: float, draws: int, seed: int) -> list[dict]:
    """Do your best (owner, 2026-09-29): the line always keeps the set the check ended on, with its range.

    A passed check keeps the promised set; a short one keeps the set its last round
    audited (the top 32 after every halving). Before any check, and headless, the line
    is the schedule's starting candidate, unchecked.
    """
    rule, m = schedule_rule(x)
    out = A.simulate(fr, rule, m, x, ALPHA, draws, seed)
    k_fin, n_fin, s_fin, r_fin = final_round(out)
    if not np.array_equal(out.k[out.k > 0], k_fin[out.k > 0]):
        raise AssertionError("a promised set differs from the set the check ended on")
    rounds = rule.rounds if rule.kind == "b" else 1
    lo, hi = likely_range(s_fin, n_fin, k_fin, range_tail(rounds))
    if not np.array_equal(confirmed_by_range := lo >= x - A.EPS, out.k > 0):
        raise AssertionError(f"{int((confirmed_by_range != (out.k > 0)).sum())} checks disagree with their range")
    hits = fr.hits(k_fin)
    prec = hits / np.maximum(k_fin, 1)
    n_pos = fr.meta["n_pos"].to_numpy()[:, None]
    ok = A.oracle_k(fr, x)
    orec = np.broadcast_to(A.recall_of(fr, ok)[:, None], k_fin.shape)
    k_unc = A.start_k(fr, rule, x)
    p_unc = np.broadcast_to((fr.hits(k_unc) / np.maximum(k_unc, 1))[:, None], k_fin.shape)
    r_unc = np.broadcast_to((fr.hits(k_unc) / fr.meta["n_pos"].to_numpy())[:, None], k_fin.shape)
    confirmed = out.k > 0
    covered = (lo - A.EPS <= prec) & (prec <= hi + A.EPS)
    cells = fr.meta["cell"].to_numpy()

    masks = [(name, np.broadcast_to(sel[:, None], k_fin.shape)) for name, sel in A.slices(fr)]
    if rule.rounds > 1:
        masks += [(f"ended in round {r}", r_fin == r) for r in range(1, rule.rounds + 1)]
    rows = []
    for name, sel in masks:
        if not sel.any():
            continue
        short = sel & ~confirmed
        conf = sel & confirmed

        def mean(v: np.ndarray, where: np.ndarray) -> float:
            return float(v[where].mean()) if where.any() else float("nan")

        def ratio(a: float, b: float) -> float:
            return a / b if b > 0 else float("nan")

        rows.append(
            {
                "world": fr.world,
                "X": x,
                "rule": rule.name,
                "K": int(rule.start),
                "rounds": rounds,
                "m": m,
                "slice": name,
                "draws": int(sel.sum()),
                "confirmed": mean(confirmed, sel),
                "votes_mean": mean(out.votes, sel),
                "returned": mean(k_fin, sel),
                "precision": mean(prec, sel),
                "reached_x": mean(prec >= x - A.EPS, sel),
                "recall_share": ratio(mean(hits / n_pos, sel), mean(orec, sel)),
                "coverage": mean(covered, sel),
                "se_coverage": A.cluster_se((covered & sel).sum(1), sel.sum(1), cells),
                "range_above_truth": mean(prec < lo - A.EPS, sel),
                "range_below_truth": mean(prec > hi + A.EPS, sel),
                "range_width": mean(hi - lo, sel),
                "coverage_confirmed": mean(covered, conf),
                "coverage_short": mean(covered, short),
                "short_precision": mean(prec, short),
                "short_range_lo": mean(lo, short),
                "short_range_hi": mean(hi, short),
                "unchecked_precision": mean(p_unc, sel),
                "unchecked_reached_x": mean(p_unc >= x - A.EPS, sel),
                "unchecked_recall_share": ratio(mean(r_unc, sel), mean(orec, sel)),
            }
        )
    return rows


def run_best_attempt(frames: dict[str, A.Frames], draws: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame([r for fr in frames.values() for x in FLOORS for r in best_attempt_rows(fr, x, draws, seed)])


def run(frames: dict[str, A.Frames], draws: int, seed: int, grid: bool = True) -> pd.DataFrame:
    rows = []
    for fr in frames.values():
        for x in FLOORS:
            rows += rows_for(fr, x, [schedule_rule(x)], draws, seed, "schedule")
            if grid and x <= 0.5:
                rows += rows_for(fr, x, grid_rules(), draws, seed, "grid")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--frames", type=Path, default=A.FRAMES_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--draws", type=int, default=A.DRAWS)
    args = ap.parse_args()
    frames = A.load_frames(args.frames, A.WORLDS)
    df = run(frames, args.draws, SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    keep = [
        "part", "world", "X", "rule", "K", "rounds", "m", "slice", "frames", "promises", "reachable", "promised",
        "broken_of_made", "se_broken_of_made", "broken_of_all", "recall", "oracle", "recall_share", "census_of_made",
        "new_pos", "votes_mean", "votes_p90", "returned", "oracle_k_median", "d_recall_vs_read32",
        "se_d_recall_vs_read32",
    ]  # fmt: skip
    df[keep].to_csv(args.out / "summary.csv", index=False, float_format="%.5g")
    best = run_best_attempt(frames, args.draws, SEED)
    best.to_csv(args.out / "best_attempt.csv", index=False, float_format="%.5g")
    sched = [{"X": x, **dict(zip(("K", "rounds", "m"), schedule_for(x)))} for x in FLOORS]
    prov = {
        "inputs": {w: hashlib.sha256((args.frames / f).read_bytes()).hexdigest()[:16] for w, f in A.WORLDS.items()},
        "alpha": ALPHA,
        "draws": args.draws,
        "seed": SEED,
        "floors": FLOORS,
        "schedule": sched,
        "grid": {"one_round": ONE_ROUND, "shrinking_to_32": SHRINKING},
        "likely_range": "Clopper-Pearson from the labels inside the set, each tail at alpha / rounds",
    }
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"wrote {args.out / 'summary.csv'} ({len(df)} rows) and best_attempt.csv ({len(best)} rows)")


if __name__ == "__main__":
    main()
