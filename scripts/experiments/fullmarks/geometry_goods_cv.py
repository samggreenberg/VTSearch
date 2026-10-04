"""Geometry cuts calibrated by the Goods' own fits, priced by class-split CV (#4477).

H1 (#4440) applies #4434's geometry cuts until a detector's first Bad vote. A class whose
clicks are all Goods never gets a Bad, so the cuts apply all session. That is right where the
loose fits are mostly negatives (``tobacco800/logo_ajj10e00_1``), and wrong where the mark itself
fits loosely (``ald41a00``, replicate 2). The hypothesis pre-registered on #4477: when most of the
Goods' own leave-one-out fits are loose, the cuts are wrong for this mark.

A rule ``(s, n)`` drops the cuts, in a state with votes but no Bad, when at least ``n`` Goods have a
leave-one-out fit and fewer than a share ``s`` of those are tight. Shipped means the cuts always
apply. Click 0 and every state after the first Bad keep the shipped line. The choice and the test
are the class-split CV of #4458 round 2 (``balance_rules_cv``), made at beta 1. Beta 1/4 shares the
line and is checked per click.

    python geometry_goods_cv.py --frames <tier m>/m-rep1/frames --frames <tier m>/m-rep2/frames \\
        --cuts <cuts.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

import balance_rules_cv as cv
from balance_rules import paired

S_GRID = (0.25, 0.5, 0.75, 0.9)
N_GRID = (2, 3, 5)
BETAS = (1.0, 0.25)
#: Hypothesis 2 (#4477): in states with votes but no Bad, a loose fit passes at F inliers.
T2_GRID = (10, 12, 16, 20, 24, 32, 48, 64)
Q2_GRID: tuple[Optional[float], ...] = (None, 0.25, 0.5, 0.75)


class LooseFloor(NamedTuple):
    t: Optional[int]  # None: shipped, a loose fit never passes
    q: Optional[float]

    def label(self) -> str:
        if self.t is None:
            return "shipped (a loose fit never passes)"
        return f"a loose fit passes at max({self.t}, ceil({self.q} x the Goods' median LOO inliers))"


SHIPPED2 = LooseFloor(None, None)


def family2() -> list[LooseFloor]:
    return [SHIPPED2] + [LooseFloor(t, q) for t in T2_GRID for q in Q2_GRID]


class Rule(NamedTuple):
    s: Optional[float]  # None: shipped, the cuts always apply
    n: int

    def label(self) -> str:
        return (
            "shipped (cuts always)"
            if self.s is None
            else f"drop the cuts if < {self.s:g} of >= {self.n} Goods fit tight"
        )


SHIPPED = Rule(None, 0)


def family() -> list[Rule]:
    return [SHIPPED] + [Rule(s, n) for s in S_GRID for n in N_GRID]


class Frame(NamedTuple):
    counts: cv.Counts
    n_fit: int  # Goods with a leave-one-out fit
    tight_share: float


def load(path: Path, cuts: dict[str, float], suffix: str) -> Frame:
    z = np.load(path, allow_pickle=True)
    r, e = z["good_loo_ratio"], z["good_loo_reproj"]
    fit = np.isfinite(r) & np.isfinite(e)
    tight = (r[fit] >= cuts["ratio_min"]) & (e[fit] <= cuts["reproj_max"])
    return Frame(cv.load_counts(path, cuts, suffix), int(fit.sum()), float(tight.mean()) if fit.any() else float("nan"))


def share(f: Frame, rule: Any, beta: float) -> float:
    """The returned set's F-beta over the best cut's under *rule* (a :class:`Rule` or :class:`LooseFloor`)."""
    c = f.counts
    best = c.best[beta]
    if not np.isfinite(best) or best <= 0:
        return float("nan")
    last = c.cum_n.shape[1] - 1
    if c.n_bads:  # the Bad ceiling, no cuts
        i, row = int(min(max(8, c.ceiling + 1), last)), 0
        n, tp = int(c.cum_n[row, i]), int(c.cum_tp[row, i])
    elif c.n_votes == 0:
        n, tp = int(c.cum_n[0, min(8, last)]), int(c.cum_tp[0, min(8, last)])
    elif isinstance(rule, LooseFloor):
        i8 = min(8, last)
        n, tp = int(c.cum_n[1, i8]), int(c.cum_tp[1, i8])  # tight fits at 8
        if rule.t is not None:
            floor = float(rule.t)
            if rule.q is not None and np.isfinite(c.good_median):
                floor = max(floor, float(np.ceil(rule.q * c.good_median)))
            j = int(min(floor, last))  # floor >= 10 > 8: add every fit at the floor, minus the tight ones counted
            n += int(c.cum_n[0, j]) - int(c.cum_n[1, j])
            tp += int(c.cum_tp[0, j]) - int(c.cum_tp[1, j])
    else:
        drop = rule.s is not None and f.n_fit >= rule.n and f.tight_share < rule.s
        i, row = min(8, last), (0 if drop else 1)
        n, tp = int(c.cum_n[row, i]), int(c.cum_tp[row, i])
    b2 = beta * beta
    return ((1 + b2) * tp / (b2 * c.pos + n)) / best if (b2 * c.pos + n) > 0 else float("nan")


def per_class(cells: dict[tuple[str, int], float], base: dict[str, str], ks: Sequence[int]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for (cid, v), x in cells.items():
        if v in ks and np.isfinite(x):
            acc[base[cid]].append(x)
    return {k: float(np.mean(x)) for k, x in acc.items()}


def mean_share(frames: list[Frame], rule: Rule, beta: float, clicks: Sequence[int]) -> float:
    per = [np.nanmean([share(f, rule, beta) for f in frames if f.counts.v == k] or [np.nan]) for k in clicks]
    per = [x for x in per if np.isfinite(x)]
    return float(np.mean(per)) if per else float("nan")


def choose(frames: list[Frame], rules: Sequence[Any]) -> Any:
    """The best rule at beta 1 over clicks 0-25; ties go to shipped (the family's first rule)."""
    scored = [(mean_share(frames, r, 1.0, cv.CHOICE_CLICKS), -i, r) for i, r in enumerate(rules)]
    return max(scored, key=lambda t: (np.nan_to_num(t[0], nan=-1.0), t[1]))[2]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, action="append", required=True)
    ap.add_argument("--cuts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--family", choices=("goods-tight", "loose-floor"), default="goods-tight")
    args = ap.parse_args(argv)
    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    frames = [
        load(p, cuts, f" ({i + 1})" if len(args.frames) > 1 else "")
        for i, d in enumerate(args.frames)
        for p in sorted(d.glob("*.npz"))
    ]
    base = {f.counts.cid: f.counts.base for f in frames}
    rules: list[Any] = family() if args.family == "goods-tight" else family2()
    ship_rule = rules[0]
    chosen = {k: choose([f for f in frames if cv.fold(f.counts.base) != k], rules) for k in (0, 1)}
    final = choose(frames, rules)
    clicks = sorted({f.counts.v for f in frames})
    early = [k for k in clicks if k <= 25]
    lines = [
        f"{len(set(base.values()))} classes, {len(frames)} frames.",
        f"- chosen on fold 1, scored on fold 0: {chosen[0].label()}",
        f"- chosen on fold 0, scored on fold 1: {chosen[1].label()}",
        f"- refit on all classes: {final.label()}",
        "",
    ]
    result: dict[str, Any] = {"fold0": chosen[0]._asdict(), "fold1": chosen[1]._asdict(), "final": final._asdict()}
    passes = True
    for beta in BETAS:
        held = {(f.counts.cid, f.counts.v): share(f, chosen[cv.fold(f.counts.base)], beta) for f in frames}
        ship = {(f.counts.cid, f.counts.v): share(f, ship_rule, beta) for f in frames}
        lines += [f"### beta {beta:g}", "", "| clicks | shipped | held-out | difference [95%] |", "|---|---:|---:|---|"]
        for label, ks in [("0–25", early)] + [(str(k), [k]) for k in clicks]:
            a, b = per_class(ship, base, ks), per_class(held, base, ks)
            common = sorted(set(a) & set(b))
            m, lo, hi = paired(np.array([b[c] - a[c] for c in common]))
            if label == "0–25" and beta == 1.0:
                passes &= lo > 0
            elif label != "0–25":
                passes &= lo >= -0.02
            lines.append(
                f"| {label} | {np.mean([a[c] for c in common]):.3f} | {np.mean([b[c] for c in common]):.3f} | "
                f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
            )
        lines.append("")
    # Where the held-out rule changes the returned set at beta 1.
    changed: dict[str, int] = defaultdict(int)
    for f in frames:
        r = chosen[cv.fold(f.counts.base)]
        if share(f, r, 1.0) != share(f, ship_rule, 1.0):
            changed[f.counts.base] += 1
    lines += [f"**{'PASSES' if passes else 'fails'}.** Frames the held-out rule changes, by class: {dict(changed)}", ""]
    result["passes"] = bool(passes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cv.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (args.out / "cv.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
