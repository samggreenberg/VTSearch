"""The document line at the precision end of the balance: a subset of beta 1's set (#4479).

The mirror of #4458's recall end (#4475). Beta 1/4's set is beta 1's set intersected with
{verified pages with inliers >= T, and a tight fit if ``geo2`` and a Bad exists}, where
T = max(t, ceil(q x the Goods' median leave-one-out inliers), Bad ceiling + d + 1). With t >= 8
and d >= 0 the intersection is one threshold per state:

* click 0 (the example sort's plain gate): inliers >= t;
* votes but no Bad (H1's gate, which already demands a tight fit): tight and inliers >= T;
* after a Bad (the Bad ceiling): inliers >= T, and tight if ``geo2``.

Chosen and tested by the class-split CV of #4458 round 2 (``balance_rules_cv``), at beta 1/4, as
pre-registered on #4479.

    python precision_end_cv.py --frames <m-rep1>/frames --frames <m-rep2>/frames --cuts <cuts.json> --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

import balance_rules_cv as cv
from balance_rules import paired

BETA = 0.25
T_GRID = (8, 10, 12, 16, 20, 24)
Q_GRID: tuple[Optional[float], ...] = (None, 0.25, 0.5, 0.75)
D_GRID = (0, 2, 4, 8)


class Rule(NamedTuple):
    t: int
    q: Optional[float]
    d: int
    geo2: bool

    def label(self) -> str:
        return f"t={self.t} q={self.q} d={self.d} geo2={'on' if self.geo2 else 'off'}"


SHIPPED = Rule(8, None, 0, False)  # the intersection with beta 1's set is beta 1's set


def family() -> list[Rule]:
    return [Rule(*p) for p in itertools.product(T_GRID, Q_GRID, D_GRID, (False, True))]


def distance(r: Rule) -> float:
    return abs(r.t - 8) + (0 if r.q is None else 10 * r.q) + r.d + (2 if r.geo2 else 0)


def share(c: cv.Counts, r: Rule, beta: float) -> float:
    best = c.best[beta]
    if not np.isfinite(best) or best <= 0:
        return float("nan")
    last = c.cum_n.shape[1] - 1
    thr = float(r.t)
    if r.q is not None and np.isfinite(c.good_median):
        thr = max(thr, float(np.ceil(r.q * c.good_median)))
    if c.n_bads:
        thr = max(thr, c.ceiling + r.d + 1)
        row = 1 if r.geo2 else 0
    elif c.n_votes:
        row = 1  # H1: votes without a Bad already demand a tight fit
    else:
        thr, row = float(r.t), 0  # click 0: no Goods for q, no Bads for d
    i = int(min(thr, last))
    n, tp = int(c.cum_n[row, i]), int(c.cum_tp[row, i])
    b2 = beta * beta
    return ((1 + b2) * tp / (b2 * c.pos + n)) / best if (b2 * c.pos + n) > 0 else float("nan")


def choose(frames: list[cv.Counts], rules: Sequence[Rule]) -> Rule:
    def mean(r: Rule) -> float:
        per = [np.nanmean([share(c, r, BETA) for c in frames if c.v == k] or [np.nan]) for k in cv.CHOICE_CLICKS]
        per = [x for x in per if np.isfinite(x)]
        return float(np.mean(per)) if per else float("nan")

    return max(rules, key=lambda r: (np.nan_to_num(mean(r), nan=-1.0), -distance(r)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, action="append", required=True)
    ap.add_argument("--cuts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    frames = cv.load_all(args.frames, cuts)
    base = {c.cid: c.base for c in frames}
    rules = family()
    chosen = {k: choose([c for c in frames if cv.fold(c.base) != k], rules) for k in (0, 1)}
    final = choose(frames, rules)
    clicks = sorted({c.v for c in frames})
    early = [k for k in clicks if k <= 25]
    held = {(c.cid, c.v): share(c, chosen[cv.fold(c.base)], BETA) for c in frames}
    ship = {(c.cid, c.v): share(c, SHIPPED, BETA) for c in frames}
    lines = [
        f"{len(set(base.values()))} classes, {len(frames)} frames, {len(rules)} rules; choice and bar at beta {BETA:g}.",
        f"- chosen on fold 1, scored on fold 0: {chosen[0].label()}",
        f"- chosen on fold 0, scored on fold 1: {chosen[1].label()}",
        f"- refit on all classes: {final.label()}",
        "",
        "| clicks | shipped | held-out | difference [95%] |",
        "|---|---:|---:|---|",
    ]
    passes = True
    for label, ks in [("0–25", early)] + [(str(k), [k]) for k in clicks]:
        a, b = cv.per_class(ship, base, ks), cv.per_class(held, base, ks)
        common = sorted(set(a) & set(b))
        m, lo, hi = paired(np.array([b[x] - a[x] for x in common]))
        passes &= (lo > 0) if label == "0–25" else (lo >= -0.02)
        lines.append(
            f"| {label} | {np.mean([a[x] for x in common]):.3f} | {np.mean([b[x] for x in common]):.3f} | "
            f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
        )
    lines += ["", f"**{'PASSES' if passes else 'fails'}.**", ""]
    result: dict[str, Any] = {
        "passes": bool(passes),
        "fold0": chosen[0]._asdict(),
        "fold1": chosen[1]._asdict(),
        "final": final._asdict(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cv.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (args.out / "cv.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
