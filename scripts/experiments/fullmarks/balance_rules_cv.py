"""A document line that follows the balance, round 2: presets 1/4, 1, 4, class-split CV (#4458).

Round 1 (``balance_rules.py``) fitted per-beta inlier floors on tier ``s`` and failed on tier
``m``. Round 2 widens the family for the stronger presets (#4472) and, with no unseen document
data left (FullMarks tiers are nested), tests by class-split cross-validation on tier ``m``'s
frames, as pre-registered on #4458:

* the classes split in two by the parity of the first byte of ``sha256(class id)``;
* each beta's rule is chosen on one fold's classes (mean share of the best cut over clicks 0-25,
  ties to the rule closest to shipped) and scored on the other's;
* every class is scored with a rule chosen without it, and the pooled held-out shares are paired
  against the shipped line by class.

A page is accepted if it is verified with inliers >= T, or, with the tail on, unverified with a
Stage-1 score >= a percentile of the Goods' leave-one-out Stage-1 scores.
T = max(t, ceil(q x the Goods' median leave-one-out inliers), Bad ceiling + d + 1); the geometry
cuts apply per state (click 0, votes without a Bad, after a Bad).

Every rule reduces to one threshold per frame, so each frame keeps cumulative counts by inlier
value (with and without the geometry cuts) and all 6,048 rules are scored at once.

    python balance_rules_cv.py --frames <tier m>/m-rep1/frames --frames <tier m>/m-rep2/frames \\
        --cuts <cuts.json> --out <dir> [--check-frames <tier s>/frames]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

from balance_rules import paired

BETAS = (0.25, 1.0, 4.0)
T_GRID = (4, 6, 8, 10, 12, 16, 20, 24, 32)
Q_GRID: tuple[Optional[float], ...] = (None, 0.25, 0.5, 0.75)
D_GRID: tuple[Optional[int], ...] = (None, -4, -2, 0, 2, 4, 8)
TAIL_GRID: tuple[Optional[int], ...] = (None, 10, 25)
CHOICE_CLICKS = (0, 1, 2, 3, 5, 10, 15, 25)


class Rule(NamedTuple):
    t: int
    q: Optional[float]
    d: Optional[int]
    geo0: bool
    geo1: bool
    geo2: bool
    tail: Optional[int]

    def label(self) -> str:
        on = lambda b: "on" if b else "off"  # noqa: E731
        return (
            f"t={self.t} q={self.q} d={self.d} geo={on(self.geo0)}/{on(self.geo1)}/{on(self.geo2)} "
            f"tail={'off' if self.tail is None else f'p{self.tail}'}"
        )


SHIPPED = Rule(t=8, q=None, d=0, geo0=False, geo1=True, geo2=False, tail=None)


def family() -> list[Rule]:
    return [
        Rule(t, q, d, g0, g1, g2, tail)
        for t, q, d, g0, g1, g2, tail in itertools.product(
            T_GRID, Q_GRID, D_GRID, (False, True), (False, True), (False, True), TAIL_GRID
        )
    ]


def distance(r: Rule) -> float:
    """How far a rule is from the shipped line, to break ties toward it."""
    return (
        abs(r.t - SHIPPED.t)
        + (0 if r.q is None else 10 * r.q)
        + (99 if r.d is None else abs(r.d))
        + (r.geo0 != SHIPPED.geo0)
        + (r.geo1 != SHIPPED.geo1)
        + (r.geo2 != SHIPPED.geo2)
        + (0 if r.tail is None else 5)
    )


class Counts(NamedTuple):
    """One frame, reduced to what any rule in the family needs."""

    cid: str  # with the replicate suffix
    base: str  # the class
    v: int
    n_votes: int
    n_bads: int
    ceiling: float
    good_median: float  # the Goods' median leave-one-out inliers (nan with < 2)
    cum_n: np.ndarray  # (2, L): verified test pages with inliers >= T; row 1 also tight
    cum_tp: np.ndarray
    tail_n: tuple[int, int, int]  # off, p10, p25
    tail_tp: tuple[int, int, int]
    pos: int
    best: dict[float, float]


def _best(hits: np.ndarray, beta: float) -> float:
    pos = int(hits.sum())
    if pos == 0:
        return float("nan")
    tp = np.cumsum(hits)
    b2 = beta * beta
    return float(((1 + b2) * tp / (b2 * pos + np.arange(1, len(hits) + 1))).max())


def load_counts(path: Path, cuts: dict[str, float], suffix: str = "") -> Counts:
    z = np.load(path, allow_pickle=True)
    cid, v = path.stem.rsplit("__v", 1)
    cid = cid.replace("__", "/", 1)
    test = z["test"].astype(bool)
    positive = z["positive"].astype(bool)
    short = z["shortlisted"].astype(bool)
    inl = np.where(short, np.nan_to_num(z["inliers"], nan=-1.0), -1.0)
    ratio = np.nan_to_num(z["ratio"], nan=-1.0)
    reproj = np.nan_to_num(z["reproj"], nan=np.inf)
    tight = (ratio >= cuts["ratio_min"]) & (reproj <= cuts["reproj_max"])
    ver = test & short & (inl >= 0)
    vi = inl[ver].astype(np.int64)
    length = int(vi.max()) + 2 if vi.size else 2
    cum_n = np.zeros((2, length), dtype=np.int64)
    cum_tp = np.zeros((2, length), dtype=np.int64)
    for row, mask in enumerate((np.ones(vi.size, dtype=bool), tight[ver])):
        n = np.bincount(vi[mask], minlength=length)
        tp = np.bincount(vi[mask & positive[ver]], minlength=length)
        cum_n[row] = n[::-1].cumsum()[::-1]
        cum_tp[row] = tp[::-1].cumsum()[::-1]
    g_s1 = z["good_loo_stage1"][np.isfinite(z["good_loo_stage1"])]
    tail_n, tail_tp = [0], [0]
    for pct in (10, 25):
        if g_s1.size >= 2:
            sel = test & ~short & (z["stage1"] >= np.percentile(g_s1, pct))
            tail_n.append(int(sel.sum()))
            tail_tp.append(int((sel & positive).sum()))
        else:
            tail_n.append(0)
            tail_tp.append(0)
    g_inl = z["good_loo_inliers"][np.isfinite(z["good_loo_inliers"])]
    key = np.where(short, 1e6 + inl, z["stage1"])
    idx = np.flatnonzero(test)
    hits = positive[idx[np.argsort(-key[idx], kind="stable")]]
    bad = z["bad_inliers"][np.isfinite(z["bad_inliers"])]
    return Counts(
        cid=cid + suffix,
        base=cid,
        v=int(v),
        n_votes=len(z["good_ids"]) + len(z["bad_ids"]),
        n_bads=len(z["bad_ids"]),
        ceiling=float(bad.max()) if bad.size else 0.0,
        good_median=float(np.median(g_inl)) if g_inl.size >= 2 else float("nan"),
        cum_n=cum_n,
        cum_tp=cum_tp,
        tail_n=(tail_n[0], tail_n[1], tail_n[2]),
        tail_tp=(tail_tp[0], tail_tp[1], tail_tp[2]),
        pos=int((test & positive).sum()),
        best={b: _best(hits, b) for b in BETAS},
    )


class RuleArrays(NamedTuple):
    t: np.ndarray
    q: np.ndarray  # nan = off
    d: np.ndarray  # nan = no ceiling
    geo: np.ndarray  # (3, R): click 0, votes without a Bad, after a Bad
    tail: np.ndarray  # 0 off, 1 p10, 2 p25


def rule_arrays(rules: Sequence[Rule]) -> RuleArrays:
    return RuleArrays(
        t=np.array([r.t for r in rules], dtype=np.int64),
        q=np.array([np.nan if r.q is None else r.q for r in rules]),
        d=np.array([np.nan if r.d is None else r.d for r in rules]),
        geo=np.array([[r.geo0 for r in rules], [r.geo1 for r in rules], [r.geo2 for r in rules]], dtype=np.int64),
        tail=np.array([0 if r.tail is None else (1 if r.tail == 10 else 2) for r in rules], dtype=np.int64),
    )


def shares(c: Counts, ra: RuleArrays, beta: float) -> np.ndarray:
    """Every rule's share of the best cut on one frame (nan without test positives)."""
    best = c.best[beta]
    if not np.isfinite(best) or best <= 0:
        return np.full(len(ra.t), np.nan)
    state = 2 if c.n_bads else (1 if c.n_votes else 0)
    geo = ra.geo[state]
    thr = ra.t.astype(np.float64)
    if np.isfinite(c.good_median):
        thr = np.where(np.isnan(ra.q), thr, np.maximum(thr, np.ceil(np.nan_to_num(ra.q) * c.good_median)))
    if c.n_bads:
        thr = np.where(np.isnan(ra.d), thr, np.maximum(thr, c.ceiling + np.nan_to_num(ra.d) + 1))
    idx = np.clip(thr, 0, c.cum_n.shape[1] - 1).astype(np.int64)
    n = c.cum_n[geo, idx] + np.asarray(c.tail_n)[ra.tail]
    tp = c.cum_tp[geo, idx] + np.asarray(c.tail_tp)[ra.tail]
    b2 = beta * beta
    denom = b2 * c.pos + n
    f = np.where(denom > 0, (1 + b2) * tp / np.where(denom > 0, denom, 1), np.nan)
    return f / best


def fold(cls: str) -> int:
    return hashlib.sha256(cls.encode()).digest()[0] % 2


def mean_over_clicks(per_frame: dict[tuple[str, int], np.ndarray], clicks: Sequence[int]) -> np.ndarray:
    """Per rule: the mean over clicks of the mean over class sessions."""
    out = []
    for k in clicks:
        rows = [s for (_cid, v), s in per_frame.items() if v == k]
        if rows:
            out.append(np.nanmean(np.vstack(rows), axis=0))
    return np.nanmean(np.vstack(out), axis=0)


def choose(per_frame: dict[tuple[str, int], np.ndarray], rules: Sequence[Rule]) -> Rule:
    m = mean_over_clicks(per_frame, CHOICE_CLICKS)
    order = sorted(range(len(rules)), key=lambda i: (-np.nan_to_num(m[i], nan=-1.0), distance(rules[i])))
    return rules[order[0]]


def load_all(dirs: Sequence[Path], cuts: dict[str, float]) -> list[Counts]:
    out = []
    for i, d in enumerate(dirs):
        suffix = f" ({i + 1})" if len(dirs) > 1 else ""
        out += [load_counts(p, cuts, suffix) for p in sorted(d.glob("*.npz"))]
    return out


def per_class(cells: dict[tuple[str, int], float], base_of: dict[str, str], clicks: Sequence[int]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for (cid, v), s in cells.items():
        if v in clicks and np.isfinite(s):
            acc[base_of[cid]].append(s)
    return {c: float(np.mean(x)) for c, x in acc.items()}


def run(args: argparse.Namespace) -> int:
    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    frames = load_all(args.frames, cuts)
    rules = family()
    ra = rule_arrays(rules)
    ship_i = rules.index(SHIPPED)
    base_of = {c.cid: c.base for c in frames}
    clicks = sorted({c.v for c in frames})
    early = [k for k in clicks if k <= 25]
    result: dict[str, Any] = {}
    lines = [
        f"{len({c.base for c in frames})} classes, {len(frames)} frames, {len(rules)} rules per beta; "
        f"folds: {sum(fold(b) == 0 for b in set(base_of.values()))} / {sum(fold(b) == 1 for b in set(base_of.values()))} classes.",
        "",
    ]
    for beta in BETAS:
        per_frame = {(c.cid, c.v): shares(c, ra, beta) for c in frames}
        chosen = {f: choose({k: s for k, s in per_frame.items() if fold(base_of[k[0]]) != f}, rules) for f in (0, 1)}
        held = {(cid, v): s[rules.index(chosen[fold(base_of[cid])])] for (cid, v), s in per_frame.items()}
        ship = {(cid, v): s[ship_i] for (cid, v), s in per_frame.items()}
        final = choose(per_frame, rules)
        rows = []
        verdict_ok = True
        for label, ks in [("0–25 (the bar)", early)] + [(str(k), [k]) for k in clicks]:
            a, b = per_class(ship, base_of, ks), per_class(held, base_of, ks)
            common = sorted(set(a) & set(b))
            d = np.array([b[c] - a[c] for c in common])
            m, lo, hi = paired(d)
            if label.startswith("0–25"):
                verdict_ok &= lo > 0
            else:
                verdict_ok &= lo >= -0.02
            rows.append(
                f"| {label} | {np.mean([a[c] for c in common]):.3f} | {np.mean([b[c] for c in common]):.3f} | "
                f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
            )
        lines += [
            f"### beta {beta:g}: {'PASSES' if verdict_ok else 'fails'}",
            "",
            f"- chosen on fold 1, scored on fold 0: {chosen[0].label()}",
            f"- chosen on fold 0, scored on fold 1: {chosen[1].label()}",
            f"- refit on all classes: {final.label()}",
            "",
            "| clicks | shipped | held-out chosen | difference [95%] |",
            "|---|---:|---:|---|",
            *rows,
            "",
        ]
        result[f"{beta:g}"] = {
            "passes": bool(verdict_ok),
            "fold0": chosen[0]._asdict(),
            "fold1": chosen[1]._asdict(),
            "final": final._asdict(),
        }
        if args.check_frames:
            check = load_all(args.check_frames, cuts)
            cb = {c.cid: c.base for c in check}
            fi, si = rules.index(final), ship_i
            cells = {(c.cid, c.v): shares(c, ra, beta) for c in check}
            ck = sorted({c.v for c in check})
            a = per_class({k: s[si] for k, s in cells.items()}, cb, ck)
            b = per_class({k: s[fi] for k, s in cells.items()}, cb, ck)
            common = sorted(set(a) & set(b))
            m, lo, hi = paired(np.array([b[c] - a[c] for c in common]))
            lines += [f"Check on the other tier (final rule, clicks {ck}): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cv.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (args.out / "cv.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, action="append", required=True)
    ap.add_argument("--cuts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-frames", type=Path, action="append")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
