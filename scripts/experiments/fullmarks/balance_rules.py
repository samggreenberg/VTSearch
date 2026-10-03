"""A returned-set line that follows the balance, priced offline on per-page frames (#4458).

The structural line ignores the user's balance (F-beta's beta, #4413). The line does not
change which page the user clicks next, so the sessions are the same at every beta and a
beta-aware rule can be scored exactly on frames saved along the shipped sessions
(``sota_documents.py --frames 0,1,2,3,5,10,15,25[,50]``).

The rule family pre-registered on #4458, one rule per beta:

``t``      a minimum inlier count, in every state
``geo0``   #4434's geometry cuts at click 0 (on the crop's fit), on or off
``geo1``   the geometry cuts with votes but no Bad yet, on or off
``d``      after the first Bad: accept inliers > the Bads' best inliers + d (``None``: no ceiling)
``geo2``   the geometry cuts after the first Bad, on or off

The shipped line, at every beta, is ``t=8, geo0 off, geo1 on, d=0, geo2 off``.

The score of a rule at one frame is the returned set's F-beta as a share of the best F-beta any
cut of the app's order reaches, on the test half. ``search`` picks the best rule per beta on one
tier's frames (ties to the rule closest to shipped). ``score`` pairs a chosen rule against the
shipped line on another tier's frames, by class.

    python balance_rules.py search --frames <tier s run>/frames --cuts <cuts.json> --out <dir>
    python balance_rules.py score --frames <tier m run>/frames [--frames ...] --cuts <cuts.json> \\
        --chosen <dir>/chosen.json --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

MIN_INLIERS = 8
BETAS = (0.5, 2.0)  # beta 1 keeps the shipped line (#4458)
T_GRID = (8, 10, 12, 16, 20, 24, 32)
D_GRID: tuple[Optional[int], ...] = (None, -2, 0, 2, 4, 8)
SEARCH_CLICKS = (0, 1, 2, 3, 5, 10, 15, 25)
BOOTSTRAP = 10000


class Rule(NamedTuple):
    t: int
    geo0: bool
    geo1: bool
    d: Optional[int]
    geo2: bool

    def label(self) -> str:
        on = lambda b: "on" if b else "off"  # noqa: E731
        return f"t={self.t} geo0={on(self.geo0)} geo1={on(self.geo1)} d={self.d} geo2={on(self.geo2)}"


SHIPPED = Rule(t=8, geo0=False, geo1=True, d=0, geo2=False)


def distance(rule: Rule) -> float:
    """How far a rule is from the shipped line, to break ties toward it."""
    d = 99 if rule.d is None else abs(rule.d)
    return (
        abs(rule.t - SHIPPED.t)
        + d
        + (rule.geo0 != SHIPPED.geo0)
        + (rule.geo1 != SHIPPED.geo1)
        + (rule.geo2 != SHIPPED.geo2)
    )


def family() -> list[Rule]:
    return [Rule(*p) for p in itertools.product(T_GRID, (False, True), (False, True), D_GRID, (False, True))]


class Frame(NamedTuple):
    cid: str
    v: int
    test: np.ndarray
    positive: np.ndarray
    order: np.ndarray  # test-half page indices in the app's order
    inliers: np.ndarray  # -1 where not verified
    tight: np.ndarray  # passes the geometry cuts
    n_votes: int
    n_bads: int
    ceiling: float


def load_frame(path: Path, cuts: dict[str, float]) -> Frame:
    z = np.load(path, allow_pickle=True)
    stem = path.stem
    cid, v = stem.rsplit("__v", 1)
    inl = np.where(z["shortlisted"], np.nan_to_num(z["inliers"], nan=-1.0), -1.0)
    ratio = np.nan_to_num(z["ratio"], nan=-1.0)
    reproj = np.nan_to_num(z["reproj"], nan=np.inf)
    tight = (ratio >= cuts["ratio_min"]) & (reproj <= cuts["reproj_max"])
    key = np.where(z["shortlisted"], 1e6 + inl, z["stage1"])
    test = z["test"].astype(bool)
    idx = np.flatnonzero(test)
    order = idx[np.argsort(-key[idx], kind="stable")]
    bad = z["bad_inliers"][np.isfinite(z["bad_inliers"])]
    return Frame(
        cid=cid.replace("__", "/", 1),
        v=int(v),
        test=test,
        positive=z["positive"].astype(bool),
        order=order,
        inliers=inl,
        tight=tight,
        n_votes=len(z["good_ids"]) + len(z["bad_ids"]),
        n_bads=len(z["bad_ids"]),
        ceiling=float(bad.max()) if bad.size else 0.0,
    )


def accept(rule: Rule, f: Frame) -> np.ndarray:
    ok = f.inliers >= max(rule.t, MIN_INLIERS)
    if f.n_bads == 0:
        geo = rule.geo0 if f.n_votes == 0 else rule.geo1
    else:
        geo = rule.geo2
        if rule.d is not None:
            ok &= f.inliers > f.ceiling + rule.d
    if geo:
        ok &= f.tight
    return ok & f.test


def f_beta(tp: Any, k: Any, pos: int, beta: float) -> Any:
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * pos + k)


def best_f_beta(f: Frame, beta: float) -> float:
    hits = f.positive[f.order]
    pos = int(hits.sum())
    if pos == 0:
        return float("nan")
    tp = np.cumsum(hits)
    return float(f_beta(tp, np.arange(1, len(hits) + 1), pos, beta).max())


def share(rule: Rule, f: Frame, beta: float, best: float) -> float:
    if not np.isfinite(best) or best <= 0:
        return float("nan")
    acc = accept(rule, f)
    pos = int((f.positive & f.test).sum())
    k, tp = int(acc.sum()), int((acc & f.positive).sum())
    return float(f_beta(tp, k, pos, beta)) / best if (k + pos) else float("nan")


def load_frames(dirs: Sequence[Path], cuts: dict[str, float], tag: str = "") -> list[Frame]:
    out = []
    for i, d in enumerate(dirs):
        suffix = f" ({i + 1})" if len(dirs) > 1 else tag
        for p in sorted(d.glob("*.npz")):
            fr = load_frame(p, cuts)
            out.append(fr._replace(cid=fr.cid + suffix))
    return out


def table(frames: list[Frame], rules: Sequence[Rule], beta: float) -> dict[Rule, dict[tuple[str, int], float]]:
    """``{rule: {(class, click): share}}``."""
    best = {(f.cid, f.v): best_f_beta(f, beta) for f in frames}
    return {r: {(f.cid, f.v): share(r, f, beta, best[(f.cid, f.v)]) for f in frames} for r in rules}


def mean_over(cells: dict[tuple[str, int], float], clicks: Sequence[int]) -> float:
    """Mean over clicks of the mean over classes."""
    per_click = [np.nanmean([s for (c, v), s in cells.items() if v == k]) for k in clicks]
    per_click = [x for x in per_click if np.isfinite(x)]
    return float(np.mean(per_click)) if per_click else float("nan")


def search(args: argparse.Namespace) -> int:
    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    frames = load_frames(args.frames, cuts)
    clicks = [k for k in SEARCH_CLICKS if any(f.v == k for f in frames)]
    rules = family()
    chosen: dict[str, Any] = {"clicks": clicks}
    lines = [f"Search on {len({f.cid for f in frames})} classes, clicks {clicks}.", ""]
    rows = []
    for beta in BETAS:
        cells = table(frames, rules, beta)
        scored = sorted(((mean_over(cells[r], clicks), -distance(r), r) for r in rules), reverse=True)
        top, shipped = scored[0], mean_over(cells[SHIPPED], clicks)
        chosen[str(beta)] = top[2]._asdict()
        lines += [f"### beta {beta}", "", "| rule | mean share, clicks 0–25 |", "|---|---:|"]
        lines += [f"| {r.label()} | {m:.3f} |" for m, _d, r in scored[:8]]
        lines += [f"| **shipped** {SHIPPED.label()} | {shipped:.3f} |", ""]
        for m, _d, r in scored:
            rows.append({"beta": beta, "rule": r.label(), "mean_share": round(m, 4)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chosen.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    with (args.out / "search.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["beta", "rule", "mean_share"])
        w.writeheader()
        w.writerows(rows)
    text = "\n".join(lines) + "\n"
    (args.out / "search.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def paired(d: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(BOOTSTRAP, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def score(args: argparse.Namespace) -> int:
    """The chosen rule against the shipped line, paired by class (a class's replicates averaged)."""
    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    chosen = json.loads(args.chosen.read_text(encoding="utf-8"))
    frames = load_frames(args.frames, cuts)
    clicks = sorted({f.v for f in frames})
    lines = [f"Scored on {len({f.cid for f in frames})} class sessions, clicks {clicks}.", ""]
    for beta in BETAS:
        rule = Rule(**chosen[str(beta)])
        cells = table(frames, (SHIPPED, rule), beta)
        base = lambda cid: cid.split(" (")[0]  # noqa: E731

        def per_class(r: Rule, ks: Sequence[int]) -> dict[str, float]:
            acc: dict[str, list[float]] = defaultdict(list)
            for (cid, v), s in cells[r].items():
                if v in ks and np.isfinite(s):
                    acc[base(cid)].append(s)
            return {c: float(np.mean(x)) for c, x in acc.items()}

        lines += [f"### beta {beta}: {rule.label()}", ""]
        lines += ["| clicks | shipped | chosen | difference [95%] |", "|---|---:|---:|---|"]
        early = [k for k in clicks if k <= 25]
        for label, ks in [("0–25 (the bar)", early)] + [(str(k), [k]) for k in clicks]:
            a, b = per_class(SHIPPED, ks), per_class(rule, ks)
            common = sorted(set(a) & set(b))
            if not common:
                continue
            d = np.array([b[c] - a[c] for c in common])
            m, lo, hi = paired(d)
            lines.append(
                f"| {label} | {np.mean([a[c] for c in common]):.3f} | {np.mean([b[c] for c in common]):.3f} | "
                f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
            )
        lines.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (args.out / "score.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--frames", type=Path, action="append", required=True)
    s.add_argument("--cuts", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("score")
    c.add_argument("--frames", type=Path, action="append", required=True)
    c.add_argument("--cuts", type=Path, required=True)
    c.add_argument("--chosen", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return search(args) if args.cmd == "search" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
