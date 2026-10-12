#!/usr/bin/env python
"""Which drop marks the end of a person's photos (#4783), explored on the opening centroid.

Since #4759 the Goods' centroid keeps ``beta ** 0.708`` times the larger of #4603's count and the count
above the widest drop between neighbours in the top 11, and never fewer than that drop's count
(``count_gap_floor``). When a person's best photo stands far above their others, the widest drop is the
one right after it, and the line keeps that one photo. This script prices other drops in the same
composition, offline, on the cosines ``centroid_count_4759.py dump`` wrote (each person's K example photos'
centroid against the withheld half):

- ``widest``: the shipped drop;
- ``skip1_b<b>``: when the widest drop is the first, the widest one below the second cosine instead,
  if it is at least ``b`` times the first;
- ``rank_top<k>``: the drop that is widest once each is weighted by its rank (``i * d_i``, which
  evens out a bulk tail's spacings, wider at the top), over the top ``k + 1``;
- ``deepest_f<f>``: the deepest drop at least ``f`` times the widest;
- ``oracle_run``: the person's photos ranked above the first stranger, the ceiling a drop could reach.

Writes ``cells.csv`` (one row per cell x rule x beta; names people, owner-only), ``summary.csv`` (per dataset
x K x rule x beta: F-beta, the paired difference to ``widest`` with its SE on the person, kept, and how
often the rule moves the drop off a standout or off a drop that was right) and ``summary.md``.

Usage::

    python centroid_drop_4783.py price --dumps <explore>/dumps --out <dir>
    python centroid_drop_4783.py examples --dumps <explore>/dumps --dataset fhibe_faces_1024 --k 1 --rule R --out <md>
    python centroid_drop_4783.py labelled --dumps <explore>/dumps --out <dir>   # the app's labelled collection

``labelled`` prices the app's rules (``centroid_cut``) where the app's Test and example sort draw them on the
collection the user labelled: the one example photo is in the corpus, at cosine 1.0 to its own centroid. The line
is drawn over the example and the withheld half, and scored on the withheld half (the unvoted media), K = 1 only:
the dumps do not hold the K = 4 examples' own cosines.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from centroid_count_4759 import BETAS, EXPONENT, at_top, bulk, n_hat  # noqa: E402

TOP = 10  # the shipped depth: the top TOP + 1 cosines (CENTROID_GAP_TOP)


def drops(head: np.ndarray) -> np.ndarray:
    """``d_i`` = ``head[i - 1] - head[i]`` for i = 1..len(head) - 1 (head sorted descending)."""
    return head[:-1] - head[1:]


def drop_count(head: np.ndarray, rule: str, y_sorted: np.ndarray | None = None) -> int:
    """How many cosines sit above *rule*'s drop in *head* (the top cosines, descending)."""
    d = drops(head)
    if d.size == 0:
        return int(head.size)
    widest = int(np.argmax(d))
    if rule == "widest":
        return widest + 1
    if rule.startswith("skip1_b"):
        b = float(rule[len("skip1_b") :])
        if widest != 0 or d.size < 2:
            return widest + 1
        j = int(np.argmax(d[1:])) + 1
        return j + 1 if d[j] >= b * d[0] else 1
    if rule.startswith("rank_top"):
        k = int(rule[len("rank_top") :])
        dd = d[:k]
        return int(np.argmax(dd * np.arange(1, dd.size + 1))) + 1
    if rule.startswith("deepest_f"):
        f = float(rule[len("deepest_f") :])
        ok = np.flatnonzero(d >= f * d[widest])
        return int(ok.max()) + 1
    if rule == "oracle_run":
        assert y_sorted is not None
        run = int(np.argmax(y_sorted == 0)) if (y_sorted == 0).any() else int(y_sorted.size)
        return max(run, 1)
    raise ValueError(rule)


RULES = (
    ["widest"]
    + [f"skip1_b{b:g}" for b in (0.25, 0.33, 0.5, 0.75)]
    + [f"rank_top{k}" for k in (3, 5, 10)]
    + [f"deepest_f{f:g}" for f in (0.5, 0.75)]
    + ["oracle_run"]
)


def price_cell(s: np.ndarray, y: np.ndarray) -> list[dict]:
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    head = ss[: TOP + 1]
    counted = n_hat(s, "gauss", 4.0)
    if counted is None:
        return []
    run = drop_count(head, "oracle_run", ys)
    shipped = drop_count(head, "widest")
    base = {"n_pos": int(y.sum()), "clean_run": run, "widest_g": shipped,
            "standout": bool(run >= 2 and shipped == 1), "widest_right": bool(shipped == run)}  # fmt: skip
    rows = []
    for rule in RULES:
        g = drop_count(head, rule, ys)
        for beta in BETAS:
            c = beta**EXPONENT
            keep = min(max(round(c * max(counted, float(g))), g, 1), s.size)
            p, r, f = at_top(ys, keep, beta)
            rows.append(
                {**base, "rule": rule, "beta": beta, "g": g, "keep": keep, "precision": p, "recall": r, "fbeta": f}
            )
    return rows


def price(dumps: Path, out: Path) -> None:
    import pandas as pd  # noqa: PLC0415

    rows = []
    for path in sorted(dumps.glob("cosines_*_k*.npz")):
        z = np.load(path)
        off = z["offsets"]
        for i in range(len(off) - 1):
            s = z["scores"][off[i] : off[i + 1]].astype(np.float64)
            y = z["labels"][off[i] : off[i + 1]].astype(np.int64)
            for r in price_cell(s, y):
                r.update(
                    dataset=str(z["dataset"]), k=int(z["k"]), category=str(z["category"][i]), seed=int(z["seed"][i])
                )
                rows.append(r)
        print(f"{path.name}: {len(off) - 1} cells", file=sys.stderr)
    cells = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "cells.csv", index=False)
    key = ["dataset", "k", "category", "seed", "beta"]
    ref = cells[cells["rule"] == "widest"].set_index(key)
    agg = []
    for (ds, k, rule, beta), h in cells.groupby(["dataset", "k", "rule", "beta"], sort=False):
        hh = h.set_index(key)
        d = (hh["fbeta"] - ref["fbeta"].reindex(hh.index)).dropna()
        per = d.groupby(level="category").mean()
        moved = hh["g"] != ref["g"].reindex(hh.index)
        agg.append(
            {
                "dataset": ds, "k": k, "rule": rule, "beta": beta, "cells": len(h),
                "fbeta": h["fbeta"].mean(), "d_vs_widest": d.mean(),
                "d_se": per.std(ddof=1) / math.sqrt(len(per)) if len(per) > 1 else float("nan"),
                "kept_median": h["keep"].median(),
                "standouts": int(h["standout"].sum()),
                "standouts_fixed": float((h["standout"] & (h["g"] >= 2)).sum() / max(h["standout"].sum(), 1)),
                "right_broken": float((hh["widest_right"] & moved).sum() / max(hh["widest_right"].sum(), 1)),
            }
        )  # fmt: skip
    summary = pd.DataFrame(agg).round(4)
    summary.to_csv(out / "summary.csv", index=False)
    lines = ["# #4783 exploration: which drop, in count_gap_floor's composition (opening centroid)", ""]
    lines.append("dF = rule minus `widest` (the shipped drop), paired per person; fixed = share of standout "
                 "cells (clean run >= 2, widest drop keeps 1) the rule keeps >= 2 in; broken = share of cells "
                 "where the widest drop was exactly the clean run that the rule moves.")  # fmt: skip
    lines.append("")
    for (ds, k), g in summary.groupby(["dataset", "k"], sort=False):
        lines += [
            f"## {ds}, K = {k}",
            "",
            "| rule | beta | F | dF | kept | fixed | broken |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in g.sort_values(["beta", "rule"]).iterrows():
            lines.append(
                f"| {r['rule']} | {r['beta']:g} | {r['fbeta']:.3f} | {r['d_vs_widest']:+.3f} ± {r['d_se']:.3f} "
                f"| {r['kept_median']:.0f} | {r['standouts_fixed']:.0%} | {r['right_broken']:.0%} |"
            )
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print((out / "summary.md").read_text())


def examples(dumps: Path, dataset: str, k: int, rule: str, out: Path, n_each: int = 3, top: int = 12) -> None:
    """Literal cells where *rule* keeps a different drop from ``widest``: the first wins and losses, seeded."""
    z = np.load(dumps / f"cosines_{dataset}_k{k}.npz")
    off = z["offsets"]
    kinds: dict[str, list[str]] = {"rule wins": [], "rule loses": []}
    for i in np.random.default_rng(0).permutation(len(off) - 1):
        s = z["scores"][off[i] : off[i + 1]].astype(np.float64)
        y = z["labels"][off[i] : off[i + 1]].astype(np.int64)
        order = np.argsort(-s, kind="stable")
        ys, ss = y[order], s[order]
        counted = n_hat(s, "gauss", 4.0)
        if counted is None:
            continue
        g0, g1 = drop_count(ss[: TOP + 1], "widest"), drop_count(ss[: TOP + 1], rule, ys)
        k0, k1 = max(round(counted), g0, 1), max(round(counted), g1, 1)
        f0, f1 = at_top(ys, k0, 1.0)[2], at_top(ys, k1, 1.0)[2]
        kind = "rule wins" if f1 > f0 + 0.05 else "rule loses" if f1 < f0 - 0.05 else None
        if kind is None or len(kinds[kind]) >= n_each:
            continue
        med, sigma = bulk(s)
        cells = " ".join(f"{(ss[j] - med) / sigma:.2f}{'*' if ys[j] else ''}" for j in range(top))
        kinds[kind].append(f"| {z['category'][i]} | {int(y.sum())} | {cells} | {k0} | {k1} | {f0:.2f} | {f1:.2f} |")
        if all(len(v) >= n_each for v in kinds.values()):
            break
    lines = [f"# #4783 examples: {dataset}, K = {k}, `{rule}` against `widest`, beta 1", ""]
    for kind, rows in kinds.items():
        lines += [f"## {kind}", "", "| person | photos | top 12 (z) | widest keeps | rule keeps | F widest | F rule |",
                  "|---|---|---|---|---|---|---|", *rows, ""]  # fmt: skip
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


LABELLED_RULES = ("count", "count_gap_floor", "count_gap_floor_skip25", "count_gap_floor_skip25_b1")


def labelled(dumps: Path, out: Path) -> None:
    import pandas as pd  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from vtscore.detectors.centroid_head import centroid_cut  # noqa: PLC0415

    rows = []
    for path in sorted(dumps.glob("cosines_*_k1.npz")):
        z = np.load(path)
        off = z["offsets"]
        for i in range(len(off) - 1):
            s = z["scores"][off[i] : off[i + 1]].astype(np.float64)
            y = z["labels"][off[i] : off[i + 1]].astype(np.int64)
            corpus = np.concatenate([[1.0], s])  # the example, as the labelled collection holds it
            for rule in LABELLED_RULES:
                for beta in BETAS:
                    kept = s >= centroid_cut(corpus, rule=rule, beta=beta)
                    tp, k = int((kept & (y == 1)).sum()), int(kept.sum())
                    p = tp / k if k else 0.0
                    r = tp / int(y.sum())
                    b2 = beta * beta
                    f = 0.0 if not (p > 0 or r > 0) else (1 + b2) * p * r / (b2 * p + r)
                    rows.append({"dataset": str(z["dataset"]), "category": str(z["category"][i]), "rule": rule,
                                 "beta": beta, "kept_new": k, "fbeta": f})  # fmt: skip
        print(f"{path.name}: {len(off) - 1} cells", file=sys.stderr)
    cells = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    agg = (
        cells.assign(nothing_new=cells["kept_new"] == 0)
        .groupby(["dataset", "rule", "beta"], sort=False)
        .agg(
            people=("category", "nunique"),
            fbeta=("fbeta", "mean"),
            nothing_new=("nothing_new", "mean"),
            kept_new_median=("kept_new", "median"),
        )  # fmt: skip
        .reset_index()
        .round(4)
    )
    agg.to_csv(out / "labelled_summary.csv", index=False)
    print(agg.pivot_table(index=["dataset", "rule"], columns="beta", values="fbeta").round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("price")
    p.add_argument("--dumps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("examples")
    e.add_argument("--dumps", type=Path, required=True)
    e.add_argument("--dataset", required=True)
    e.add_argument("--k", type=int, required=True)
    e.add_argument("--rule", required=True)
    e.add_argument("--out", type=Path, required=True)
    lab = sub.add_parser("labelled")
    lab.add_argument("--dumps", type=Path, required=True)
    lab.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "price":
        price(args.dumps, args.out)
    elif args.cmd == "labelled":
        labelled(args.dumps, args.out)
    else:
        examples(args.dumps, args.dataset, args.k, args.rule, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
