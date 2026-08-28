"""Paired difference between two grids that differ in a named way.

Two runs of the same design, one knob apart, are the cheapest measurement there
is -- the cells are already paid for.  What makes them *readable* is that the
pairing is exact: the same category, the same seed, the same click, so every
source of variance except the knob cancels within the pair rather than being
averaged over.  What makes them *wrong* is the same property abused: two grids
usually differ in more than one way (a code change landed between them, the pile
was rebuilt, the launcher's defaults moved), and a paired delta reports their
SUM while looking exactly like a measurement of one thing.

Hence ``--differs``, which is required and is echoed into the output.  It does
not check anything -- nothing here can -- but it makes the claim the number
depends on a written part of the result rather than an assumption in the reader's
head.  If you cannot fill it in with one short phrase, the comparison is not one.

Rows pair on ``(dataset, embedder, category, seed)`` at one click count, and the
report prints what *failed* to pair beside what did: a delta over 60% of the grid
is a different quantity from a delta over all of it, and the difference is
invisible in the mean.

Usage::

    python compare_grids.py --a /expscratch/$USER/scale-3156-fixed \\
        --b /expscratch/$USER/scale-3156-map --metric cost --t 150 \\
        --differs "Train/Calibrate split 0.5 -> per-space 0.3/0.5 (#3290), plus the split dither (#3286)"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import common

common.setup_env()

import pandas as pd  # noqa: E402


def _band(cat: str) -> str:
    return cat.rsplit("@", 1)[1] if "@" in str(cat) else ""


def _load(exp: Path, metric: str, t: int | None) -> tuple[pd.DataFrame, int]:
    import analyze_spikes as sp

    df, _prov = sp.load_arm(exp / "results")
    if df.empty:
        raise SystemExit(f"no rows under {exp}/results")
    if metric not in df.columns:
        raise SystemExit(f"{exp}: no column {metric!r}")
    t_use = int(df["t"].max()) if t is None else int(t)
    df = df[df["t"] == t_use]
    keep = ["dataset", "embedder", "category", "seed", metric]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise SystemExit(f"{exp}: rows lack {missing}")
    return df.loc[:, keep].copy(), t_use


def _mean_se(xs: pd.Series) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = float(xs.mean())
    if n < 2:
        return m, float("nan")
    return m, float(xs.std(ddof=1) / math.sqrt(n))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="the earlier grid (the baseline of the delta)")
    ap.add_argument("--b", required=True, help="the later grid; the delta is b - a")
    ap.add_argument("--metric", default="cost")
    ap.add_argument("--t", type=int, default=None, help="click count to compare at (default: the deepest shared)")
    ap.add_argument(
        "--differs",
        required=True,
        help="what actually differs between the two grids - printed with the result, because a paired "
        "delta reports the SUM of every difference while looking like a measurement of one",
    )
    args = ap.parse_args(argv)

    a_path, b_path = Path(args.a), Path(args.b)
    a, ta = _load(a_path, args.metric, args.t)
    b, tb = _load(b_path, args.metric, args.t)
    if ta != tb:
        raise SystemExit(f"different click counts: {a_path.name} has t={ta}, {b_path.name} has t={tb}")

    keys = ["dataset", "embedder", "category", "seed"]
    m = a.merge(b, on=keys, how="inner", suffixes=("_a", "_b"))
    m["delta"] = m[f"{args.metric}_b"] - m[f"{args.metric}_a"]
    m["band"] = m["category"].map(_band)

    print(f"A: {a_path}  ({len(a)} rows at t={ta})")
    print(f"B: {b_path}  ({len(b)} rows at t={tb})")
    print(f"differs: {args.differs}")
    print(f"metric: {args.metric}   delta = B - A   paired on {'+'.join(keys)}")
    print()

    # Unpaired rows first: a delta over part of the grid is a different quantity
    # from a delta over all of it, and nothing downstream can tell them apart.
    only_a = len(a) - len(m)
    only_b = len(b) - len(m)
    print(f"paired {len(m)} rows; {only_a} in A only, {only_b} in B only")
    if only_a or only_b:
        a_emb = set(a["embedder"].unique())
        b_emb = set(b["embedder"].unique())
        if a_emb - b_emb:
            print(f"  embedders only in A: {sorted(a_emb - b_emb)}")
        if b_emb - a_emb:
            print(f"  embedders only in B: {sorted(b_emb - a_emb)}")
        sa, sb = set(a["seed"].unique()), set(b["seed"].unique())
        if sa - sb or sb - sa:
            print(f"  seeds: A has {len(sa)}, B has {len(sb)}, shared {len(sa & sb)}")
    print()

    rows = [("ALL", m)]
    rows += [(f"{e}", g) for e, g in m.groupby("embedder")]
    print(f"{'slice':<26} {'n':>6} {'A':>8} {'B':>8} {'B - A':>16}")
    print("-" * 70)
    for name, g in rows:
        mu, se = _mean_se(g["delta"])
        print(
            f"{name:<26} {len(g):>6} {g[f'{args.metric}_a'].mean():>8.3f} "
            f"{g[f'{args.metric}_b'].mean():>8.3f} {mu:>9.3f} +- {se:.3f}"
        )
    print()
    print(f"{'embedder x band':<26} {'n':>6} {'A':>8} {'B':>8} {'B - A':>16}")
    print("-" * 70)
    for (e, bd), g in m.groupby(["embedder", "band"]):
        mu, se = _mean_se(g["delta"])
        print(
            f"{e + ' @ ' + bd:<26} {len(g):>6} {g[f'{args.metric}_a'].mean():>8.3f} "
            f"{g[f'{args.metric}_b'].mean():>8.3f} {mu:>9.3f} +- {se:.3f}"
        )
    print()
    print("A difference smaller than twice its standard error is not resolvable here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
