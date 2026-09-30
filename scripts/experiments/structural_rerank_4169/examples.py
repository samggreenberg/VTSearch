"""Literal examples for #4169: where the match-statistic MLP loses to the inlier ranking.

For each matrix directory, replays the shared sequence at *v* votes (the top *v* of the
query crop's ranking, labelled from ground truth), and for the classes where the MLP
(``a5_mlp``) loses most AP against max-over-templates inliers (``a1_max``) prints:

* the Bad votes the MLP trained on, with their best-template inliers;
* the top of each arm's ranking of the unlabelled remainder, with ground truth,
  inliers and the MLP's probability.

    python examples.py --matrix <dir> [--matrix <dir> ...] [--v 10] [--worst 3] [--top 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fullmarks"))

import vote_curve as vc  # noqa: E402


def class_data(npz: Path) -> vc.ClassData:
    n = len(np.load(npz)["pool_ids"])
    none = np.zeros((n, 1), dtype=np.float32)
    return vc.ClassData(npz, none, none, np.zeros(1, np.float32), np.zeros(1, np.float32))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, action="append", required=True)
    ap.add_argument("--v", type=int, default=10)
    ap.add_argument("--worst", type=int, default=3)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    for matrix in args.matrix:
        print(f"# {matrix}\n")
        scored = []
        for f in sorted(matrix.glob("*.npz")):
            if f.name.startswith("vectors-"):
                continue
            cd = class_data(f)
            if not cd.positive.any():
                continue
            a0 = vc.rank("a0_exemplar", cd, [], [])
            seq = [int(i) for i in a0[: args.v]]
            goods = [i for i in seq if cd.positive[i]]
            bads = [i for i in seq if not cd.positive[i]]
            rest = set(seq)
            orders = {arm: vc.remainder(vc.rank(arm, cd, goods, bads), rest) for arm in ("a1_max", "a5_mlp")}
            if not cd.positive[orders["a1_max"]].any():
                continue
            aps = {arm: vc.average_precision(o, cd.positive) for arm, o in orders.items()}
            scored.append((aps["a5_mlp"] - aps["a1_max"], f, cd, goods, bads, orders, aps))
        scored.sort(key=lambda t: t[0])
        losers = sum(1 for d, *_ in scored if d < -0.005)
        winners = sum(1 for d, *_ in scored if d > 0.005)
        print(f"{len(scored)} classes: MLP below the inliers in {losers}, above in {winners}.\n")
        for diff, f, cd, goods, bads, orders, aps in scored[: args.worst]:
            inl, _tent, arg = cd.best([0, *cd.templates_for(goods)])
            model = vc._mlp(cd, [0, *cd.templates_for(goods)], goods, bads)
            prob = vc._mlp_scores(model, cd.stats[arg, np.arange(cd.n)]) if model is not None else np.full(cd.n, np.nan)
            print(
                f"## {f.stem}: AP inliers {aps['a1_max']:.2f}, MLP {aps['a5_mlp']:.2f} ({diff:+.2f}); "
                f"{len(goods)} Good / {len(bads)} Bad after {args.v} votes"
            )
            print("Bads trained on: " + ", ".join(f"{cd.pool_ids[b]} ({inl[b]} inl)" for b in bads))
            for arm, label in (("a1_max", "inliers"), ("a5_mlp", "MLP")):
                print(f"Top {args.top} by {label}:")
                for i in orders[arm][: args.top]:
                    mark = "POS" if cd.positive[i] else "neg"
                    print(f"  {mark} {cd.pool_ids[i]:<28} inliers {inl[i]:>4}  MLP p {prob[i]:.2f}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
