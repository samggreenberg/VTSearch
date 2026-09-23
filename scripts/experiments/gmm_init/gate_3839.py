#!/usr/bin/env python
"""Replay the #3585 fold corpus through the #3839 arms - one shard of it.

    python gate_3839.py --corpus <dir> --out <dir> [--shard i/n]

The #3825 gate, unchanged in what it measures, with ``arms_3839`` installed in
place of ``arms_3825``: the same decision frame (the shipped chain's threshold
and admitted set per case and arm), the same estimator frame (each fold's
refit: iterations, convergence, the fit under both objectives), no trace frame
(#3825 settled monotonicity).  ``gate_3825._run_cell`` reads its arms off a
module global, so pointing that global at this study's module is the whole of
the change - which is what keeps the two gates' columns comparable.

Sharded because one arm here (``limit``) runs up to 20,000 iterations on the
folds #3839 is about, and the shipped chain fits every fold twice (decision
frame, then estimator frame).  Shards write ``gate3839_{cuts,fits}.<i>.csv``;
``analyze_3839.py`` concatenates them and refuses a set with a shard missing.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import arms_3839 as A  # noqa: E402
import gate_3825 as G25  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default=os.environ.get("GATE_SHARD", ""), help="i/n: this process's slice of the cells")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cells (a smoke run)")
    ap.add_argument("--arms", default="", help="comma list: run only these arms (smoke runs)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    # Install this study's arms where the #3825 gate looks for them.
    arms = dict(A.ARMS)
    if args.arms:
        keep = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in keep if a not in arms]
        if unknown:
            print(f"unknown arms {unknown}", file=sys.stderr)
            return 2
        arms = {a: arms[a] for a in keep}
    A.ARMS = arms
    G25.A = A

    A.assert_swap_reaches_the_refit()
    from vtscore.training.thresholds import FOLD_ANCHOR_WEIGHT

    G25.FOLD_WEIGHT = FOLD_ANCHOR_WEIGHT
    print(f"arms: {list(arms)}; kappa = {FOLD_ANCHOR_WEIGHT}", flush=True)

    corpus = sorted(Path(args.corpus).glob("cell_*.npz"))
    tag = "all"
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        corpus = corpus[i::n]
        tag = str(i)
    if args.limit:
        corpus = corpus[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cut_rows: list[dict] = []
    fit_rows: list[dict] = []
    failed = 0
    for k, path in enumerate(corpus, 1):
        t0 = time.monotonic()
        try:
            G25._run_cell(path, cut_rows, fit_rows, [], [0])
        except Exception as exc:
            failed += 1
            print(f"FAILED {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"[{k}/{len(corpus)}] {path.name}: {time.monotonic() - t0:.1f}s", flush=True)

    for name, rows, cols in (
        (f"gate3839_cuts.{tag}.csv", cut_rows, G25.CUT_COLUMNS),
        (f"gate3839_fits.{tag}.csv", fit_rows, G25.FIT_COLUMNS),
    ):
        tmp = out / (name + ".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols))
            w.writeheader()
            w.writerows(rows)
        tmp.rename(out / name)
        print(f"wrote {out / name}: {len(rows)} rows")
    print(f"cells: {len(corpus)} attempted, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
