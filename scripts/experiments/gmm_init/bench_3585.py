#!/usr/bin/env python
"""Per-call cost of each candidate fit, on real score shapes (#3585).

    python bench_3585.py --corpus <dir> --out <dir>/bench.csv

``gate_3585.py`` already times every fit it runs, but each of those is a single
call inside a loop doing other work, which is the wrong instrument for a
headline number.  This one takes a stratified sample of the same corpus and
times each arm **min-of-k** on it, at the sample's own size and at the sizes the
app actually fits on.

WHY THE RESAMPLED SIZES ARE HERE, AND WHAT THEY ARE NOT.  The pile's datasets
are 838-4952 medias; a GUI Find fits on ~250k scores subsampled to
``_GMM_MAX_SAMPLES`` = 50k, and that 50k row is the one the issue's cost
argument rests on.  There is no real 50k haystack in the corpus, so the larger
sizes are a **bootstrap resample of a real one** - the shape (bimodality,
skew, saturation) is preserved and the granularity is not.  Iteration count
follows the shape, so the projection is honest about the thing that drives
cost; it is still a projection, and rows carry ``resampled=1`` so no table can
quote it as a measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import arms_3585 as A  # noqa: E402

#: Sizes to price.  The first entry is the sample's own size; the rest are
#: bootstrap resamples of it, at the sizes the app fits on.
RESAMPLE_SIZES = (20_000, 50_000)

COLUMNS = (
    "sample",
    "dataset",
    "embedder",
    "style",
    "kind",
    "case",
    "n",
    "resampled",
    "arm",
    "seconds",
    "midpoint",
    "loglik",
)


def _pick(corpus: Path, per_cell: int, rng: np.random.Generator) -> "list[tuple[dict, np.ndarray]]":
    """A stratified sample of arrays: *per_cell* from each captured cell."""
    picked: list[tuple[dict, np.ndarray]] = []
    for path in sorted(corpus.glob("cell_*.npz")):
        z = np.load(path)
        meta = json.loads(bytes(z["_meta"].tobytes()).decode("utf-8")) if "_meta" in z else {}
        keys = [k for k in z.files if k != "_meta" and (k.endswith("|final") or k.endswith("|scores"))]
        if not keys:
            z.close()
            continue
        for key in rng.choice(keys, size=min(per_cell, len(keys)), replace=False):
            kind, style, case, _field = str(key).split("|")
            picked.append(
                (
                    {
                        "sample": f"{path.stem}:{key}",
                        "dataset": meta.get("dataset", ""),
                        "embedder": meta.get("embedder", ""),
                        "style": style,
                        "kind": kind,
                        "case": case,
                    },
                    np.asarray(z[str(key)], dtype=np.float64),
                )
            )
        z.close()
    return picked


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=2, help="arrays sampled from each captured cell")
    ap.add_argument("--reps", type=int, default=5, help="repeats per timing (the minimum is reported)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    rng = np.random.default_rng(3585)
    samples = _pick(Path(args.corpus), args.per_cell, rng)
    if not samples:
        print(f"no arrays under {args.corpus}", file=sys.stderr)
        return 1
    print(f"{len(samples)} arrays x {len(A.ARMS)} arms x {1 + len(RESAMPLE_SIZES)} sizes", flush=True)

    rows: list[dict] = []
    for i, (meta, x) in enumerate(samples, 1):
        for size in (None, *RESAMPLE_SIZES):
            xx = x if size is None else rng.choice(x, size=size, replace=True)
            for arm, (fit_fn, _desc) in A.ARMS.items():
                fit_fn(xx)  # warm: the first sklearn call in a process pays its import
                best = float("inf")
                fit = None
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    fit = fit_fn(xx)
                    best = min(best, time.perf_counter() - t0)
                rows.append(
                    {
                        **meta,
                        "n": int(xx.size),
                        "resampled": 0 if size is None else 1,
                        "arm": arm,
                        "seconds": f"{best:.6f}",
                        "midpoint": "" if fit is None else repr(fit.midpoint()),
                        "loglik": "" if fit is None else f"{A.mean_loglik(fit, xx):.10f}",
                    }
                )
        if i % 10 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
