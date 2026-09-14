#!/usr/bin/env python
"""Per-call cost of each anchored stopping rule, on real fold shapes (#3825).

    python bench_3825.py --corpus <dir> --out <dir>/bench3825.csv

``gate_3825.py`` already times every fit it runs, but each of those is one call
inside a loop doing other work - the right instrument for a distribution over
real inputs, the wrong one for a headline.  This one takes a stratified sample
of the same corpus and times each arm **min-of-k** on it: the unanchored init
alone, the anchored fit as a whole, and the difference between them, which is
the split the issue is about.

**Both halves are timed in the same process on the same array**, because "the
refit is 90% of a fold's fit" is a claim about a ratio and a ratio measured
across two processes is a claim about two machines.  ``init_seconds`` is a real
min-of-k of :func:`fit_score_gmm`, not a subtraction of averages.

WHY THE RESAMPLED SIZES ARE HERE, AND WHAT THEY ARE NOT.  The pile's datasets
are 838-4952 medias, so a fold haystack here is ~2k scores; a detector trained
on a GUI-scale dataset fits on up to ``_GMM_MAX_SAMPLES`` = 50k.  There is no
real 50k fold haystack in the corpus, so the larger sizes are a **bootstrap
resample of a real one** with the anchors held fixed - the shape (bimodality,
skew, saturation) and the anchor mass ratio are what drive the iteration count,
and those are preserved; the granularity is not.  Rows carry ``resampled=1`` so
no table can quote one as a measurement.
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

import arms_3825 as A  # noqa: E402

#: Sizes to price the haystack at.  The first entry is the sample's own size;
#: the rest are bootstrap resamples of it.
RESAMPLE_SIZES = (20_000, 50_000)

COLUMNS = (
    "sample",
    "dataset",
    "embedder",
    "style",
    "case",
    "fold",
    "n",
    "n_anchors",
    "resampled",
    "arm",
    "init_seconds",
    "total_seconds",
    "refit_seconds",
    "n_iter",
    "converged",
    "midpoint",
)


def _pick(corpus: Path, per_cell: int, rng: np.random.Generator) -> list:
    """A stratified sample of ``(identity, haystack, anchor scores, labels)``."""
    picked: list = []
    for path in sorted(corpus.glob("cell_*.npz")):
        z = np.load(path)
        meta = json.loads(bytes(z["_meta"].tobytes()).decode("utf-8")) if "_meta" in z else {}
        cases: dict = {}
        for key in z.files:
            if key == "_meta":
                continue
            parts = key.split("|")
            if len(parts) != 4 or parts[0] != "fold":
                continue
            cases.setdefault((parts[1], parts[2]), {})[parts[3]] = z[key]
        keys = sorted(cases)
        if not keys:
            z.close()
            continue
        for idx in rng.choice(len(keys), size=min(per_cell, len(keys)), replace=False):
            style, case = keys[int(idx)]
            arrays = cases[(style, case)]
            n_folds = len([k for k in arrays if k.startswith("hay")])
            for fold in range(min(1, n_folds)):
                if f"anchor{fold}" not in arrays:
                    continue
                picked.append(
                    (
                        {
                            "sample": f"{path.stem}:{style}:{case}",
                            "dataset": meta.get("dataset", ""),
                            "embedder": meta.get("embedder", ""),
                            "style": style,
                            "case": case,
                            "fold": fold,
                        },
                        np.asarray(arrays[f"hay{fold}"], dtype=np.float64),
                        np.asarray(arrays[f"anchor{fold}"], dtype=np.float64),
                        np.asarray(arrays[f"label{fold}"], dtype=np.float64),
                    )
                )
        z.close()
    return picked


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=2, help="cases sampled from each captured cell")
    ap.add_argument("--reps", type=int, default=5, help="repeats per timing (the minimum is reported)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    from vtscore.training.thresholds import (
        FOLD_ANCHOR_WEIGHT,
        fit_anchored_score_gmm,
        fit_score_gmm,
        gmm_fit_array,
        scored_ordering,
    )
    from vtscore.utils.scores import scored_only

    rng = np.random.default_rng(3825)
    samples = _pick(Path(args.corpus), args.per_cell, rng)
    if not samples:
        print(f"no fold arrays under {args.corpus}", file=sys.stderr)
        return 1
    print(f"{len(samples)} folds x {len(A.ARMS)} arms x {1 + len(RESAMPLE_SIZES)} sizes", flush=True)

    rows: list[dict] = []
    for i, (meta, hay, a_raw, l_raw) in enumerate(samples, 1):
        a_s, a_l = scored_ordering((a_raw.tolist(), l_raw.tolist()))
        if not len(a_s):
            continue
        base = gmm_fit_array(scored_only(hay))
        for size in (None, *RESAMPLE_SIZES):
            x = base if size is None else rng.choice(base, size=size, replace=True)
            init_best = float("inf")
            for _ in range(args.reps):
                t0 = time.perf_counter()
                fit_score_gmm(x)
                init_best = min(init_best, time.perf_counter() - t0)
            for arm, (stop, _desc) in A.ARMS.items():
                with A.swap_anchored_stop(stop):
                    stats: dict[str, float] = {}
                    # At the SHIPPED anchor mass, not ``fit_anchored_score_gmm``'s
                    # bare default: nothing on the shipped path calls it bare, and
                    # the iteration count - which is the whole cost - depends on it.
                    fit_anchored_score_gmm(x, a_s, a_l, anchor_weight=FOLD_ANCHOR_WEIGHT, stats=stats)  # warm
                    best = float("inf")
                    fit = None
                    for _ in range(args.reps):
                        stats = {}
                        t0 = time.perf_counter()
                        fit, _prov = fit_anchored_score_gmm(x, a_s, a_l, anchor_weight=FOLD_ANCHOR_WEIGHT, stats=stats)
                        best = min(best, time.perf_counter() - t0)
                rows.append(
                    {
                        **meta,
                        "n": int(x.size),
                        "n_anchors": int(np.size(a_s)),
                        "resampled": 0 if size is None else 1,
                        "arm": arm,
                        "init_seconds": f"{init_best:.6f}",
                        "total_seconds": f"{best:.6f}",
                        "refit_seconds": f"{max(0.0, best - init_best):.6f}",
                        "n_iter": int(stats.get("n_iter", 0)),
                        "converged": int(stats.get("converged", 0)),
                        "midpoint": "" if fit is None else repr(fit.midpoint()),
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
