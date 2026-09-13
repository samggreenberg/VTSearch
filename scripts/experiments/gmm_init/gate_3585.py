#!/usr/bin/env python
"""Replay a captured corpus through every candidate fit - the #3585 gate.

    python gate_3585.py --corpus <dir> --out <dir>/analysis

Two frames, because the issue asks two different questions of the same swap.

``gate_cuts.csv`` is the **decision** frame: one row per (cell, case, arm) with
the threshold the whole production chain produces and how many medias it
admits.  For a fold case that chain is the shipped one - per-fold anchored fit,
per-fold quantile, combine, realise on the final haystack, snap - so a row pair
answers "did the admitted set change at all", which is the number the issue says
the decision turns on.  For a sort case it is ``calculate_gmm_threshold``, the
cosine/text sort's own cut.

``gate_fits.csv`` is the **estimator** frame: one row per (cell, case, sample,
arm) with the fitted parameters and the mean log-likelihood of that fit on that
sample.  It exists because "the thresholds differ" does not say which one is
right, and both arms are maximising the same objective: a candidate that lands
higher on it has moved the cut toward a better fit of the data rather than
merely somewhere else.

Every fit is timed here too (one call, not a min-of-k), so the frames carry a
cost distribution over real inputs.  ``bench_3585.py`` is where the headline
per-call numbers come from.
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

#: Inclusion values the fold chain is re-cut at.  0 is the shipped default and
#: the one the gate turns on; the other two are the knob's working range, where
#: the ``mid_tilt`` rule mixes the rate cut in - a fit difference that is
#: invisible at the midpoint can still show up once the variances are read.
INCLUSIONS = (-3, 0, 3)

CUT_COLUMNS = (
    "cell",
    "dataset",
    "embedder",
    "category",
    "seed",
    "style",
    "kind",
    "case",
    "arm",
    "n",
    "n_folds",
    "n_anchored",
    "provenance",
    "seconds",
    *[f"threshold_i{v}" for v in INCLUSIONS],
    *[f"n_admitted_i{v}" for v in INCLUSIONS],
)

FIT_COLUMNS = (
    "cell",
    "dataset",
    "embedder",
    "category",
    "seed",
    "style",
    "kind",
    "case",
    "sample",
    "arm",
    "n",
    "seconds",
    "loglik",
    "w_lo",
    "mu_lo",
    "var_lo",
    "w_hi",
    "mu_hi",
    "var_hi",
    "midpoint",
)


def _meta(z) -> dict:
    """The cell identity written into the capture, or an empty dict."""
    if "_meta" not in z:
        return {}
    return json.loads(bytes(z["_meta"].tobytes()).decode("utf-8"))


def _cases(z) -> "dict[tuple[str, str, str], dict[str, np.ndarray]]":
    """Group an npz's flat keys back into ``(kind, style, case) -> arrays``."""
    out: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for key in z.files:
        if key == "_meta":
            continue
        kind, style, case, field = key.split("|")
        out.setdefault((kind, style, case), {})[field] = z[key]
    return out


def _fit_row(base: dict, sample: str, arm: str, x: np.ndarray, fit, seconds: float) -> dict:
    return {
        **base,
        "sample": sample,
        "arm": arm,
        "n": int(x.size),
        "seconds": f"{seconds:.6f}",
        "loglik": "" if fit is None else f"{A.mean_loglik(fit, x):.10f}",
        "w_lo": "" if fit is None else repr(fit.w_lo),
        "mu_lo": "" if fit is None else repr(fit.mu_lo),
        "var_lo": "" if fit is None else repr(fit.var_lo),
        "w_hi": "" if fit is None else repr(fit.w_hi),
        "mu_hi": "" if fit is None else repr(fit.mu_hi),
        "var_hi": "" if fit is None else repr(fit.var_hi),
        "midpoint": "" if fit is None else repr(fit.midpoint()),
    }


def _run_cell(path: Path, cut_rows: list, fit_rows: list) -> None:
    """Every case in one captured cell, under every arm."""
    from vtscore.training.thresholds import calculate_gmm_threshold, fit_fold_anchored_cut

    z = np.load(path)
    meta = _meta(z)
    base = {
        "cell": path.stem,
        "dataset": meta.get("dataset", ""),
        "embedder": meta.get("embedder", ""),
        "category": meta.get("category", ""),
        "seed": meta.get("seed", ""),
    }
    for (kind, style, case), arrays in sorted(_cases(z).items()):
        row_base = {**base, "style": style, "kind": kind, "case": case}
        for arm, (fit_fn, _desc) in A.ARMS.items():
            with A.swap_fit(fit_fn):
                if kind == "sort":
                    x = arrays["scores"]
                    t0 = time.perf_counter()
                    threshold = calculate_gmm_threshold(x.tolist())
                    seconds = time.perf_counter() - t0
                    admitted = int(np.count_nonzero(x >= threshold))
                    cut_rows.append(
                        {
                            **row_base,
                            "arm": arm,
                            "n": int(x.size),
                            "n_folds": 0,
                            "n_anchored": 0,
                            "provenance": "sort",
                            "seconds": f"{seconds:.6f}",
                            **{f"threshold_i{v}": repr(threshold) for v in INCLUSIONS},
                            **{f"n_admitted_i{v}": admitted for v in INCLUSIONS},
                        }
                    )
                    t0 = time.perf_counter()
                    fit = fit_fn(x)
                    fit_rows.append(_fit_row(row_base, "sort", arm, x, fit, time.perf_counter() - t0))
                    continue

                hays = [arrays[k] for k in sorted(arrays) if k.startswith("hay")]
                orderings = [
                    (arrays[f"anchor{i}"].tolist(), arrays[f"label{i}"].tolist())
                    for i in range(len([k for k in arrays if k.startswith("anchor")]))
                ]
                final = arrays["final"]
                t0 = time.perf_counter()
                cut = fit_fold_anchored_cut(hays, orderings, final.tolist())
                seconds = time.perf_counter() - t0
                if cut is None:
                    cut_rows.append(
                        {
                            **row_base,
                            "arm": arm,
                            "n": int(final.size),
                            "n_folds": len(hays),
                            "n_anchored": 0,
                            "provenance": "no_cut",
                            "seconds": f"{seconds:.6f}",
                            **{f"threshold_i{v}": "" for v in INCLUSIONS},
                            **{f"n_admitted_i{v}": "" for v in INCLUSIONS},
                        }
                    )
                else:
                    thresholds = {v: cut.threshold_at(v) for v in INCLUSIONS}
                    cut_rows.append(
                        {
                            **row_base,
                            "arm": arm,
                            "n": int(final.size),
                            "n_folds": len(cut.fits),
                            "n_anchored": cut.n_anchored,
                            "provenance": cut.provenance,
                            "seconds": f"{seconds:.6f}",
                            **{f"threshold_i{v}": repr(thresholds[v]) for v in INCLUSIONS},
                            **{
                                f"n_admitted_i{v}": int(np.count_nonzero(cut.final_haystack >= thresholds[v]))
                                for v in INCLUSIONS
                            },
                        }
                    )
                # The unanchored fit of each fold haystack on its own - the
                # estimator frame.  This is the fit the arm actually changes;
                # everything above is what the change propagates into.
                for i, hay in enumerate(hays):
                    t0 = time.perf_counter()
                    fit = fit_fn(hay)
                    fit_rows.append(_fit_row(row_base, f"hay{i}", arm, hay, fit, time.perf_counter() - t0))
    z.close()


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="directory of cell_*.npz captures")
    ap.add_argument("--out", required=True, help="directory for gate_cuts.csv / gate_fits.csv")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cells (a smoke run)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    corpus = sorted(Path(args.corpus).glob("cell_*.npz"))
    if args.limit:
        corpus = corpus[: args.limit]
    if not corpus:
        print(f"no captures under {args.corpus}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cut_rows: list[dict] = []
    fit_rows: list[dict] = []
    for i, path in enumerate(corpus, 1):
        t0 = time.monotonic()
        try:
            _run_cell(path, cut_rows, fit_rows)
        except Exception as exc:
            # Named and counted, never skipped in silence: a corpus analysed
            # 80/84 while reporting neither number is how a disk incident
            # becomes a verdict.
            print(f"FAILED {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"[{i}/{len(corpus)}] {path.name}: {time.monotonic() - t0:.1f}s", flush=True)

    for name, rows, cols in (
        ("gate_cuts.csv", cut_rows, CUT_COLUMNS),
        ("gate_fits.csv", fit_rows, FIT_COLUMNS),
    ):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out / name}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
