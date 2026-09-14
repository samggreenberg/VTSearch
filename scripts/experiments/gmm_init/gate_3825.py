#!/usr/bin/env python
"""Replay the captured fold corpus through every anchored stopping rule - the #3825 gate.

    python gate_3825.py --corpus <dir> --out <dir>/analysis3825

Same corpus as #3585 (``capture_folds_3585.py``, 84 cells of real
``fit_fold_anchored_cut`` inputs), a different thing swapped: there it was the
*initialiser*, here it is where the anchored refit is allowed to stop.  Nothing
needed re-running to ask this question, which is the whole point of capturing
arrays instead of verdicts.

Three frames.

``gate3825_cuts.csv`` is the **decision** frame: one row per (cell, case, arm)
carrying the threshold the shipped chain produces - per-fold anchored fit,
per-fold quantile, combine, realise on the final haystack, snap - and how many
medias it admits at each inclusion.  A row pair against ``baseline`` answers the
only question that decides this issue: does the admitted set move?  Unlike
#3585 there is no second estimator downstream to absorb a tolerance-level
difference, so this frame is the gate rather than a supporting table.

``gate3825_fits.csv`` is the **estimator** frame: one row per (cell, case, fold,
arm) with the refit's parameters, how many iterations it took, whether it
converged at all, and the fit's value under *both* objectives - the weighted
semi-supervised one this EM ascends and the free sample's alone.  "The
thresholds differ" does not say which fit is better; these columns do, and they
also price the two halves of a fold's fit separately, which is the measurement
that opened the issue.

``gate3825_trace.csv`` is the **monotonicity** frame: for one arm per objective,
the whole iteration-by-iteration objective trace of a sample of folds.  A
stopping rule that watches a quantity the loop does not monotonically increase
is not a convergence criterion at all, and the free-sample likelihood under an
anchored M-step has no theorem saying it is one.  Cheap to check; not checking
it is how a well-formed number gets believed.
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

#: Inclusion values the fold chain is re-cut at.  0 is the shipped default and
#: the one the gate turns on; the other two are the knob's working range, where
#: ``mid_tilt`` mixes the rate cut in and the fitted *variances* start to matter
#: - a difference invisible at the midpoint can still show up there.
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
    "n_unconverged",
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
    "fold",
    "arm",
    "n",
    "n_anchors",
    "provenance",
    "n_iter",
    "converged",
    "init_seconds",
    "total_seconds",
    "refit_seconds",
    "loglik_anchored",
    "loglik_free",
    "w_lo",
    "mu_lo",
    "var_lo",
    "w_hi",
    "mu_hi",
    "var_hi",
    "midpoint",
)

TRACE_COLUMNS = ("cell", "case", "fold", "objective", "iteration", "value")


def _meta(z) -> dict:
    if "_meta" not in z:
        return {}
    return json.loads(bytes(z["_meta"].tobytes()).decode("utf-8"))


def _cases(z) -> "dict[tuple[str, ...], dict[str, np.ndarray]]":
    """Group a fold capture's flat ``kind|style|case|field`` keys by case.

    Only ``kind == "fold"`` comes back.  The captures also carry ``"sort"``
    cases - a whole cosine-sort score array under a four-part key, and a
    separate five-part key shape for the sort-only files - and the anchored
    refit is not on that path at all, so a sort case here is not a fold with no
    anchors, it is a different question's corpus sharing a directory.  Skipping
    it is named rather than incidental: the *count* is returned so the caller
    can say how many cases it dropped and why, because a silently dropped case
    is indistinguishable from one that was never captured (#3585).  Any other
    key shape or kind raises.
    """
    out: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
    for key in z.files:
        if key == "_meta":
            continue
        parts = key.split("|")
        if len(parts) == 5:
            continue
        if len(parts) != 4:
            raise ValueError(f"unrecognised capture key {key!r}")
        kind, style, case, field = parts
        if kind == "sort":
            continue
        if kind != "fold":
            raise ValueError(f"unrecognised capture kind {kind!r} in key {key!r}")
        arrays = out.setdefault((kind, style, case), {})
        if field in arrays:
            raise ValueError(f"two captured arrays collide on {(kind, style, case)!r} at field {field!r}")
        arrays[field] = z[key]
    return out


#: The anchor mass the shipped fold path fits at, bound in :func:`main` from
#: :data:`~vtscore.training.thresholds.FOLD_ANCHOR_WEIGHT` so there is one source
#: of it.  ``fit_anchored_score_gmm``'s own default is ``ANCHOR_WEIGHT_DEFAULT``
#: = 10.0, but **nothing on the shipped path calls it bare**:
#: ``fit_fold_anchored_cut`` passes 0.3, which the 2026-08-06 anchor-mass sweep
#: picked.  The first run of this gate took the bare default in its estimator
#: frame while its decision frame went through the shipped function, so the two
#: frames described fits at different anchor masses - nothing crashed, and the
#: iteration counts were simply about a configuration nobody runs.
FOLD_WEIGHT = 0.3


def _fold_inputs(arrays: "dict[str, np.ndarray]"):
    """``(fold haystacks, per-fold (scores, labels) orderings, final scores)``."""
    hays = [arrays[k] for k in sorted(arrays) if k.startswith("hay")]
    n_ord = len([k for k in arrays if k.startswith("anchor")])
    orderings = [(arrays[f"anchor{i}"].tolist(), arrays[f"label{i}"].tolist()) for i in range(n_ord)]
    return hays, orderings, arrays["final"]


def _trace(x, a_lo, a_hi, init, weight, anchored_objective: bool, max_iter: int = 400) -> "list[float]":
    """The objective after each of *max_iter* iterations, by running one at a time.

    Restarting from the previous iteration's parameters reproduces the loop
    exactly - EM is a fixed point map with no state between iterations beyond
    the parameters - so this is the real trajectory, not a re-derivation of it.
    """
    from vtscore.training.thresholds.gmm import _anchored_em

    values: list[float] = []
    cur = init
    prev = None
    for _ in range(max_iter):
        stats: dict[str, float] = {}
        # One iteration, then read the objective the NEXT iteration would see.
        nxt = _anchored_em(x, a_lo, a_hi, cur, weight, 1, 1e-30, None, stats, anchored_objective)
        if nxt is None:
            break
        probe: dict[str, float] = {}
        _anchored_em(x, a_lo, a_hi, nxt, weight, 1, 1e-30, 1e30, probe, anchored_objective)
        v = probe.get("loglik")
        if v is None:
            break
        values.append(v)
        if prev is not None and abs(v - prev) < 1e-12:
            break
        prev = v
        cur = nxt
    return values


def _run_cell(path: Path, cut_rows: list, fit_rows: list, trace_rows: list, trace_budget: list) -> None:
    from vtscore.training.thresholds import (
        fit_anchored_score_gmm,
        fit_fold_anchored_cut,
        fit_score_gmm,
        gmm_fit_array,
        scored_ordering,
    )
    from vtscore.utils.scores import scored_only

    z = np.load(path)
    meta = _meta(z)
    base = {
        "cell": path.stem,
        "dataset": meta.get("dataset", ""),
        "embedder": meta.get("embedder", ""),
        "category": meta.get("category", ""),
        "seed": meta.get("seed", ""),
    }
    cases = _cases(z)
    n_sort = len({tuple(k.split("|")[:3]) for k in z.files if k != "_meta" and k.split("|")[0] == "sort"})
    print(f"  {path.name}: {len(cases)} fold cases ({n_sort} sort cases skipped - not this path)", flush=True)
    for case_id, arrays in sorted(cases.items()):
        kind, style, case = case_id
        row_base = {**base, "style": style, "kind": kind, "case": case}
        hays, orderings, final = _fold_inputs(arrays)
        if not hays or not orderings:
            continue
        for arm, (stop, _desc) in A.ARMS.items():
            with A.swap_anchored_stop(stop):
                # 1. the decision frame: the whole shipped chain, timed.
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
                            "n_unconverged": "",
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
                            "n_unconverged": cut.n_unconverged,
                            "provenance": cut.provenance,
                            "seconds": f"{seconds:.6f}",
                            **{f"threshold_i{v}": repr(thresholds[v]) for v in INCLUSIONS},
                            **{
                                f"n_admitted_i{v}": int(np.count_nonzero(cut.final_haystack >= thresholds[v]))
                                for v in INCLUSIONS
                            },
                        }
                    )

                # 2. the estimator frame: each fold's refit on its own, with the
                #    init timed separately - the split the issue's headline
                #    ("~90% of a fold's fit") is a claim about.
                for i, hay in enumerate(hays):
                    if i >= len(orderings):
                        continue
                    a_s, a_l = scored_ordering(orderings[i])
                    arr = gmm_fit_array(scored_only(hay))
                    t0 = time.perf_counter()
                    fit_score_gmm(arr)
                    init_seconds = time.perf_counter() - t0
                    stats: dict[str, float] = {}
                    t0 = time.perf_counter()
                    fit, provenance = fit_anchored_score_gmm(arr, a_s, a_l, anchor_weight=FOLD_WEIGHT, stats=stats)
                    total_seconds = time.perf_counter() - t0
                    fit_rows.append(
                        {
                            **row_base,
                            "fold": i,
                            "arm": arm,
                            "n": int(arr.size),
                            "n_anchors": int(np.size(a_s)),
                            "provenance": provenance,
                            "n_iter": int(stats.get("n_iter", 0)),
                            "converged": int(stats.get("converged", 0)),
                            "init_seconds": f"{init_seconds:.6f}",
                            "total_seconds": f"{total_seconds:.6f}",
                            "refit_seconds": f"{max(0.0, total_seconds - init_seconds):.6f}",
                            "loglik_anchored": ""
                            if fit is None
                            else f"{A.objective_value(fit, arr, a_s, a_l, True, FOLD_WEIGHT):.10f}",
                            "loglik_free": ""
                            if fit is None
                            else f"{A.objective_value(fit, arr, a_s, a_l, False, FOLD_WEIGHT):.10f}",
                            "w_lo": "" if fit is None else repr(fit.w_lo),
                            "mu_lo": "" if fit is None else repr(fit.mu_lo),
                            "var_lo": "" if fit is None else repr(fit.var_lo),
                            "w_hi": "" if fit is None else repr(fit.w_hi),
                            "mu_hi": "" if fit is None else repr(fit.mu_hi),
                            "var_hi": "" if fit is None else repr(fit.var_hi),
                            "midpoint": "" if fit is None else repr(fit.midpoint()),
                        }
                    )

        # 3. the monotonicity frame, on a budget: the traces are only worth
        #    having for a sample, and they cost 400 restarts a fold.
        if trace_budget[0] > 0:
            trace_budget[0] -= 1
            for i, hay in enumerate(hays[:1]):
                if i >= len(orderings):
                    continue
                a_s, a_l = scored_ordering(orderings[i])
                arr = gmm_fit_array(scored_only(hay))
                init = fit_score_gmm(arr)
                if init is None:
                    continue
                a = np.asarray(a_s, dtype=np.float64)
                z_lab = np.asarray(a_l, dtype=np.float64)
                a_hi, a_lo = a[z_lab == 1.0], a[z_lab != 1.0]

                for objective, flag in (("anchored", True), ("free", False)):
                    for k, v in enumerate(_trace(arr, a_lo, a_hi, init, FOLD_WEIGHT, flag), 1):
                        trace_rows.append(
                            {
                                "cell": path.stem,
                                "case": case,
                                "fold": i,
                                "objective": objective,
                                "iteration": k,
                                "value": repr(float(v)),
                            }
                        )
    z.close()


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="directory of cell_*.npz captures")
    ap.add_argument("--out", required=True, help="directory for the gate frames")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cells (a smoke run)")
    ap.add_argument("--traces", type=int, default=8, help="how many cells contribute an objective trace")
    args = ap.parse_args(list(argv) if argv is not None else None)

    A.assert_swap_reaches_the_refit()
    print("the anchored-stop swap reaches the refit", flush=True)

    global FOLD_WEIGHT
    from vtscore.training.thresholds import FOLD_ANCHOR_WEIGHT

    FOLD_WEIGHT = FOLD_ANCHOR_WEIGHT
    print(f"fitting at the shipped fold anchor weight, kappa = {FOLD_WEIGHT}", flush=True)

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
    trace_rows: list[dict] = []
    trace_budget = [args.traces]
    failed = 0
    for i, path in enumerate(corpus, 1):
        t0 = time.monotonic()
        try:
            _run_cell(path, cut_rows, fit_rows, trace_rows, trace_budget)
        except Exception as exc:
            failed += 1
            print(f"FAILED {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"[{i}/{len(corpus)}] {path.name}: {time.monotonic() - t0:.1f}s", flush=True)

    for name, rows, cols in (
        ("gate3825_cuts.csv", cut_rows, CUT_COLUMNS),
        ("gate3825_fits.csv", fit_rows, FIT_COLUMNS),
        ("gate3825_trace.csv", trace_rows, TRACE_COLUMNS),
    ):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out / name}: {len(rows)} rows")
    # The whole identity, not just the case name: a cell captures the same case
    # under each patch style, so keying on (cell, case) undercounts by the
    # number of styles and reports a shortfall that is not one.
    n_cases = len({(r["cell"], r["style"], r["kind"], r["case"]) for r in cut_rows})
    print(f"cells: {len(corpus)} attempted, {failed} failed; {n_cases} fold cases x {len(A.ARMS)} arms")
    expected = n_cases * len(A.ARMS)
    if len(cut_rows) != expected:
        print(
            f"WARNING: {len(cut_rows)} cut rows for {n_cases} cases x {len(A.ARMS)} arms = {expected}", file=sys.stderr
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
