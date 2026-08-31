#!/usr/bin/env python
"""Capture one cell's score samples and labels for the #3329 worked-cell figure.

    python worked_cell_3329.py --index 0 --out <dir>/worked_0.npz

WHY THIS EXISTS.  The `__fitq` frame carries the fitted PARAMETERS but not the
points, and the figure #3329 actually asks for - the score histogram with the
fitted components overlaid and the TRUE CLASSES coloured underneath - needs the
points.  Widening 23 064 rows with a histogram summary to draw two panels would
be the wrong trade, so this re-runs a single cell with a recording wrapper
around ``_fit_quality_rows`` and writes only the arrays that wrapper saw.

It changes nothing in the eval tier: the wrapper calls straight through and
returns the real rows, so the cell it runs is the cell the array ran.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

#: Clicks to capture.  Four panels: the first trainable fit, one early, one
#: mid-run, and the horizon - enough to watch the fit move without a wall of
#: near-identical histograms.
CHECKPOINTS = (5, 20, 50, 100)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, required=True, help="cell index, as `launch_fitq_3329.sh list` prints it")
    ap.add_argument("--out", required=True, help="npz to write")
    ap.add_argument("--outdir", default=None, help="cell CSV dir (default: a throwaway beside --out)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    from vtscore.eval import voting_iterations as vi

    import run_cells  # noqa: PLC0415

    store: dict[str, np.ndarray] = {}
    original = vi._fit_quality_rows

    def recording(base_row, safe_cut, sim_scores_by_geometry, sim_labels_by_geometry, threshold):
        rows = original(base_row, safe_cut, sim_scores_by_geometry, sim_labels_by_geometry, threshold)
        t = int(base_row.get("t", -1))
        if t in CHECKPOINTS:
            style = str(base_row.get("style", "?"))
            for geometry, scores in sim_scores_by_geometry.items():
                if scores is None:
                    continue
                labels = sim_labels_by_geometry.get(geometry)
                if labels is None:
                    continue
                key = f"{style}|sim:{geometry}|{t}"
                store[f"{key}|scores"] = np.asarray(scores, dtype=np.float64).ravel()
                store[f"{key}|labels"] = np.asarray(labels, dtype=np.float64).ravel()
            # The fitted parameters that go over the histogram, taken from the
            # rows themselves so the figure cannot drift from the frame.
            for r in rows:
                key = f"{style}|{r['scope']}|{t}"
                store[f"{key}|fit"] = np.array(
                    [r["fq_w_lo"], r["fq_mu_lo"], r["fq_var_lo"], r["fq_w_hi"], r["fq_mu_hi"], r["fq_var_hi"]],
                    dtype=np.float64,
                )
                store[f"{key}|cut"] = np.array([float(threshold)], dtype=np.float64)
        return rows

    vi._fit_quality_rows = recording
    try:
        outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "_worked_cells")
        os.makedirs(outdir, exist_ok=True)
        rc = run_cells.main(["--index", str(args.index), "--outdir", outdir])
    finally:
        vi._fit_quality_rows = original
    if rc != 0:
        return rc

    if not store:
        print("captured nothing - did the cell reach a checkpoint?", file=sys.stderr)
        return 1
    np.savez_compressed(args.out, **store)
    keys = sorted({k.rsplit("|", 1)[0] for k in store})
    print(f"wrote {args.out}: {len(keys)} (style, scope, t) captures")
    for k in keys:
        print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
