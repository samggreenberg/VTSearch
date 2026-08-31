#!/usr/bin/env python
"""Does the Coverage Atlas's domain-shift guard fire when it should? (#3329, part B)

    python domain_shift_3329.py --embedder siglip --datasets a,b,c --out DIR

`structure_fits_3329.py` asks whether the guard's NULL holds - whether in-domain
p-values are uniform, as ``domain_shift_report``'s docstring claims.  This asks
the other half, which no amount of null-checking answers: **what does the guard
actually do when pointed at a different domain?**

Every ordered pair of datasets under one embedder, atlas built on the row and
queried with the column.  The diagonal is the null (a held-out split of the
atlas's own data, so the guard should say "not shifted"); the off-diagonal is
the alternative.  A guard whose diagonal is fine and whose off-diagonal never
fires is a guard that cannot do its job, and neither cell alone would say so.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()

from structure_fits_3329 import HOLDOUT_FRACTION, ks_uniform, load_matrix  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--datasets", required=True, help="comma-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-query", type=int, default=4000, help="cap on query rows per pair")
    args = ap.parse_args(list(argv) if argv is not None else None)

    import pandas as pd

    from vtscore.state.coverage_atlas import CoverageAtlas, auto_max_depth, domain_shift_report

    names = [d for d in args.datasets.split(",") if d]
    mats: dict[str, np.ndarray] = {}
    ids: dict[str, list[int]] = {}
    for d in names:
        m, i, _ = load_matrix(d, args.embedder)
        mats[d], ids[d] = m, i
        common.log(f"loaded {d}: {m.shape}")

    rng = np.random.default_rng(args.seed)
    atlases, holdouts = {}, {}
    for d in names:
        perm = rng.permutation(mats[d].shape[0])
        n_hold = max(1, int(round(HOLDOUT_FRACTION * mats[d].shape[0])))
        hold, build = perm[:n_hold], perm[n_hold:]
        holdouts[d] = mats[d][hold]
        atlases[d] = CoverageAtlas(
            {int(ids[d][j]): mats[d][j] for j in build}, k=3, max_depth=auto_max_depth(len(build), k=3)
        )
        common.log(f"  atlas[{d}]: {atlases[d].total_nodes} nodes, depth {atlases[d].depth()}")

    rows = []
    for build_ds in names:
        atlas = atlases[build_ds]
        for query_ds in names:
            # The diagonal uses the HELD-OUT split, never the build points: an
            # atlas scored against its own build set is the one comparison that
            # cannot fail, and reporting it as the null would be circular.
            q = holdouts[query_ds] if query_ds == build_ds else mats[query_ds]
            if q.shape[0] > args.max_query:
                q = q[np.random.default_rng(args.seed + 7).choice(q.shape[0], args.max_query, replace=False)]
            rep = domain_shift_report(atlas, q)
            p = np.asarray(atlas.typicality_pvalues(q), dtype=np.float64)
            rows.append(
                {
                    "embedder": args.embedder,
                    "build_dataset": build_ds,
                    "query_dataset": query_ds,
                    "is_self": build_ds == query_ds,
                    "n_items": rep["n_items"],
                    "frac_atypical": rep["frac_atypical"],
                    "z_score": rep["z_score"],
                    "median_pvalue": rep["median_pvalue"],
                    "shifted": bool(rep["shifted"]),
                    "ks_uniform": ks_uniform(p),
                    "mean_pvalue": float(p.mean()),
                    "sd_pvalue": float(p.std()),
                }
            )
            common.log(
                f"  {build_ds} <- {query_ds}: z={rep['z_score']:+.1f} "
                f"frac={rep['frac_atypical']:.3f} shifted={rep['shifted']}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"domainshift_{args.embedder}.csv", index=False)
    common.log(f"wrote {len(rows)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
