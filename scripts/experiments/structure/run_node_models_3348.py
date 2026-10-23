#!/usr/bin/env python
"""Score every candidate atlas node model on one (embedder, seed) (#3348).

    python run_node_models_3348.py --embedder dinov3_patch --seed 0 \
        --datasets a,b,c --mats DIR --out DIR

One task covers both halves of the #3329 part-2 measurement for one embedder
and one build/holdout split, for every candidate in ``node_models_3348``:

- **in-domain PIT**: each dataset's held-out 20 % scored against its own atlas
  -- the null, where a calibrated p-value is uniform;
- **cross-corpus power**: every ordered (build, query) dataset pair, the
  diagonal being that same holdout -- the guard's verdict, false alarm on the
  diagonal and detection off it.

The split is the #3329 split (``default_rng(seed).permutation``, 20 % held
out), so ``grid_mean`` at seed 0 is the same estimator on the same split the
part-2 shift stage ran.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()

import node_models_3348 as nm  # noqa: E402
from structure_fits_3329 import (  # noqa: E402
    CONFORMAL_K,
    HOLDOUT_FRACTION,
    _pit_stats,
    _rows,
    aggregate_paths,
    path_pvalues,
)

#: Cap on query rows per pair, as in domain_shift_3329.
MAX_QUERY = 4000


def load(mats: Path, dataset: str, embedder: str) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    stem = f"{dataset}__{embedder}"
    z = np.load(mats / f"{stem}.npz")
    meta = json.loads((mats / f"{stem}.json").read_text())
    return z["vecs"].astype(np.float32), z["ids"].astype(np.int64), meta["categories"]


def class_conformal(add, vecs: np.ndarray, cats: list[list[str]], seed: int) -> np.ndarray | None:
    """#3329's per-class kNN conformal control, unchanged (its KS 0.047 reading)."""
    from vtscore.detectors.evidence_coverage import support_pvalues  # noqa: PLC0415

    counts: dict[str, int] = {}
    for c in cats:
        for name in c:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    target = max(counts, key=lambda c: counts[c])
    pos = np.array([i for i, c in enumerate(cats) if target in c], dtype=int)
    if pos.size < 40:
        return None
    pos = pos[np.random.default_rng(seed + 1).permutation(pos.size)]
    cut = int(0.7 * pos.size)
    p = np.asarray(support_pvalues(vecs[pos[cut:]], vecs[pos[:cut]], k=CONFORMAL_K), dtype=np.float64)
    _pit_stats(add, "class_conformal", "in_class_holdout", p)
    return p


class Candidates:
    """Every candidate fitted on one dataset's build split."""

    def __init__(self, ids: np.ndarray, build_rows: np.ndarray, vecs: np.ndarray, seed: int, add):
        from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

        b_ids = ids[build_rows]
        b_vecs = vecs[build_rows]
        t0 = time.time()
        # The shipped build, exactly as production and #3329 call it.
        self.full = CoverageAtlas(
            {int(i): v for i, v in zip(b_ids, b_vecs, strict=True)}, k=3, max_depth=auto_max_depth(len(b_ids), k=3)
        )
        t_full = time.time() - t0
        # Split-conformal: the tree sees one half, calibration the other.
        perm = np.random.default_rng(seed + 11).permutation(len(b_ids))
        half = len(b_ids) // 2
        tree, cal = perm[:half], perm[half:]
        t0 = time.time()
        self.half = CoverageAtlas(
            {int(b_ids[j]): b_vecs[j] for j in tree}, k=3, max_depth=auto_max_depth(len(tree), k=3)
        )
        t_half = time.time() - t0

        for tag, atlas, secs in (("full", self.full, t_full), ("half", self.half, t_half)):
            rb = np.array([n["rbar"] for n in atlas.nodes.values()])
            add("structure", tag, "n_nodes", atlas.total_nodes)
            add("structure", tag, "depth", atlas.depth())
            add("structure", tag, "rbar_median", float(np.median(rb)), rb.size)
            add("structure", tag, "rbar_p10", float(np.quantile(rb, 0.10)), rb.size)
            add("structure", tag, "build_seconds", secs)

        full_f = nm.to_frame(self.full, b_vecs)
        full_row = {int(i): j for j, i in enumerate(b_ids)}
        tree_f = nm.to_frame(self.half, b_vecs[tree])
        tree_row = {int(b_ids[j]): r for r, j in enumerate(tree)}
        cal_f = nm.to_frame(self.half, b_vecs[cal])

        self.models: dict[str, nm.NodeModel] = {}
        for stat in nm.STATISTICS:
            for calib in nm.CALIBRATIONS:
                if calib == "loo":
                    m = nm.NodeModel(self.full, stat, "loo", full_f, full_row)
                else:
                    m = nm.NodeModel(self.half, stat, "split", tree_f, tree_row, cal=cal_f)
                self.models[f"{stat}_{calib}"] = m
                add("fit", f"{stat}_{calib}", "seconds", m.fit_seconds)
                add("fit", f"{stat}_{calib}", "eligible_nodes", len(m.models))
        self.knn_global = nm.GlobalKnn(b_vecs[tree], b_vecs[cal])
        add("fit", "knn_global", "seconds", self.knn_global.fit_seconds)

    def pvalues(self, x: np.ndarray, timing: dict[str, float]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        t0 = time.time()
        out["grid_mean_shipped"] = np.asarray(self.full.typicality_pvalues(x), dtype=np.float64)
        aggs = aggregate_paths(path_pvalues(self.full, x))
        out["grid_median"] = aggs["median"]
        out["grid_deepest"] = aggs["deepest"]
        # Consistency check on the path-capture route: must be ~0.
        timing["grid_mean_path_diff"] = max(
            timing.get("grid_mean_path_diff", 0.0), float(np.max(np.abs(aggs["mean"] - out["grid_mean_shipped"])))
        )
        timing["grid"] = timing.get("grid", 0.0) + time.time() - t0
        for name, m in self.models.items():
            t0 = time.time()
            for comb, p in nm.combine(m.paths(nm.to_frame(m.atlas, x))).items():
                out[f"{name}_{comb}"] = p
            timing[name] = timing.get(name, 0.0) + time.time() - t0
        t0 = time.time()
        out["knn_global"] = self.knn_global.pvalues(x)
        timing["knn_global"] = timing.get("knn_global", 0.0) + time.time() - t0
        return out


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--datasets", required=True, help="comma-separated")
    ap.add_argument("--mats", required=True, help="prepare_matrices_3348 output dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    import pandas as pd  # noqa: PLC0415

    t_start = time.time()
    names = [d for d in args.datasets.split(",") if d]
    data = {d: load(Path(args.mats), d, args.embedder) for d in names}
    pit_rows: list[dict] = []
    cands: dict[str, Candidates] = {}
    holdouts: dict[str, np.ndarray] = {}
    store: dict[str, np.ndarray] = {}
    for d in names:
        vecs, ids, cats = data[d]
        rows, add = _rows(embedder=args.embedder, seed=args.seed, dataset=d)
        perm = np.random.default_rng(args.seed).permutation(vecs.shape[0])
        n_hold = max(1, int(round(HOLDOUT_FRACTION * vecs.shape[0])))
        holdouts[d] = vecs[perm[:n_hold]]
        cands[d] = Candidates(ids, perm[n_hold:], vecs, args.seed, add)
        cc = class_conformal(add, vecs, cats, args.seed)
        if cc is not None:
            store[f"pit|{d}|class_conformal"] = cc.astype(np.float32)
        pit_rows += rows
        common.log(
            f"{d}: n={vecs.shape[0]} built; full atlas {cands[d].full.total_nodes} nodes depth {cands[d].full.depth()}"
        )

    shift_rows: list[dict] = []
    timing: dict[str, float] = {}
    for b in names:
        for q in names:
            x = holdouts[q] if q == b else data[q][0]
            if x.shape[0] > MAX_QUERY:
                x = x[np.random.default_rng(args.seed + 7).choice(x.shape[0], MAX_QUERY, replace=False)]
            pv = cands[b].pvalues(x, timing)
            if q == b:
                rows, add = _rows(embedder=args.embedder, seed=args.seed, dataset=b)
                for var, p in pv.items():
                    _pit_stats(add, var, "holdout", p)
                    store[f"pit|{b}|{var}"] = p.astype(np.float32)
                pit_rows += rows
            for var, p in pv.items():
                v = nm.shift_verdict(p)
                shift_rows.append(
                    {
                        "embedder": args.embedder,
                        "seed": args.seed,
                        "build_dataset": b,
                        "query_dataset": q,
                        "is_self": b == q,
                        "variant": var,
                        **v,
                    }
                )
                if q != b:
                    store[f"shift|{b}|{q}|{var}"] = p.astype(np.float16)
            common.log(
                f"  {b} <- {q}: n={x.shape[0]} shipped frac={np.mean(pv['grid_mean_shipped'] < 0.05):.3f} "
                f"knn_split_deepest frac={np.mean(pv['knn_split_deepest'] < 0.05):.3f}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.embedder}__s{args.seed}"
    pd.DataFrame(pit_rows).to_csv(out / f"pit_{stem}.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(out / f"shift_{stem}.csv", index=False)
    np.savez_compressed(out / f"pvals_{stem}.npz", **store)
    (out / f"timing_{stem}.json").write_text(
        json.dumps(
            {
                "embedder": args.embedder,
                "seed": args.seed,
                "query_seconds": timing,
                "total_seconds": time.time() - t_start,
            },
            indent=2,
        )
    )
    common.log(f"wrote {len(pit_rows)} pit rows, {len(shift_rows)} shift rows in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
