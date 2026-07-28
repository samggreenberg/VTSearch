"""CPU-backend verification of the sweep winners (plan §Locked scope, §Analysis.5).

The sweep fits UMAP with **cuML** (the GPU production path). Production may
instead run CPU **umap-learn**, which optimizes differently and can produce a
different layout for identical params — so a defaults table chosen on cuML is
only safe to ship if the ranking transfers. This script re-fits, with umap-learn
directly (forcing the CPU path), the **winning** params per embedder and the
**current default** baseline (n_neighbors=15, min_dist=0.1, compact=True), scores
both with the same separability metric, and reports whether the winner still
beats the baseline on CPU.

Large matrices are subsampled (CPU umap-learn is slow) — the ranking, not the
absolute number, is what must transfer.

    python cpu_verify.py            # uses summary.json recommendations
"""

from __future__ import annotations

import csv
import json

import numpy as np

import common as C
import metric as M

BASELINE = {"n_neighbors": 15, "min_dist": 0.1, "compact": True}  # current production default
# The defaults chosen from the cuML sweep — verify these exact values transfer to
# the CPU umap-learn backend (not summary.json's argmax, which we round to the grid).
CHOSEN = {
    "clap": {"n_neighbors": 15, "min_dist": 0.10, "compact": False},
    "clip": {"n_neighbors": 10, "min_dist": 0.05, "compact": False},
    "siglip": {"n_neighbors": 10, "min_dist": 0.05, "compact": False},
    "siglip_l": {"n_neighbors": 10, "min_dist": 0.05, "compact": False},
}
CPU_SUBSAMPLE = 6000
SEEDS = [0, 1, 2]


def _cpu_umap(X, nn, md, seed):
    import umap  # umap-learn (CPU)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(nn, X.shape[0] - 1),
        min_dist=md,
        metric="euclidean",
        random_state=seed,
    )
    return np.ascontiguousarray(reducer.fit_transform(X), dtype=np.float32)


def _score(X, taxonomy, coords, highd_knn, k):
    return M.taxonomy_separability(coords, X, taxonomy, k=k, highd_knn=highd_knn).ratio


def verify():
    from vtscore.projection.compaction import compact_layout

    recs = CHOSEN
    k = C.METRIC_K
    out = open(C.RESULTS_ROOT / "cpu_verify.csv", "w", newline="")
    w = csv.writer(out)
    w.writerow(
        ["dataset", "embedder", "setting", "n_neighbors", "min_dist", "compact", "ratio_cpu_mean", "ratio_cpu_std"]
    )

    verdicts = []
    for tag_path in sorted(C.MATRIX_DIR.glob("*.npz")):
        tag = tag_path.stem
        dataset, embedder = tag.split("__", 1)
        if embedder not in recs:
            continue
        d = np.load(tag_path, allow_pickle=True)
        X = d["X"].astype(np.float32)
        leaf = d["leaf"].astype(str)
        ml = d["ml_labels"] if "ml_labels" in d else None
        lineage_rows = d["lineage"] if "lineage" in d else None
        if X.shape[0] > CPU_SUBSAMPLE:
            rng = np.random.default_rng(0)
            sub = rng.choice(X.shape[0], CPU_SUBSAMPLE, replace=False)
            X, leaf = X[sub], leaf[sub]
            if ml is not None:
                ml = ml[sub]
            if lineage_rows is not None:
                lineage_rows = lineage_rows[sub]
        spec = C.ROSTER_BY_NAME[dataset]
        kw = {
            "lineage": (
                {leaf[i]: list(map(str, lineage_rows[i])) for i in range(len(leaf))} if lineage_rows is not None else {}
            )
        }
        if ml is not None:
            kw.update(ml_labels=ml, ml_names=d["ml_names"], ml_isroot=d["ml_isroot"])
        taxonomy = C.TAXONOMY_BUILDERS[spec.taxonomy](leaf, **kw)
        if sum(len(v) for v in taxonomy.values()) == 0:
            continue
        highd_knn = M.knn_indices(X, k, metric="cosine")

        rec = recs[embedder]
        settings = {
            "winner": {"n_neighbors": rec["n_neighbors"], "min_dist": rec["min_dist"], "compact": rec["compact"]},
            "baseline": BASELINE,
        }
        ratios = {}
        for name, s in settings.items():
            rs = []
            for seed in SEEDS:
                coords = _cpu_umap(X, s["n_neighbors"], s["min_dist"], seed)
                if s["compact"]:
                    coords = compact_layout(coords)
                rs.append(_score(X, taxonomy, coords, highd_knn, k))
            ratios[name] = rs
            w.writerow(
                [
                    dataset,
                    embedder,
                    name,
                    s["n_neighbors"],
                    s["min_dist"],
                    s["compact"],
                    round(float(np.mean(rs)), 5),
                    round(float(np.std(rs)), 5),
                ]
            )
            out.flush()
        transfers = np.mean(ratios["winner"]) >= np.mean(ratios["baseline"]) - 0.003  # within noise counts as transfer
        verdicts.append((dataset, embedder, np.mean(ratios["winner"]), np.mean(ratios["baseline"]), transfers))
        print(
            f"  {tag:26s} winner={np.mean(ratios['winner']):.3f} baseline={np.mean(ratios['baseline']):.3f} "
            f"{'✓ transfers' if transfers else '✗ REGRESSION'}"
        )
    out.close()

    print("\n=== CPU verify summary (per embedder) ===")
    by_emb = {}
    for _ds, emb, wv, bv, t in verdicts:
        by_emb.setdefault(emb, []).append((wv, bv, t))
    for emb, rows in by_emb.items():
        frac = np.mean([t for _w, _b, t in rows])
        dw = np.mean([w - b for w, b, _t in rows])
        print(
            f"  {emb:9s} winner−baseline Δ={dw:+.4f} on CPU · transfers on {int(frac * len(rows))}/{len(rows)} datasets"
        )
    json.dump(
        {
            "per_dataset": [
                {"dataset": d, "embedder": e, "winner": float(w), "baseline": float(b), "transfers": bool(t)}
                for d, e, w, b, t in verdicts
            ]
        },
        open(C.RESULTS_ROOT / "cpu_verify.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    verify()
