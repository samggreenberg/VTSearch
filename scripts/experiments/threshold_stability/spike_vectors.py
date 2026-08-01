"""What's geometrically special about the spiking vectors? (#2790, pure embedding)

No eval framework — just the cached embeddings. For each class c (one embedder), take
the positive image vectors ``G``, the negative image vectors ``B``, and the spikers
``S`` (the negatives that repeatedly stressed the cut, from ``spike_items.json``), and
ask whether ``S`` is distinguishable from ordinary bads ``B\\S`` by any of:

* ``cos_G`` — cosine to the good centroid (how class-like);
* ``proj_w`` — projection on the good↔bad axis ``mean(G)-mean(B)`` (Fisher direction);
* ``max_cos_G`` — nearest single good (a look-alike to one exemplar);
* ``knn_good`` — fraction of good among the k nearest neighbors in ``G∪B`` (does it sit
  in mixed/boundary territory);
* ``b_outlier`` — mean distance to the k nearest bads (is it an outlier among negatives);
* ``margin`` — ``cos_G - cos_B``.

For each spiker it computes the **percentile** of that feature among all bads, then
aggregates the mean spiker percentile per feature across classes: a feature where
spikers land at a consistently extreme percentile is the signature. All vectors are
the whole-image embeddings from ``regions/<ds>/<embedder>/whole/<id>.npz`` (same space
for good and bad), L2-normalized.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sod"))
from datasets import SodDataset  # noqa: E402


def _load_vecs(regions_dir: Path, ids: list[int]) -> tuple[np.ndarray, list[int]]:
    vs, keep = [], []
    for i in ids:
        p = regions_dir / f"{i}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            if "whole_vec" in z:
                vs.append(np.asarray(z["whole_vec"], dtype=np.float64))
                keep.append(i)
    if not vs:
        return np.empty((0, 0)), []
    X = np.vstack(vs)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, keep


def _features(V: np.ndarray, G: np.ndarray, gc: np.ndarray, bc: np.ndarray, w: np.ndarray) -> dict:
    """Feature matrix for rows of V, given good set G and axes."""
    cos_G = V @ gc
    cos_B = V @ bc
    proj_w = V @ w
    max_cos_G = (V @ G.T).max(axis=1) if len(G) else np.full(len(V), np.nan)
    return {"cos_G": cos_G, "cos_B": cos_B, "margin": cos_G - cos_B, "proj_w": proj_w, "max_cos_G": max_cos_G}


def _knn_good_and_outlier(B: np.ndarray, G: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each bad row: fraction of good among k NN in G∪B, and mean dist to k NN bads."""
    allX = np.vstack([G, B]) if len(G) else B
    is_good = np.concatenate([np.ones(len(G)), np.zeros(len(B))]) if len(G) else np.zeros(len(B))
    knn_good, b_out = [], []
    for i in range(len(B)):
        v = B[i]
        sims = allX @ v
        order = np.argsort(-sims)
        order = order[order != (len(G) + i)][:k]  # drop self
        knn_good.append(float(is_good[order].mean()))
        bsims = B @ v
        b_order = np.argsort(-bsims)
        b_order = b_order[b_order != i][:k]
        b_out.append(float((1.0 - bsims[b_order]).mean()))
    return np.array(knn_good), np.array(b_out)


def _pctile(x: float, arr: np.ndarray) -> float:
    return float((arr < x).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Embedding-space signature of spiking vectors (#2790).")
    ap.add_argument("--items", required=True, help="spike_items.json")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--embedder", default="siglip2")
    ap.add_argument("--min-spikes", type=int, default=5, help="an image is a spiker if it spiked in >= this many seeds")
    ap.add_argument("--neg-sample", type=int, default=1200)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    data = json.loads(Path(args.items).read_text())
    by_class: dict[str, list[int]] = {}
    for it in data["items"]:
        if it["gt_label"] == "bad" and it["n_spikes"] >= args.min_spikes:
            by_class.setdefault(it["cls"], []).append(int(it["image_id"]))

    regions = Path(args.cache) / "regions" / args.dataset / args.embedder / "whole"
    feats = ["cos_G", "cos_B", "margin", "proj_w", "max_cos_G", "knn_good", "b_outlier"]
    per_class_pcts: dict[str, list[dict]] = {}
    rng = np.random.default_rng(0)

    with SodDataset(args.dataset) as ds:
        for cls, spikers in sorted(by_class.items()):
            name = cls.replace("-", " ")
            try:
                split = ds.class_split(name, min_box_frac=0.03, neg_multiple=1, seed=0)
            except Exception:
                continue
            neg_pool = [i for i in split.negative_ids if i not in set(spikers)]
            if len(neg_pool) > args.neg_sample:
                neg_pool = list(rng.choice(neg_pool, size=args.neg_sample, replace=False))
            G, _ = _load_vecs(regions, list(split.gt_boxes))  # positive whole-image vecs
            B, bkeep = _load_vecs(regions, neg_pool + spikers)  # bads incl. spikers
            if len(G) < 3 or len(B) < 20:
                continue
            gc = G.mean(0)
            gc /= np.linalg.norm(gc) + 1e-12
            bc = B[: len(neg_pool)].mean(0) if len(neg_pool) else B.mean(0)
            bc /= np.linalg.norm(bc) + 1e-12
            w = gc - bc
            w /= np.linalg.norm(w) + 1e-12
            fx = _features(B, G, gc, bc, w)
            knn_good, b_out = _knn_good_and_outlier(B, G, args.k)
            fx["knn_good"] = knn_good
            fx["b_outlier"] = b_out
            sp_idx = [bkeep.index(i) for i in spikers if i in bkeep]
            if not sp_idx:
                continue
            recs = []
            for si in sp_idx:
                recs.append({f: _pctile(fx[f][si], fx[f]) for f in feats})
            per_class_pcts[cls] = recs

    # Aggregate: mean spiker percentile per feature, across all spikers of all classes.
    allrecs = [r for recs in per_class_pcts.values() for r in recs]
    summary = {f: round(statistics.fmean(r[f] for r in allrecs), 3) for f in feats} if allrecs else {}
    out = {"n_classes": len(per_class_pcts), "n_spikers": len(allrecs), "mean_spiker_percentile": summary,
           "per_class": {c: {f: round(statistics.fmean(r[f] for r in recs), 3) for f in feats}
                         for c, recs in per_class_pcts.items()}}  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"classes={out['n_classes']}  spikers={out['n_spikers']}")
    print("\nMean spiker PERCENTILE among bads (0.5 = typical bad; >>0.5 or <<0.5 = signature):")
    for f in feats:
        print(f"  {f:<12} {summary.get(f, float('nan')):.3f}")
    print("\nPer-class (mean spiker percentile):")
    print(f"  {'class':<15} " + " ".join(f"{f:>9}" for f in feats))
    for c, m in sorted(out["per_class"].items()):
        print(f"  {c:<15} " + " ".join(f"{m[f]:>9.2f}" for f in feats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
