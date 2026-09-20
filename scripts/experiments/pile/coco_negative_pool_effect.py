#!/usr/bin/env python3
"""How much does a detector's number move when the negative pool stops being barren?

`coco_negative_pool.py` settled the arithmetic of #3986: "holds none of C" caps
the class list, and it draws negatives that co-occur with another class in C
**0%** of the time where a representative set for the same class would be ~86%.
What it could not say is the thing every published cell is conditioned on --
*how much a measured number moves* -- because that needs scoring, not counting.

This is that measurement, and it is deliberately the SAME instrument
`cross_class_negatives_difficulty.py` used for #3667, so the two are directly
comparable rather than two scales that both say "harder".

Same positives, same query, two negative sets, one number each:

* **shared** -- the shipped 9,900. Today's rule: an image holding no class in C.
* **per-class** -- 9,900 drawn from `{i : i does not hold A}`, which is COCO's
  own co-occurrence rate. This is what "one pool, filtered per class" delivers;
  a finite *P* only changes which images are available, not the distribution,
  and `coco_negative_pool.py` already reports the |P| each class needs.

**The two sets are the same size on purpose.** #3667's AP fell partly because
its pool grew, and a prevalence change is not a difficulty change. Holding
n_neg at `SCALE_N_NEG` makes prevalence identical in both arms, so AP moves here
only if composition moved it.

Three numbers per cell:

* **AUC** ranks positives against negatives and is prevalence-free. A drop is
  the claim being true: the representative pool is harder.
* **AP** is reported beside it because it is what a ship decision reads. With
  prevalence pinned, a fall here is the real cost of the old pool's optimism.
* **AUC against the CO-OCCURRING negatives alone** -- the subset holding some
  other class in C -- is the direct statement of the shortcut: these are the
  images the shared pool can never contain.

Scoring is unit-normalised cosine against the text vector, which is what
`patch_styles`'s whole-image `exemplar_sims` ranks in, so this is the harness's
geometry and not one invented here.

Usage::

    python coco_negative_pool_effect.py --embedder siglip --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import pile_config as pc  # noqa: E402

pc.setup_env()

import numpy as np  # noqa: E402

from _cells_io import load_medias  # noqa: E402


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def media_vec(d: dict) -> np.ndarray:
    emb = d.get("embeddings") or {}
    for key in ("image", "clip", "default"):
        if key in emb:
            return np.asarray(emb[key], dtype=np.float32)
    # One embedding per media in a pile cell; take whatever it is called.
    return np.asarray(next(iter(emb.values())), dtype=np.float32)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties counted as half."""
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.concatenate([pos, neg])
    ranks = order.argsort().argsort().astype(np.float64)
    _, inv, counts = np.unique(order, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = int(y.sum())
    return float((prec * y).sum() / n_pos) if n_pos else float("nan")


def coco_holders(anchor_dir: Path, classes: list[str]) -> tuple[dict[str, set[int]], list[int]]:
    """`holders[cls]` over the WHOLE corpus, and every COCO 2017 image id.

    Read from the instances JSONs rather than the built cell on purpose: a cell's
    ``regions`` carry only the *banded* boxes, so an image whose only `car` is
    too small to band looks car-free there. The shared pool's rule is about
    HOLDING a class, not about banding it, and the two differ.
    """
    holders: dict[str, set[int]] = collections.defaultdict(set)
    want = set(classes)
    every: list[int] = []
    for split in ("val2017", "train2017"):
        with (anchor_dir / f"instances_{split}.json").open() as fh:
            data = json.load(fh)
        cats = {c["id"]: c["name"] for c in data["categories"]}
        every.extend(int(im["id"]) for im in data["images"])
        for a in data["annotations"]:
            if a.get("iscrowd"):
                continue
            name = cats[a["category_id"]]
            if name in want:
                holders[name].add(int(a["image_id"]))
        del data
    return holders, sorted(every)


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--embedder", default="siglip")
    ap_.add_argument("--anchor-dir", type=Path, default=pc.COCO_ANCHOR_DIR)
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--json", type=Path, default=None)
    args = ap_.parse_args()

    classes = list(pc.SCALE_CLASSES)
    holders, every = coco_holders(args.anchor_dir, classes)

    full = load_medias(pc.EMBEDDINGS / f"coco_quarry_full__{args.embedder}.pkl")
    des = load_medias(pc.EMBEDDINGS / f"coco_quarry__{args.embedder}.pkl")
    missing = [i for i in every if i not in full]
    if missing:
        print(f"full-corpus cell is missing {len(missing):,} COCO images; cannot measure", file=sys.stderr)
        return 2

    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415

    qvec = {}
    for c in classes:
        v = embed_text_query(c, "image", enrich=False, embedder_name=args.embedder)
        if v is None:
            print(f"{args.embedder} has no text tower; nothing to measure", file=sys.stderr)
            return 2
        qvec[c] = unit(np.asarray(v, dtype=np.float32))

    ids = sorted(full)
    idx = {i: k for k, i in enumerate(ids)}
    matrix = np.stack([unit(media_vec(full[i])) for i in ids])

    # The SHIPPED cell: its designated positives, and the one shared pool every
    # cell in it is scored against.
    pos_of: dict[str, list[int]] = collections.defaultdict(list)
    shared: list[int] = []
    for i, d in des.items():
        cats = d.get("categories") or []
        if cats:
            for cell in cats:
                pos_of[cell].append(int(i))
        elif d.get("evaluable_categories"):
            shared.append(int(i))
    shared.sort()
    n_neg = len(shared)

    # How often a negative holds SOME other class in C. 0% for the shared pool
    # by construction; this is the quantity the whole issue turns on.
    in_c: collections.Counter = collections.Counter()
    for c in classes:
        for i in holders[c]:
            in_c[i] += 1

    cells = [pc.scale_cell(c, b) for c in classes for b in pc.BOX_BANDS]
    rows = []
    print(f"{args.embedder}: |C| = {len(classes)}, n_neg pinned at {n_neg:,} in both arms\n")
    print(f"{'cell':<20}{'AUC shr':>9}{'AUC p/c':>9}{'d':>8}{'AUC co':>8}{'AP shr':>8}{'AP p/c':>8}{'d':>8}")
    print("-" * 78)
    for cell in cells:
        c = cell.split("@", 1)[0]
        p = np.array([idx[i] for i in sorted(pos_of[cell])])
        if not len(p):
            continue
        # Per-class negatives: COCO's own co-occurrence rate, same size as the
        # shared pool so prevalence is identical and AP is comparable.
        cand = [i for i in every if i not in holders[c]]
        rng = random.Random(args.seed)
        perclass = sorted(rng.sample(cand, n_neg))
        cooccur = [i for i in perclass if in_c[i] > 0]

        sims = matrix @ qvec[c]
        ns = np.array([idx[i] for i in shared])
        np_ = np.array([idx[i] for i in perclass])
        nc = np.array([idx[i] for i in cooccur])

        a_s, a_p = auc(sims[p], sims[ns]), auc(sims[p], sims[np_])
        a_c = auc(sims[p], sims[nc]) if len(nc) else float("nan")
        ap_s = average_precision(
            np.concatenate([sims[p], sims[ns]]), np.concatenate([np.ones(len(p)), np.zeros(len(ns))])
        )
        ap_p = average_precision(
            np.concatenate([sims[p], sims[np_]]), np.concatenate([np.ones(len(p)), np.zeros(len(np_))])
        )
        rows.append(
            {
                "cell": cell,
                "n_pos": len(p),
                "n_neg": n_neg,
                "cooccur_shared": 0.0,
                "cooccur_perclass": len(cooccur) / len(perclass),
                "auc_shared": a_s,
                "auc_perclass": a_p,
                "auc_cooccur_only": a_c,
                "ap_shared": ap_s,
                "ap_perclass": ap_p,
            }
        )
        if cell.endswith("@medium"):
            print(
                f"{cell:<20}{a_s:>9.3f}{a_p:>9.3f}{a_p - a_s:>+8.3f}{a_c:>8.3f}"
                f"{ap_s:>8.3f}{ap_p:>8.3f}{ap_p - ap_s:>+8.3f}"
            )

    d_auc = np.array([r["auc_perclass"] - r["auc_shared"] for r in rows])
    d_co = np.array([r["auc_cooccur_only"] - r["auc_shared"] for r in rows])
    d_ap = np.array([r["ap_perclass"] - r["ap_shared"] for r in rows])
    se = lambda a: float(a.std(ddof=1) / np.sqrt(len(a)))  # noqa: E731
    co = float(np.mean([r["cooccur_perclass"] for r in rows]))
    print(f"\n{len(rows)} cells, {len(classes)} classes")
    print(f"co-occurrence with another class in C:  shared 0.0%   per-class {100 * co:.0f}%")
    print(f"paired dAUC (shared -> per-class):      {d_auc.mean():+.3f} +- {se(d_auc):.3f}")
    print(f"paired dAUC (shared -> CO-OCCURRING):   {d_co.mean():+.3f} +- {se(d_co):.3f}")
    print(f"paired dAP  (shared -> per-class):      {d_ap.mean():+.3f} +- {se(d_ap):.3f}")
    print(
        f"mean AP {np.mean([r['ap_shared'] for r in rows]):.3f} -> "
        f"{np.mean([r['ap_perclass'] for r in rows]):.3f}  (prevalence identical)"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "embedder": args.embedder,
                    "n_classes": len(classes),
                    "n_cells": len(rows),
                    "n_neg": n_neg,
                    "seed": args.seed,
                    "cooccur_perclass_mean": co,
                    "d_auc_mean": float(d_auc.mean()),
                    "d_auc_se": se(d_auc),
                    "d_auc_cooccur_mean": float(d_co.mean()),
                    "d_auc_cooccur_se": se(d_co),
                    "d_ap_mean": float(d_ap.mean()),
                    "d_ap_se": se(d_ap),
                    "cells": rows,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
