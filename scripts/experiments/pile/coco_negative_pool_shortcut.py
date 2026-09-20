#!/usr/bin/env python3
"""Does a head trained on the BARREN pool fire on ordinary images? And by how much?

This is #3986's real question, and neither of the other two scripts can answer
it. `coco_negative_pool.py` counts images. `coco_negative_pool_effect.py` ranks
by a TEXT query -- and #3667 established that a text query is structurally
unable to see this: it cannot learn a shortcut, only report whether the other
negatives are semantically nearer. Its text probe read -0.004 AUC on a contrast
where its *trained* probe read 1.88x. Reading the text number as the answer
would have concluded there was nothing there.

So this trains the head the benchmark trains, exactly as `coco_quarry` poses it:

1. Fit a linear head on the cell as SHIPPED. That is NOT the barren shared pool
   alone: `coco_quarry` inherits #3667's fix, so a designated positive of one
   class is an evaluable negative for the other 73 cells. Measured on the built
   cell, a cell's shipped negative set is 16,535 images, **60% barren and 40%
   cross-class** -- not the ~84% barren the issue quotes, which is #3670's
   figure for `vg_scale` and does not transfer. Training on the barren 9,900
   alone measures the PRE-#3667 benchmark, which is not the one that ships;
   `--train-pool barren` does that on purpose, for the contrast.
2. Score two held-out sets with it: unseen **shipped** negatives, and a
   **representative** per-class draw (`{i : i does not hold A}`, at COCO's own
   co-occurrence rate) of the same size, plus the **co-occurring subset** of
   that draw.
3. Read the false-positive rate on each, at a threshold pinned to 5% on the
   held-out shipped negatives.

Both are there because they answer different questions. The representative draw
is *what would ship* under one-pool-filtered-per-class, so its FPR is the cost of
the change. The co-occurring subset is the apples-to-apples with #3667's 1.88 --
that study's "added" set was other classes' positives, so every image in it held
something, where a representative draw is only ~63% co-occurring and dilutes the
ratio by construction.

**AP is reported against a size-matched pair** as well, because "every published
cell is conditioned on this" is a claim about the numbers a ship decision reads.
Same head, same positives, n_neg pinned, so any AP move is composition.

Folded 5 ways over the positives and the negatives together, so the baseline FPR
is out-of-sample rather than the training set's own.

Usage::

    python coco_negative_pool_shortcut.py --embedder siglip --json out.json
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
from coco_negative_pool_effect import average_precision, auc, coco_holders, media_vec, unit  # noqa: E402

SEED = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--anchor-dir", type=Path, default=pc.COCO_ANCHOR_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fpr", type=float, default=0.05, help="baseline FPR the threshold is pinned to")
    ap.add_argument(
        "--train-pool",
        choices=("shipped", "barren"),
        default="shipped",
        help="shipped = the cell's real evaluable negatives (#3667 included); barren = the pre-#3667 pool",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    classes = list(pc.SCALE_CLASSES)
    holders, every = coco_holders(args.anchor_dir, classes)

    full = load_medias(pc.EMBEDDINGS / f"coco_quarry_full__{args.embedder}.pkl")
    des = load_medias(pc.EMBEDDINGS / f"coco_quarry__{args.embedder}.pkl")

    ids = sorted(full)
    idx = {i: k for k, i in enumerate(ids)}
    X = np.stack([unit(media_vec(full[i])) for i in ids])

    pos_of: dict[str, list[int]] = collections.defaultdict(list)
    # The cell's REAL negatives: every media the build marks evaluable for it and
    # does not designate into it. On `coco_quarry` that is the barren pool plus
    # #3667's cross-class negatives, which are 40% of it.
    ev_of: dict[str, list[int]] = collections.defaultdict(list)
    barren: list[int] = []
    for i, d in des.items():
        cats = set(d.get("categories") or [])
        for cell in cats:
            pos_of[cell].append(int(i))
        for cell in d.get("evaluable_categories") or []:
            if cell not in cats:
                ev_of[cell].append(int(i))
        if not cats and d.get("evaluable_categories"):
            barren.append(int(i))
    barren.sort()
    barren_set = set(barren)

    in_c: collections.Counter = collections.Counter()
    for c in classes:
        for i in holders[c]:
            in_c[i] += 1

    cells = [pc.scale_cell(c, b) for c in classes for b in pc.BOX_BANDS]
    rng = np.random.RandomState(SEED)
    rows = []
    print(f"{args.embedder}: {len(classes)} classes, train-pool={args.train_pool}, {args.folds}-fold\n")
    print(f"{'cell':<20}{'FPR base':>10}{'FPR p/c':>9}{'ratio':>7}{'FPR co':>8}{'ratio':>7}{'AP base':>9}{'AP p/c':>8}")
    print("-" * 76)
    for cell in cells:
        c = cell.split("@", 1)[0]
        p = np.array([idx[i] for i in sorted(pos_of[cell])])
        if len(p) < 10:
            continue
        neg_ids = sorted(barren) if args.train_pool == "barren" else sorted(ev_of[cell])
        n_neg = len(neg_ids)
        neg_set = set(neg_ids)
        nb = np.array([idx[i] for i in neg_ids])
        # Representative negatives, held out of training by construction: the
        # baseline pool is excluded because it IS the training set. It is ~15% of
        # the candidates, and dropping it shifts the barren share a little, so
        # the realised co-occurrence rate is reported rather than assumed.
        cand = [i for i in every if i not in holders[c] and i not in neg_set]
        r = random.Random(SEED)
        rep = sorted(r.sample(cand, n_neg))
        nr = np.array([idx[i] for i in rep])

        sp, sn = np.zeros(len(p)), np.zeros(len(nb))
        sr = np.zeros((args.folds, len(nr)))
        fp, fn = rng.permutation(len(p)) % args.folds, rng.permutation(len(nb)) % args.folds
        for k in range(args.folds):
            tr = np.concatenate([X[p[fp != k]], X[nb[fn != k]]])
            y = np.concatenate([np.ones((fp != k).sum()), np.zeros((fn != k).sum())])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(tr, y)
            w, b = clf.coef_[0], float(clf.intercept_[0])
            sp[fp == k] = X[p[fp == k]] @ w + b
            sn[fn == k] = X[nb[fn == k]] @ w + b
            # Never in training, so every fold scores all of them; averaging is
            # the out-of-sample analogue of the held-out score the others get.
            sr[k] = X[nr] @ w + b
        sr_m = sr.mean(axis=0)
        co_mask = np.array([in_c[i] > 0 for i in rep])
        sc_m = sr_m[co_mask]

        thr = float(np.quantile(sn, 1 - args.fpr))
        fpr_s = float((sn > thr).mean())
        fpr_r = float((sr_m > thr).mean())
        fpr_c = float((sc_m > thr).mean()) if len(sc_m) else float("nan")
        ap_s = average_precision(np.concatenate([sp, sn]), np.concatenate([np.ones(len(sp)), np.zeros(len(sn))]))
        ap_r = average_precision(np.concatenate([sp, sr_m]), np.concatenate([np.ones(len(sp)), np.zeros(len(sr_m))]))
        rows.append(
            {
                "cell": cell,
                "n_pos": len(p),
                "n_neg": n_neg,
                "cooccur_rate": float(co_mask.mean()),
                "cooccur_rate_baseline": float(np.mean([i not in barren_set for i in neg_ids])),
                "fpr_shared": fpr_s,
                "fpr_perclass": fpr_r,
                "fpr_cooccur": fpr_c,
                "ratio_perclass": fpr_r / fpr_s if fpr_s else float("nan"),
                "ratio_cooccur": fpr_c / fpr_s if fpr_s else float("nan"),
                "auc_shared": auc(sp, sn),
                "auc_perclass": auc(sp, sr_m),
                "ap_shared": ap_s,
                "ap_perclass": ap_r,
            }
        )
        if cell.endswith("@medium"):
            t = rows[-1]
            print(
                f"{cell:<20}{fpr_s:>9.3f}{fpr_r:>9.3f}{t['ratio_perclass']:>7.2f}"
                f"{fpr_c:>8.3f}{t['ratio_cooccur']:>7.2f}{ap_s:>8.3f}{ap_r:>8.3f}"
            )

    def agg(key: str) -> tuple[float, float]:
        a = np.array([r[key] for r in rows], dtype=float)
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    by_band = {
        b: agg_band
        for b in pc.BOX_BANDS
        if (
            agg_band := (
                float(np.mean([r["ratio_cooccur"] for r in rows if r["cell"].endswith("@" + b)])),
                float(np.mean([r["ratio_perclass"] for r in rows if r["cell"].endswith("@" + b)])),
            )
        )
    }
    d_ap = np.array([r["ap_perclass"] - r["ap_shared"] for r in rows])
    d_auc = np.array([r["auc_perclass"] - r["auc_shared"] for r in rows])
    rp, rp_se = agg("ratio_perclass")
    rc, rc_se = agg("ratio_cooccur")
    base_co = 100 * np.mean([r["cooccur_rate_baseline"] for r in rows])
    print(f"\n{len(rows)} cells, train-pool={args.train_pool}, threshold pinned to {args.fpr:.0%} FPR on it")
    print(f"n_neg per cell: {int(np.median([r['n_neg'] for r in rows])):,} (median)")
    print(
        f"co-occurrence: baseline pool {base_co:.0f}%, representative draw {100 * np.mean([r['cooccur_rate'] for r in rows]):.0f}%"
    )
    print(f"FPR ratio (representative / baseline):  {rp:.2f} +- {rp_se:.2f}")
    print(f"FPR ratio (co-occurring  / baseline):   {rc:.2f} +- {rc_se:.2f}   <- comparable to #3667's 1.88")
    print(
        f"paired dAUC (baseline -> representative): {d_auc.mean():+.3f} +- {d_auc.std(ddof=1) / np.sqrt(len(d_auc)):.3f}"
    )
    print(
        f"paired dAP  (baseline -> representative): {d_ap.mean():+.3f} +- {d_ap.std(ddof=1) / np.sqrt(len(d_ap)):.3f}"
    )
    print(f"mean AP {np.mean([r['ap_shared'] for r in rows]):.3f} -> {np.mean([r['ap_perclass'] for r in rows]):.3f}")
    print("\nby band (co-occurring ratio, representative ratio):")
    for b, (rc_b, rp_b) in by_band.items():
        print(f"  @{b:<7}{rc_b:>6.2f}{rp_b:>8.2f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "embedder": args.embedder,
                    "folds": args.folds,
                    "train_pool": args.train_pool,
                    "fpr_target": args.fpr,
                    "n_cells": len(rows),
                    "n_neg": n_neg,
                    "ratio_perclass_mean": rp,
                    "ratio_perclass_se": rp_se,
                    "ratio_cooccur_mean": rc,
                    "ratio_cooccur_se": rc_se,
                    "d_ap_mean": float(d_ap.mean()),
                    "d_auc_mean": float(d_auc.mean()),
                    "by_band": {b: {"ratio_cooccur": v[0], "ratio_perclass": v[1]} for b, v in by_band.items()},
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
