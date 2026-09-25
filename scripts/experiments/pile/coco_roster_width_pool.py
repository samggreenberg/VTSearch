#!/usr/bin/env python3
"""Does widening *C* move the numbers of the classes that were already in it?

#4056 widens *C* from the shipped 25 to the 54 the count rule admits. Per-class
supply is not what that costs -- #3983 settled it -- but the **shared negative
pool** is drawn as *holds none of C*, so every class added shrinks the candidate
set it is drawn from: 49,503 clean images at 25, 16,058 at 54.

The pool *drawn* is ``SCALE_N_NEG`` = 9,900 either way, so no cell changes size.
What changes is where those 9,900 come from, and every published `coco_better`
number for the shipped 25 is conditioned on the wider draw. So:

1. Build the clean candidate set at each roster width -- images holding none of
   the 25, and none of the 54 -- from the instances JSONs, not the built cell,
   because the pool's rule is about HOLDING a class, not banding it.
2. Draw ``SCALE_N_NEG`` from each, size-matched, and fit the head the benchmark
   fits on the cell's positives against it, 5-fold.
3. Read AP and the FPR at a threshold pinned on the shipped-roster arm.

The expected direction is worth stating in advance, because it is the opposite
of the intuition that a smaller pool is a worse one. ``clean@54`` excludes every
image holding `person`, `dining table`, `tv`, `laptop` and 25 more, so what
survives is systematically **emptier** -- and emptier negatives are easier to
tell a positive from. If the widening moves anything, the sign to expect is
published numbers getting *better*, which would mean they are not comparable
across the roster change rather than merely noisier. Mean categories per pool
image is reported for both arms so the mechanism is visible, not inferred.

Usage::

    python coco_roster_width_pool.py --roster-json supply54.json --json out.json
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
    ap.add_argument("--roster-json", type=Path, required=True, help="coco_only_supply.py --out, for its `roster`")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fpr", type=float, default=0.05, help="FPR the threshold is pinned to on the shipped arm")
    ap.add_argument("--n-neg", type=int, default=pc.SCALE_N_NEG)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    shipped = list(pc.SCALE_CLASSES)
    wide = list(json.loads(args.roster_json.read_text())["roster"])
    missing = [c for c in shipped if c not in wide]
    if missing:
        raise SystemExit(
            f"the wide roster drops shipped classes, which this measurement assumes it does not: {missing}"
        )

    # `every` is all of COCO 2017; holders is over the union of both rosters so
    # one pass answers both widths.
    holders, every = coco_holders(args.anchor_dir, sorted(set(shipped) | set(wide)))

    full = load_medias(pc.EMBEDDINGS / f"coco_better_full__{args.embedder}.pkl")
    des = load_medias(pc.EMBEDDINGS / f"coco_better__{args.embedder}.pkl")

    ids = sorted(full)
    idx = {i: k for k, i in enumerate(ids)}
    X = np.stack([unit(media_vec(full[i])) for i in ids])

    pos_of: dict[str, list[int]] = collections.defaultdict(list)
    for i, d in des.items():
        for cell in set(d.get("categories") or []):
            pos_of[cell].append(int(i))

    held = collections.Counter()
    for c in sorted(set(shipped) | set(wide)):
        for i in holders[c]:
            held[i] += 1

    embedded = [i for i in every if i in idx]
    clean_shipped = [i for i in embedded if not any(i in holders[c] for c in shipped)]
    clean_wide = [i for i in embedded if not any(i in holders[c] for c in wide)]
    print(f"{args.embedder}: clean@{len(shipped)} = {len(clean_shipped):,}   clean@{len(wide)} = {len(clean_wide):,}")
    if len(clean_wide) < args.n_neg:
        raise SystemExit(f"clean@{len(wide)} holds {len(clean_wide):,}, under n_neg={args.n_neg:,}")

    # Emptiness is the mechanism, so measure it rather than assert it. `held`
    # counts membership of the WIDE roster, which is the honest yardstick: it is
    # the same question asked of both pools.
    def emptiness(pool: list[int]) -> float:
        return float(np.mean([held[i] for i in pool]))

    cells = [pc.scale_cell(c, b) for c in shipped for b in pc.BOX_BANDS]
    rng = np.random.RandomState(SEED)
    rows = []
    print(f"\n{'cell':<20}{'FPR ship':>10}{'FPR wide':>10}{'ratio':>7}{'AP ship':>9}{'AP wide':>9}{'dAP':>8}")
    print("-" * 73)
    for cell in cells:
        c = cell.split("@", 1)[0]
        p = np.array([idx[i] for i in sorted(pos_of[cell]) if i in idx])
        if len(p) < 10:
            continue
        r = random.Random(SEED)
        # Both arms exclude the cell's own class by construction (clean sets
        # hold none of C, and c is in both rosters), so no positive can leak in.
        pool_s = sorted(r.sample(clean_shipped, args.n_neg))
        r = random.Random(SEED)
        pool_w = sorted(r.sample(clean_wide, args.n_neg))

        out: dict[str, float] = {}
        for arm, pool in (("shipped", pool_s), ("wide", pool_w)):
            nb = np.array([idx[i] for i in pool])
            sp, sn = np.zeros(len(p)), np.zeros(len(nb))
            fp = rng.permutation(len(p)) % args.folds
            fn = rng.permutation(len(nb)) % args.folds
            for k in range(args.folds):
                tr = np.concatenate([X[p[fp != k]], X[nb[fn != k]]])
                y = np.concatenate([np.ones((fp != k).sum()), np.zeros((fn != k).sum())])
                clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(tr, y)
                w, b = clf.coef_[0], float(clf.intercept_[0])
                sp[fp == k] = X[p[fp == k]] @ w + b
                sn[fn == k] = X[nb[fn == k]] @ w + b
            lab = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
            out[f"ap_{arm}"] = average_precision(np.concatenate([sp, sn]), lab)
            out[f"auc_{arm}"] = auc(sp, sn)
            out[f"thr_{arm}"] = float(np.quantile(sn, 1 - args.fpr))
            out[f"scores_neg_{arm}"] = sn
            out[f"emptiness_{arm}"] = emptiness(pool)

        # One threshold, set on the shipped arm, read on both: the question is
        # what the widened pool does to a cut the shipped roster calibrated.
        thr = out["thr_shipped"]
        fpr_s = float((out["scores_neg_shipped"] > thr).mean())
        fpr_w = float((out["scores_neg_wide"] > thr).mean())
        rows.append(
            {
                "cell": cell,
                "n_pos": int(len(p)),
                "n_neg": int(args.n_neg),
                "fpr_shipped": fpr_s,
                "fpr_wide": fpr_w,
                "ratio_wide": fpr_w / fpr_s if fpr_s else float("nan"),
                "ap_shipped": out["ap_shipped"],
                "ap_wide": out["ap_wide"],
                "auc_shipped": out["auc_shipped"],
                "auc_wide": out["auc_wide"],
                "emptiness_shipped": out["emptiness_shipped"],
                "emptiness_wide": out["emptiness_wide"],
            }
        )
        if cell.endswith("@medium"):
            t = rows[-1]
            print(
                f"{cell:<20}{fpr_s:>10.3f}{fpr_w:>10.3f}{t['ratio_wide']:>7.2f}"
                f"{t['ap_shipped']:>9.3f}{t['ap_wide']:>9.3f}{t['ap_wide'] - t['ap_shipped']:>+8.3f}"
            )

    def paired(key_a: str, key_b: str) -> tuple[float, float]:
        a = np.array([r[key_b] - r[key_a] for r in rows], dtype=float)
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    d_ap, d_ap_se = paired("ap_shipped", "ap_wide")
    d_auc, d_auc_se = paired("auc_shipped", "auc_wide")
    ratio = np.array([r["ratio_wide"] for r in rows], dtype=float)
    print(
        f"\n{len(rows)} cells, n_neg={args.n_neg:,} in both arms, threshold pinned to {args.fpr:.0%} on the shipped arm"
    )
    print(f"clean candidates: {len(clean_shipped):,} at |C|={len(shipped)} -> {len(clean_wide):,} at |C|={len(wide)}")
    print(
        f"mean classes held per pool image: {np.mean([r['emptiness_shipped'] for r in rows]):.3f} "
        f"-> {np.mean([r['emptiness_wide'] for r in rows]):.3f}"
    )
    print(
        f"FPR ratio (wide / shipped):              {ratio.mean():.2f} +- {ratio.std(ddof=1) / np.sqrt(len(ratio)):.2f}"
    )
    print(f"paired dAUC (shipped -> wide):           {d_auc:+.4f} +- {d_auc_se:.4f}")
    print(f"paired dAP  (shipped -> wide):           {d_ap:+.4f} +- {d_ap_se:.4f}")
    print(f"mean AP {np.mean([r['ap_shipped'] for r in rows]):.3f} -> {np.mean([r['ap_wide'] for r in rows]):.3f}")
    print("\nby band (dAP, FPR ratio):")
    for b in pc.BOX_BANDS:
        sel = [r for r in rows if r["cell"].endswith("@" + b)]
        print(
            f"  @{b:<8}{np.mean([r['ap_wide'] - r['ap_shipped'] for r in sel]):>+8.4f}"
            f"{np.mean([r['ratio_wide'] for r in sel]):>8.2f}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "embedder": args.embedder,
                    "folds": args.folds,
                    "fpr_target": args.fpr,
                    "n_neg": args.n_neg,
                    "n_classes_shipped": len(shipped),
                    "n_classes_wide": len(wide),
                    "clean_shipped": len(clean_shipped),
                    "clean_wide": len(clean_wide),
                    "n_cells": len(rows),
                    "d_ap_mean": d_ap,
                    "d_ap_se": d_ap_se,
                    "d_auc_mean": d_auc,
                    "d_auc_se": d_auc_se,
                    "cells": rows,
                },
                indent=1,
            )
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
