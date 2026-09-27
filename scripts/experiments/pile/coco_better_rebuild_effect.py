#!/usr/bin/env python3
"""What did widening *C* actually do to the 75 cells that were already there?

#4056 measured one half of it. Widening empties the shared pool -- the mean count
of *C*-classes held by a clean pool image falls 1.163 to 0.000 -- and that made a
cell read **+0.24 AP** higher. But that probe drew both arms from each roster's
*clean* set, which is the barren component alone. A real ``coco_better`` cell's
negatives are barren **plus** #3667's cross-class negatives, and those grow with
*C*: 75 cells' designated positives become 159 cells'. The two effects pull
opposite ways, so the net on a rebuilt cell is not implied by either.

**Answered 2026-09-20: -0.030 +- 0.004 AP as shipped, -0.003 size-matched,
dAUC +0.002.** The net is prevalence, not composition -- the barren draw is
capped at ``SCALE_N_NEG`` in both builds, so the emptiness never reaches the
cell, while the negative count rises 16,535 -> 23,891.

This measures it on the real thing rather than simulating a third time. It reads
the **preserved 25-class build** and the **rebuilt 53-class build** of the same
embedder and, for each of the 75 cells in ``SCALE_CLASSES_25``:

1. checks the positives are the same set in both -- designation is per cell and
   does not depend on *C*, so if that fails the comparison is not paired and the
   run says so rather than reporting a difference that is really a different cell;
2. fits the head the benchmark fits on each build's own negatives, 5-fold;
3. reports AP **as shipped** (each cell with the negatives it really has) and
   again **size-matched** (both arms cut to the smaller negative count).

Both are needed and they answer different questions. *As shipped* is what someone
re-reading a published number would see, and it includes the prevalence change
from a larger negative set. *Size-matched* isolates composition, which is the
quantity #4056 and #3986 both report.

Usage::

    python coco_better_rebuild_effect.py --embedder siglip --json out.json
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
from coco_negative_pool_effect import average_precision, auc, media_vec, unit  # noqa: E402

SEED = 0
KEEP25 = Path("/expscratch/sgreenberg/keep/coco-better-25-20260920")


def split(des: dict) -> tuple[dict, dict, set]:
    """positives per cell, evaluable-but-not-designated per cell, barren ids."""
    pos: dict[str, list[int]] = collections.defaultdict(list)
    ev: dict[str, list[int]] = collections.defaultdict(list)
    barren: set[int] = set()
    for i, d in des.items():
        cats = set(d.get("categories") or [])
        evs = d.get("evaluable_categories") or []
        for cell in cats:
            pos[cell].append(int(i))
        for cell in evs:
            if cell not in cats:
                ev[cell].append(int(i))
        if not cats and evs:
            barren.add(int(i))
    return pos, ev, barren


def fit_ap(X, p, nb, folds, rng):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    sp, sn = np.zeros(len(p)), np.zeros(len(nb))
    fp = rng.permutation(len(p)) % folds
    fn = rng.permutation(len(nb)) % folds
    for k in range(folds):
        tr = np.concatenate([X[p[fp != k]], X[nb[fn != k]]])
        y = np.concatenate([np.ones((fp != k).sum()), np.zeros((fn != k).sum())])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(tr, y)
        w, b = clf.coef_[0], float(clf.intercept_[0])
        sp[fp == k] = X[p[fp == k]] @ w + b
        sn[fn == k] = X[nb[fn == k]] @ w + b
    lab = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    return average_precision(np.concatenate([sp, sn]), lab), auc(sp, sn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--keep", type=Path, default=KEEP25)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    old = load_medias(args.keep / f"coco_better__{args.embedder}.pkl")
    new = load_medias(pc.EMBEDDINGS / f"coco_better__{args.embedder}.pkl")
    print(f"{args.embedder}: preserved build {len(old):,} medias, rebuilt {len(new):,}")

    po, eo, bo = split(old)
    pn, en, bn = split(new)
    print(f"cells: {len(po)} in the preserved build, {len(pn)} in the rebuilt one")

    # One vector store keyed by id; the same image under the same model embeds the
    # same, and the assert below is what makes that safe to assume.
    vec: dict[int, np.ndarray] = {}
    for src in (old, new):
        for i, d in src.items():
            k = int(i)
            v = unit(media_vec(d))
            if k in vec and not np.allclose(vec[k], v, atol=1e-5):
                raise SystemExit(f"media {k} embeds differently in the two builds; not comparable")
            vec[k] = v
    ids = sorted(vec)
    idx = {i: k for k, i in enumerate(ids)}
    X = np.stack([vec[i] for i in ids])

    rng = np.random.RandomState(SEED)
    rows = []
    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES_25 for b in pc.BOX_BANDS]
    print(f"\n{'cell':<20}{'n_neg old':>10}{'n_neg new':>10}{'AP old':>9}{'AP new':>9}{'dAP':>8}{'dAP mch':>9}")
    print("-" * 75)
    for cell in cells:
        if set(po[cell]) != set(pn[cell]):
            raise SystemExit(f"{cell}: positives differ between builds ({len(po[cell])} vs {len(pn[cell])})")
        p = np.array([idx[i] for i in sorted(po[cell])])
        if len(p) < 10:
            continue
        no, nn = sorted(eo[cell]), sorted(en[cell])
        ap_o, auc_o = fit_ap(X, p, np.array([idx[i] for i in no]), args.folds, rng)
        ap_n, auc_n = fit_ap(X, p, np.array([idx[i] for i in nn]), args.folds, rng)
        # Size-matched arm: composition only, prevalence held still.
        m = min(len(no), len(nn))
        r = random.Random(SEED)
        ap_om, _ = fit_ap(X, p, np.array([idx[i] for i in r.sample(no, m)]), args.folds, rng)
        r = random.Random(SEED)
        ap_nm, _ = fit_ap(X, p, np.array([idx[i] for i in r.sample(nn, m)]), args.folds, rng)
        rows.append(
            {
                "cell": cell,
                "n_pos": int(len(p)),
                "n_neg_old": len(no),
                "n_neg_new": len(nn),
                "cooccur_old": float(np.mean([i not in bo for i in no])),
                "cooccur_new": float(np.mean([i not in bn for i in nn])),
                "ap_old": ap_o,
                "ap_new": ap_n,
                "auc_old": auc_o,
                "auc_new": auc_n,
                "ap_old_matched": ap_om,
                "ap_new_matched": ap_nm,
                "n_neg_matched": m,
            }
        )
        if cell.endswith("@medium"):
            t = rows[-1]
            print(
                f"{cell:<20}{t['n_neg_old']:>10,}{t['n_neg_new']:>10,}{ap_o:>9.3f}{ap_n:>9.3f}"
                f"{ap_n - ap_o:>+8.3f}{ap_nm - ap_om:>+9.3f}"
            )

    def paired(a: str, b: str) -> tuple[float, float]:
        d = np.array([r[b] - r[a] for r in rows], dtype=float)
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    d_ap, se = paired("ap_old", "ap_new")
    d_m, se_m = paired("ap_old_matched", "ap_new_matched")
    d_auc, se_a = paired("auc_old", "auc_new")
    print(f"\n{len(rows)} cells of SCALE_CLASSES_25, {args.folds}-fold")
    print(
        f"negatives per cell (median): {int(np.median([r['n_neg_old'] for r in rows])):,}"
        f" -> {int(np.median([r['n_neg_new'] for r in rows])):,}"
    )
    print(
        f"co-occurring share: {100 * np.mean([r['cooccur_old'] for r in rows]):.0f}%"
        f" -> {100 * np.mean([r['cooccur_new'] for r in rows]):.0f}%"
    )
    print(f"paired dAUC:                 {d_auc:+.4f} +- {se_a:.4f}")
    print(f"paired dAP, AS SHIPPED:      {d_ap:+.4f} +- {se:.4f}")
    print(f"paired dAP, SIZE-MATCHED:    {d_m:+.4f} +- {se_m:.4f}")
    print(f"mean AP {np.mean([r['ap_old'] for r in rows]):.3f} -> {np.mean([r['ap_new'] for r in rows]):.3f}")
    print("\nby band (dAP as shipped, dAP size-matched):")
    for b in pc.BOX_BANDS:
        sel = [r for r in rows if r["cell"].endswith("@" + b)]
        print(
            f"  @{b:<8}{np.mean([r['ap_new'] - r['ap_old'] for r in sel]):>+9.4f}"
            f"{np.mean([r['ap_new_matched'] - r['ap_old_matched'] for r in sel]):>+10.4f}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "embedder": args.embedder,
                    "folds": args.folds,
                    "n_cells": len(rows),
                    "d_ap_shipped": d_ap,
                    "d_ap_shipped_se": se,
                    "d_ap_matched": d_m,
                    "d_ap_matched_se": se_m,
                    "d_auc": d_auc,
                    "cells": rows,
                },
                indent=1,
            )
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
