#!/usr/bin/env python3
"""Is a VG image's PROVENANCE readable off its embedding?

#3670 declines to draw every negative from the COCO-anchored half on the grounds
that the positives are only ~57% COCO-sourced, so an all-COCO negative pool would
let a detector separate positives from negatives on provenance rather than on
content -- VG draws its images from COCO and from YFCC100M, and those look
different.

That is an argument, not a measurement, and the whole composition decision turns
on it: if provenance is not linearly readable then the all-provable pool is free
and the negatives stop resting on VG's silence entirely.

So ask the vectors. Fit a linear probe to predict provenance from the embedding,
on balanced classes, and report cross-validated AUC per embedder. Chance is 0.5.

Read the number as a bound on the shortcut, not as a verdict on the pool: a high
AUC says the signal EXISTS to be learned, and a low one says an all-provable pool
cannot be gamed this way even in principle.

Three corrections, all from #3997, all of which move the number:

**1. The flag.** This probe predicted ``labels_exhaustive``, calling it
"COCO-sourced". The loader defines the two flags as different claims and says so
where it sets them: ``labels_exhaustive`` is true for a COCO image **or** an image
a human reviewed for one class, while ``coco_scored`` is "COCO answered for all
eighty at once". Measured on today's cell, the loose flag puts **2,138
reviewed-but-off-COCO positives on COCO's side -- 41% of all 5,237 positives**, a
gap that widened when #3926 folded 4,709 corrections into ``exhaustive``.
``coco_scored`` is the flag a provenance question wants.

**2. The negative pool must be excluded.** Run over the whole cell, the probe is
handed a pool that is 100% ``coco_scored`` against positives that are 55%
off-COCO, so a large part of what it scores is positive-versus-negative, not
provenance. The whole-cell number is ~0.85 and means little; positives-only is
~0.577.

**3. Composition has to be matched away.** Off-COCO supply runs from 4 to 82 per
``class@band`` cell, so a pooled probe can read class and band as provenance.
Drawing equal COCO / off-COCO arms **within** each cell holds both constant, and
about half the pooled signal is composition: 0.577 falls to **0.546**.

So the honest reading is the last one, and it is small: off-COCO images are
distinguishable from COCO ones at matched band and class at roughly 0.53-0.55,
consistently across embedders. ``provenance_shortcut.py``'s docstring quotes this
script's old "AUC 0.53-0.56" -- coincidentally close to the corrected matched
number, but arrived at from the wrong partition.

Per-cell fits are **not** usable at this supply: 35 cells clear 25 per arm and
their AUCs run 0.215-0.752, straddling chance in both directions, which is what a
probe with no signal at n=25 per arm looks like.

    python provenance_probe.py [out.json]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")

import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("provenance_probe.json")
SEED = 0

#: A cell contributes to the matched draw once it holds this many per arm. The
#: threshold is about the DRAW, not about a per-cell verdict -- see the docstring
#: on why per-cell AUCs are noise at this supply.
MIN_PER_ARM = 15

EMBEDDERS = ("siglip", "siglip2_l", "clip", "clip_l", "dinov3_patch")


def vec(media: dict, embedder: str) -> np.ndarray | None:
    """The whole-image vector, without `or`-chaining numpy arrays.

    A truthiness test on an ndarray raises, so each candidate key is checked for
    presence explicitly. Patch cells carry a grid as well; this probe wants the
    single whole-image vector, which is what a text sort ranks against.
    """
    emb = media.get("embeddings")
    if not isinstance(emb, dict) or not emb:
        return None
    for key in (embedder, "whole_image"):
        if key in emb and emb[key] is not None:
            v = np.asarray(emb[key], dtype=np.float32).ravel()
            return v if v.size else None
    for value in emb.values():
        if value is None:
            continue
        v = np.asarray(value, dtype=np.float32).ravel()
        if v.size:
            return v
    return None


def _auc(X: np.ndarray, y: np.ndarray, folds: int = 5) -> tuple[float, float]:
    """Cross-validated AUC of a linear probe, on L2-normalised rows.

    These are cosine spaces, and an unnormalised probe can read vector NORM,
    which is a different property from direction.
    """
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=cv, scoring="roc_auc")
    return float(auc.mean()), float(auc.std(ddof=1))


def _balanced(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Equal-sized classes, so AUC is not read off a skewed prior."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(len(pos), len(neg))
    keep = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    return X[keep], y[keep]


def main() -> None:
    report: dict[str, dict] = {}

    for embedder in EMBEDDERS:
        pkl = pc.EMBEDDINGS / f"vg_scale__{embedder}.pkl"
        if not pkl.exists():
            report[embedder] = {"error": "cell missing"}
            continue
        rng = np.random.default_rng(SEED)
        m = load_medias(pkl)

        rows = [(vec(d, embedder), d) for d in m.values()]
        rows = [(v, d) for v, d in rows if v is not None]
        X = np.asarray([v for v, _ in rows], dtype=np.float32)
        coco = np.asarray([1 if d.get("coco_scored") else 0 for _, d in rows], dtype=np.int8)
        is_pos = np.asarray([1 if d.get("category") else 0 for _, d in rows], dtype=np.int8)

        entry: dict = {
            "n": int(len(X)),
            "n_positives": int(is_pos.sum()),
            "n_off_coco_positives": int(((coco == 0) & (is_pos == 1)).sum()),
        }

        # Positives only: the pool is 100% coco_scored, so including it scores
        # positive-vs-negative under a provenance heading.
        Xp, yp = X[is_pos == 1], coco[is_pos == 1]
        if min(int((yp == 1).sum()), int((yp == 0).sum())) >= 50:
            Xb, yb = _balanced(Xp, yp, rng)
            mean, sd = _auc(Xb, yb)
            entry["positives_pooled"] = {"auc": round(mean, 4), "sd": round(sd, 4), "n": int(len(yb))}

        # Matched: equal arms drawn INSIDE each class@band cell, so composition
        # is identical between arms by construction.
        cells: dict[str, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
        for (v, d), is_coco in zip(rows, coco):
            cat = d.get("category")
            if cat:
                cells[cat][0 if is_coco else 1].append(v)
        Xs: list[np.ndarray] = []
        ys: list[int] = []
        used = 0
        for _cell, (a, b) in cells.items():
            k = min(len(a), len(b))
            if k < MIN_PER_ARM:
                continue
            used += 1
            Xs += [a[i] for i in rng.choice(len(a), k, replace=False)]
            Xs += [b[i] for i in rng.choice(len(b), k, replace=False)]
            ys += [1] * k + [0] * k
        if used:
            mean, sd = _auc(np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int8))
            entry["positives_matched"] = {
                "auc": round(mean, 4),
                "sd": round(sd, 4),
                "n": int(len(ys)),
                "cells": used,
            }

        report[embedder] = entry
        pooled = entry.get("positives_pooled", {}).get("auc")
        matched = entry.get("positives_matched", {}).get("auc")
        print(
            f"{embedder:14s} positives pooled {pooled if pooled is not None else '--'}"
            f"   matched {matched if matched is not None else '--'}"
            f"   (off-COCO positives {entry['n_off_coco_positives']:,})"
        )

    OUT.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
