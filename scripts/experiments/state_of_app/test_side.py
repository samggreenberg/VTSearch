#!/usr/bin/env python3
"""State of the App (#4179): the TEST-side list -- held-out images the detectors confidently get wrong.

    python test_side.py --out <dir> [--top-neg 10] [--bottom-pos 5] [--random-neg 5]

``harmful_pairs.csv`` names the images that hurt when CLICKED. Most images are
never clicked, but every one of them is scored, so a label error there costs
every detector in every seed. This names those.

**Flagged by several embedders, never by the shipped one alone.** A test-side
fix always moves in the flagging detector's favour: a confident false positive
that really holds the class becomes a true positive. Letting SigLIP pick the
fixes would make the benchmark agree with SigLIP and deny credit to a later
model that is right where SigLIP is wrong. So each of the five built columns
(``siglip``, ``siglip2_l``, ``clip``, ``clip_l``, ``dinov3_patch``'s whole-image
vector) scores every cell independently, and the list is their UNION.

**Scores are out-of-fold.** Per cell and embedder, a balanced logistic head on
the cell's own labels, 5-fold stratified, so no image is scored by a head that
saw its label. That is a full-label detector, not the app's click loop: the
question is which labels the evidence disputes, not what the app would do.

Per cell and embedder this keeps the ``--top-neg`` highest-scoring negatives and
the ``--bottom-pos`` lowest-scoring positives. A class's three bands share
their negatives, so flags are pooled per (image, class, label) with how many
(band, embedder) heads flagged it, which orders the review.

**The random arm is the guard.** ``--random-neg`` negatives per class that NO
head flagged, asked the same question. Their Good rate is the label-error rate
outside the flagged set: the background that says how much the flagged fixes
could inflate a score, and whether the residual matters.

Writes ``test_pairs.csv`` in ``harmful_pairs.csv``'s shape plus ``arm``,
``n_flags`` and ``band``, for ``harmful_review.py build --extend``.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

from _cells_io import iter_medias  # noqa: E402

EMBEDDERS = ("siglip", "siglip2_l", "clip", "clip_l", "dinov3_patch")


def _read(cells_dir: Path, dataset: str, emb: str):
    ids, X, cats, evals = [], [], [], []
    for iid, m in iter_medias(cells_dir / f"{dataset}__{emb}.pkl"):
        v = m["embeddings"][emb]
        ids.append(int(iid))
        X.append(np.asarray(v, dtype=np.float32))
        cats.append(set(m.get("categories") or []))
        evals.append(set(m.get("evaluable_categories") or []))
    X = np.stack(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return np.array(ids), X, cats, evals


def _oof(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    out = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(X[tr], y[tr])
        out[te] = clf.decision_function(X[te])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=Path, default=Path("/expscratch/sgreenberg/vts-cache/datadir/embeddings"))
    ap.add_argument("--dataset", default="coco_better")
    ap.add_argument("--embedders", default=",".join(EMBEDDERS))
    ap.add_argument("--top-neg", type=int, default=10)
    ap.add_argument("--bottom-pos", type=int, default=5)
    ap.add_argument("--random-neg", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4179)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    negs_of: dict[str, set[int]] = collections.defaultdict(set)
    for emb in args.embedders.split(","):
        ids, X, cats, evals = _read(args.cells, args.dataset, emb)
        cells = sorted({c for s in evals for c in s})
        print(f"{emb}: {len(ids):,} medias, {len(cells)} cells", flush=True)
        for cell in cells:
            idx = np.array([i for i, e in enumerate(evals) if cell in e])
            y = np.array([cell in cats[i] for i in idx], dtype=int)
            if y.sum() < 5:
                continue
            s = _oof(X[idx], y, args.seed)
            cls, band = cell.rsplit("@", 1)
            neg, pos = idx[y == 0], idx[y == 1]
            negs_of[cls] |= set(ids[neg].tolist())
            sn, sp = s[y == 0], s[y == 1]
            # A negative's rank among the positives says how confident the error is.
            for j in np.argsort(-sn)[: args.top_neg]:
                rows.append((int(ids[neg[j]]), cls, 0, band, emb, float(sn[j]), float((sp > sn[j]).mean())))
            for j in np.argsort(sp)[: args.bottom_pos]:
                rows.append((int(ids[pos[j]]), cls, 1, band, emb, float(sp[j]), float((sn > sp[j]).mean())))
    flags = pd.DataFrame(rows, columns=["image_id", "class", "label", "band", "embedder", "score", "beat_share"])
    flags.to_csv(args.out / "test_flags.csv", index=False)

    g = flags.groupby(["image_id", "class", "label"])
    pairs = g.agg(
        n_flags=("embedder", "size"),
        n_embedders=("embedder", "nunique"),
        band=("band", lambda b: b.value_counts().index[0]),
        beat_share=("beat_share", "max"),
    ).reset_index()
    pairs["arm"] = "test_flagged"

    # The random arm: negatives no head flagged, per class, the same question.
    rng = random.Random(args.seed)
    flagged = set(zip(flags["image_id"], flags["class"], strict=True))
    rand = []
    for cls in sorted(negs_of):
        pool = sorted(i for i in negs_of[cls] if (i, cls) not in flagged)
        for iid in rng.sample(pool, min(args.random_neg, len(pool))):
            rand.append(
                {"image_id": iid, "class": cls, "label": 0, "n_flags": 0, "n_embedders": 0, "arm": "test_random"}
            )
    pairs = pd.concat([pairs.sort_values(["n_flags", "beat_share"], ascending=False), pd.DataFrame(rand)])
    pairs.to_csv(args.out / "test_pairs.csv", index=False)
    f = pairs[pairs["arm"] == "test_flagged"]
    print(
        f"{len(f):,} flagged pairs ({int((f['label'] == 0).sum()):,} negatives), "
        f"{len(rand)} random negatives -> {args.out / 'test_pairs.csv'}"
    )
    print(f["n_embedders"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
