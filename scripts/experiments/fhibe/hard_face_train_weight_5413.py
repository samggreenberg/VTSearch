"""One session cell with hard-view faces weighted in the trained head's fit (#5413).

#5389 weighted a hard-view starting face (Face Quality under #5372's cut) in the example centroid and
recovered nothing at beta 4: by vote 10 the session's detector is the trained head, the linear SVM
started from the Goods' centroid (:func:`vtscore.training.svm_prior.centroid_prior_direction`), and
the hard face is one of its few Goods. This runs one cell with every voted row of that fit weighted
by its face: a hard view weighs ``--weight`` and a clear face 1, for the Goods (``--on goods``) or
the Goods and the Bads (``--on all``). A class's hard rows are weighted only while that class holds
a clear one. The weight goes in two places:

* the SVM's per-row loss: the balanced class weight times the face's weight, renormalised so each
  class keeps the total mass ``class_weight="balanced"`` gives it (a hard Good counts less than the
  other Goods, and the Goods together count as much as before);
* the Goods' centroid the head starts from, so a hard Good does not steer the start either.

A fit with no row weighted calls the head unchanged, so ``--weight 1`` is the control and arms pair
cell for cell. Rows are matched to their faces by vector; a row that matches no face fails the cell.

    python hard_face_train_weight_5413.py --index N [--weight W] [--on goods|all]

Run from ``scripts/experiments/calibration`` in a run dir's environment (``sota_face.sh cell`` with
``SOTA_FACE_RUNNER`` pointing here, ``HARD_FACE_WEIGHT`` and ``HARD_FACE_ON`` the arm). Each weighted
fit is counted on stderr at the end (``hard_face_train_weight_5413: ...``).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

CALIB = Path(__file__).resolve().parents[1] / "calibration"


def _hard_by_vector(dataset: str) -> dict[bytes, bool]:
    """``{float32 vector bytes: is_hard_view}`` for every face of *dataset*'s pile cell."""
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
    from vtscore.media.face.imagekit_bridge import media_hard_view  # noqa: PLC0415

    pile = os.environ.get("VTS_PILE_CELLS") or str(
        Path(os.environ.get("VTS_FHIBE_ROOT", f"/expscratch/{os.environ['USER']}/fhibe"))
        / f"{os.environ.get('VTS_FHIBE_RELEASE', 'fhibe-full-resolution-674a7dcf')}-derived"
        / "cells"
    )
    with open(Path(pile) / f"{dataset}__face.pkl", "rb") as f:
        medias = pickle.load(f)  # noqa: S301  (our own pile cell)
    out: dict[bytes, bool] = {}
    for m in medias.values():
        vec = media_embedding(m)
        if vec is not None:
            out[np.asarray(vec, dtype=np.float32).ravel().tobytes()] = bool(media_hard_view(m))
    return out


def class_weights(hard: np.ndarray, weight: float) -> np.ndarray:
    """One class's face weights: *weight* for a hard row while some row is clear, else 1."""
    if weight == 1.0 or hard.all() or not hard.any():
        return np.ones(len(hard))
    return np.where(hard, weight, 1.0)


def weighted(original, hard_of: dict[bytes, bool], weight: float, on: str, stats: dict):
    """*original* (``centroid_prior_direction``) with each row weighted by its face; unchanged when none is."""

    def centroid_prior_direction(X, y, *, scale=None, seed=42, sample_weight=None, query=None):
        from sklearn.svm import LinearSVC  # noqa: PLC0415

        from vtscore import config  # noqa: PLC0415
        from vtscore.training import svm_prior  # noqa: PLC0415

        scale = svm_prior.CENTROID_PRIOR_SCALE if scale is None else scale
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        pos = y > 0.5
        keys = [row.astype(np.float32).tobytes() for row in X]
        missing = sum(k not in hard_of for k in keys)
        if missing:
            raise RuntimeError(f"{missing} of {len(keys)} training rows match no face of the pile cell")
        hard = np.array([hard_of[k] for k in keys])
        q = np.ones(len(y))
        q[pos] = class_weights(hard[pos], weight)
        if on == "all":
            q[~pos] = class_weights(hard[~pos], weight)
        stats["fits"] += 1
        if (q == 1.0).all():
            return original(X, y, scale=scale, seed=seed, sample_weight=sample_weight, query=query)
        stats["weighted"] += 1
        stats["hard_goods"] += int((q[pos] != 1.0).sum())
        stats["hard_bads"] += int((q[~pos] != 1.0).sum())
        # Balanced mass per class, shared within the class by the face weights.
        base = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()
        sw = np.empty(len(y))
        for cls in (pos, ~pos):
            mass = len(y) / 2.0 if sample_weight is None else base[cls].sum()
            share = base[cls] * q[cls]
            sw[cls] = mass * share / share.sum()
        goods = X[pos]
        units = goods / np.linalg.norm(goods, axis=1, keepdims=True)
        centroid = np.average(units, axis=0, weights=q[pos])
        centroid /= np.linalg.norm(centroid)
        clf = LinearSVC(C=config.SVM_HEAD_C, class_weight=None, dual=False, max_iter=5000, random_state=seed)
        priors = [(float(scale), centroid)] if scale > 0 else []
        query = svm_prior.active_query_prior() if query is None else query
        if query is not None and query[1] > 0 and np.asarray(query[0]).shape == (X.shape[1],):
            direction = np.asarray(query[0], dtype=np.float64)
            priors.append((float(query[1]), direction / np.linalg.norm(direction)))
        features = [X, *[(s * (X @ d))[:, None] for s, d in priors]]
        clf.fit(np.hstack(features), pos.astype(int), sample_weight=sw)
        coef = np.asarray(clf.coef_, dtype=np.float64).ravel()
        w = coef[: X.shape[1]].copy()
        for k, (s, d) in enumerate(priors):
            w += s * coef[X.shape[1] + k] * d
        return w, float(np.asarray(clf.intercept_).ravel()[0])

    return centroid_prior_direction


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--weight", type=float, default=float(os.environ.get("HARD_FACE_WEIGHT", "1")))
    ap.add_argument("--on", choices=("goods", "all"), default=os.environ.get("HARD_FACE_ON", "goods"))
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CALIB))
    import common  # noqa: PLC0415
    import experiment_config as cfg  # noqa: PLC0415
    import run_cells  # noqa: PLC0415

    from vtscore.training import svm_prior  # noqa: PLC0415

    info = json.loads((common.RESULTS / "prepare_info.json").read_text())
    cell = cfg.array_cells(run_cells._categories_by_dataset(info))[args.index]
    stats = {"index": args.index, "weight": args.weight, "on": args.on, "fits": 0, "weighted": 0}
    stats.update(hard_goods=0, hard_bads=0)
    hard_of = _hard_by_vector(cell["dataset"])
    svm_prior.centroid_prior_direction = weighted(
        svm_prior.centroid_prior_direction, hard_of, args.weight, args.on, stats
    )
    atexit.register(lambda: print("hard_face_train_weight_5413: " + json.dumps(stats), file=sys.stderr, flush=True))
    return run_cells.main(["--index", str(args.index)])


if __name__ == "__main__":
    sys.exit(main())
