"""#3945 Stage R: the SVM head's regularisation path along real Autopilot sessions.

"How many votes does it take to overtrain?" is a question about one fitted
object - the shipped ``Linear(D, 1)`` SVM head - as its vote set grows.  This
script takes the vote sets Autopilot actually collected (the ``picks`` side
frame of a Stage B arm, #3197's or this study's 400-click one), cuts each
session at a ladder of click counts, and at every cut refits the head along a
fine grid of ``C``.  Every fit is scored three ways:

* on the harness's own **held-out half** (never voted on, true labels): the
  ground truth for "is it over-fitting";
* on its **own votes** (train AUROC): what memorisation looks like from inside;
* on **vote-only gauges** the app could compute today, with no ground truth:
  ``app`` = the shipped cross-calibration splits (``calibrate_count`` = 2,
  ``calibration_fraction`` = 0.3 for a single-vector space), held-out AUROC on
  the calibrate part; ``cv5`` = stratified 5-fold CV on the votes.  Whether a
  gauge tracks the held-out truth, and whether picking C by it lands near the
  best C, is the "how do we detect it" half of the issue.

Replaying the ``svm`` arm's votes at another C is **off-policy**: those votes
were chosen by the C = 1 head.  The report says what that does and does not
license; an in-loop arm confirms anything that would ship.

    python c_path.py --dataset coco_val --embedder siglip --category dog \
        --replay-root /expscratch/$USER/svmlog-3197/stageB --arm svm --out DIR
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "svm_vs_logistic"))
sys.path.insert(0, str(HERE.parent / "calibration"))

import heads as H  # noqa: E402
import stage_a as SA  # noqa: E402

#: Half-decade grid, wide enough that both ends are clearly worse than the
#: interior on every environment #3197 measured (its A_bestC picked 0.01..1).
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
T_GRID = (5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400)
SEEDS = (0, 1, 2, 3, 4)
APP_SPLITS = 2  # DEFAULT_CALIBRATE_COUNT
APP_FRACTION = 0.3  # production_split_for(patch_space=False), #3287
CV_FOLDS = 5


def fit_svm_w(X: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    """LinearSVC exactly as the shipped head builds it (``heads.fit_svm``), weights only."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC

    clf = LinearSVC(C=C, class_weight="balanced", dual="auto", max_iter=5000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y).astype(int))
    return clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])


def _split_auroc(X: np.ndarray, y: np.ndarray, splits, C: float) -> tuple[float, float]:
    """Mean held-out AUROC and AP over ``splits``; NaN when no split is usable."""
    au, apv = [], []
    for tr, te in splits:
        if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
            continue
        w, b = fit_svm_w(X[tr], y[tr], C)
        s = X[te] @ w + b
        au.append(SA.auroc(s, y[te]))
        apv.append(SA.avg_prec(s, y[te]))
    if not au:
        return float("nan"), float("nan")
    return float(np.mean(au)), float(np.mean(apv))


def app_splits(y: np.ndarray, seed: int) -> list:
    """The shipped cross-calibration draws: stratified Train/Calibrate, 2 of them."""
    from sklearn.model_selection import StratifiedShuffleSplit

    n_pos = int(y.sum())
    if n_pos < 2 or len(y) - n_pos < 2:
        return []
    sss = StratifiedShuffleSplit(n_splits=APP_SPLITS, test_size=APP_FRACTION, random_state=seed)
    try:
        return list(sss.split(np.zeros(len(y)), y))
    except ValueError:
        return []


def cv_splits(y: np.ndarray, seed: int) -> list:
    from sklearn.model_selection import StratifiedKFold

    k = min(CV_FOLDS, int(y.sum()), int(len(y) - y.sum()))
    if k < 2:
        return []
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def replay_cuts(root: Path, arm: str, dataset: str, embedder: str, category: str, seed: int, cache: dict):
    """``{t: (picked ids, labels)}`` for every ``T_GRID`` cut the session reached."""
    import pandas as pd

    path = root / arm / "picks_all.csv.gz"
    allp = cache.get(path)
    if allp is None:
        allp = pd.read_csv(path)
        cache[path] = allp
    df = allp[
        (allp["dataset"] == dataset)
        & (allp["embedder"] == embedder)
        & (allp["category"] == category)
        & (allp["seed"] == seed)
    ].sort_values("t")
    out = {}
    for t in T_GRID:
        sub = df[df["t"] <= t]  # clicks are numbered from 1
        if len(sub) != t:
            continue
        out[t] = (sub["picked_id"].astype(int).tolist(), sub["picked_label"].astype(int).to_numpy())
    return out, len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--replay-root", required=True, type=Path)
    ap.add_argument("--arm", default="svm")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()

    import vtscore

    assert str(Path(vtscore.__file__).resolve()).startswith(str(REPO)), (
        f"vtscore resolved to {vtscore.__file__}, not this worktree {REPO} (source gridenv.sh)"
    )
    from vtscore.eval.labels import media_is_positive

    X, ids, medias = SA.load_cell(args.dataset, args.embedder)
    labels = np.array([1 if media_is_positive(medias[i], args.category) else 0 for i in ids])
    pos_of = {cid: i for i, cid in enumerate(ids)}

    args.out.mkdir(parents=True, exist_ok=True)
    slug = f"{args.dataset}__{args.embedder}__{args.category}".replace(" ", "_").replace("/", "_")
    rows: list[dict] = []
    meta: dict = {
        "dataset": args.dataset,
        "embedder": args.embedder,
        "category": args.category,
        "arm": args.arm,
        "n_medias": len(ids),
        "n_pos": int(labels.sum()),
        "fidelity": None,
        "session_len": {},
    }
    t0 = time.time()
    cache: dict = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        # Only the split is needed from Cell; its reference ranker is not used here.
        sim, test = SA.harness_split(ids, seed)
        test_idx = np.array([pos_of[c] for c in test])
        yt = labels[test_idx]
        Xte = X[test_idx]
        cuts, n_session = replay_cuts(
            args.replay_root, args.arm, args.dataset, args.embedder, args.category, seed, cache
        )
        meta["session_len"][seed] = n_session
        test_set = set(test_idx.tolist())
        for t, (pids, y) in cuts.items():
            idx = np.array([pos_of[p] for p in pids])
            if test_set.intersection(idx.tolist()):
                raise AssertionError(f"replayed votes overlap the test half: t={t} seed={seed}")
            if not np.array_equal(y, labels[idx]):
                raise AssertionError(f"replayed labels disagree with the cell's: t={t} seed={seed}")
            n_pos = int(y.sum())
            base = {
                "dataset": args.dataset,
                "embedder": args.embedder,
                "category": args.category,
                "seed": seed,
                "t": t,
                "n_pos_votes": n_pos,
                "n_neg_votes": int(len(y) - n_pos),
                "n_unique": len(set(pids)),
            }
            if n_pos == 0 or n_pos == len(y):
                rows.append({**base, "C": float("nan"), "one_class": 1})
                continue
            Xv = X[idx]
            if meta["fidelity"] is None and n_pos >= 2:
                meta["fidelity"] = H.fidelity(Xv, y)  # raises if the replica is not the shipped fit
            asp, cvs = app_splits(y, seed), cv_splits(y, seed)
            for C in C_GRID:
                w, b = fit_svm_w(Xv, y, C)
                s_te = Xte @ w + b
                s_tr = Xv @ w + b
                ypm = np.where(y == 1, 1.0, -1.0)
                app_au, app_ap = _split_auroc(Xv, y, asp, C)
                cv_au, cv_ap = _split_auroc(Xv, y, cvs, C)
                rows.append(
                    {
                        **base,
                        "C": C,
                        "one_class": 0,
                        "test_auroc": SA.auroc(s_te, yt),
                        "test_ap": SA.avg_prec(s_te, yt),
                        "test_oracle_cost": SA.oracle_cost(s_te, yt),
                        "train_auroc": SA.auroc(s_tr, y),
                        "train_err": float(np.mean(ypm * s_tr <= 0)),
                        "active_frac": float(np.mean(ypm * s_tr < 1.0)),
                        "wnorm": float(np.linalg.norm(w)),
                        "app_auroc": app_au,
                        "app_ap": app_ap,
                        "cv_auroc": cv_au,
                        "cv_ap": cv_ap,
                    }
                )
    if not rows:
        raise SystemExit(f"0 rows for {slug}: check the picks file ({args.replay_root}/{args.arm})")
    meta["seconds"] = round(time.time() - t0, 1)
    meta["n_rows"] = len(rows)
    out_path = args.out / f"{slug}.csv.gz"
    keys = sorted({k for r in rows for k in r})
    with gzip.open(out_path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (args.out / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps({k: meta[k] for k in ("dataset", "embedder", "category", "n_rows", "seconds", "fidelity")}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
