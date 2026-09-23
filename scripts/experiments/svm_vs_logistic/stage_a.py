"""#3197 Stage A: the heads on FIXED vote sets, one (dataset, embedder, category) per task.

Stage B runs each head inside the Autopilot loop, where the head also chooses
the next vote - so a difference there mixes "a better head on the same votes"
with "a head that asks better questions".  Stage A removes the second half:
every arm is fitted on the *same* vote set and scored on the *same* held-out
half, so a paired difference is purely what the objective did with the votes.

Two vote-set sources, both drawn from the harness's own simulation half (the
split is ``RandomState(seed).permutation`` of the sorted media ids, exactly as
``voting_iterations._split_media_ids`` draws it, so Stage A's test half IS the
harness's test half):

* ``controlled`` conditions - composed to switch one candidate mechanism on or
  off (see ``CONDITIONS``);
* ``replay`` - the vote sets Stage B's own Autopilot runs actually collected,
  read from each arm's ``picks`` side frame and cut at fixed click counts.

Hardness is defined by a **reference ranker** fitted on the whole simulation
half with its true labels (a supervised skyline the heads never see): a hard
negative is one it scores near the top of the negatives, an easy one near the
bottom.  The test half is never touched by it.

    python stage_a.py --dataset coco_val --embedder siglip --category dog --out DIR
    python stage_a.py ... --replay-root /expscratch/$USER/svmlog-3197/stageB
"""

from __future__ import annotations

import argparse
import gzip
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import heads as H  # noqa: E402

DATADIR = Path("/expscratch/sgreenberg/svmlog-3197/datadir/embeddings")
PILE = Path("/expscratch/sgreenberg/vts-cache/datadir/embeddings")
SEEDS = (0, 1, 2, 3, 4)
SIZES = (4, 8, 16, 32)  # Good votes per set; Bad votes = RATIO x this
RATIO = 4
HARD_BAND = 0.05  # top 5% of the simulation negatives by reference score
REPLAY_T = (10, 20, 40, 80, 150)
REPLAY_ARMS = ("svm", "linear")

#: condition -> (levels, arms).  "all" = every fast arm; "core" = the arms a
#: manipulation is read off.  The slow arms (lr_ep2000, mlp) ride only on the
#: base condition and the replays.
CONDITIONS = {
    "base": "the reference set: random Goods, half hard / half random Bads",
    "dilute": "a uniformly HARD core plus k x its own size in EASY votes (M2)",
    "noise": "the base set with a fraction r of its votes flipped (M3)",
    "misvote": "the base set with its hardest Bad relabelled Good - one boundary mis-vote (M3)",
    "ratio": "Bad:Good ratio swept at 8 Goods (M5)",
    "shuffled": "every label in the cell permuted - the #3945 control",
}
DILUTE_K = (0, 1, 3, 9)
NOISE_R = (0.0, 0.05, 0.1, 0.2, 0.3)
RATIOS = (1, 4, 16)
CORE_ARMS = (
    "svm",
    "lr",
    "svm_C0.1",
    "svm_C10",
    "svm_hinge",
    "lrconv_C0.1",
    "lrconv_C1",
    "lrconv_C10",
    "lrconv_C100",
    "ridge_a1",
    "centroid",
    "lr_nosmooth",
    "lr_zeroinit",
)


def load_cell(dataset: str, embedder: str) -> tuple[np.ndarray, list[int], dict[int, dict]]:
    """``(X, ids, medias)`` for one cell; ``embedder@raw`` reads the pile un-normalised."""
    from _cells_io import load_medias

    raw = embedder.endswith("@raw")
    emb = embedder.removesuffix("@raw")
    path = (PILE if raw else DATADIR) / f"{dataset}__{emb}.pkl"
    medias = load_medias(path)
    ids = sorted(medias)
    X = np.stack([np.asarray(medias[i]["embeddings"][emb], dtype=np.float32) for i in ids])
    if not raw:
        n = np.linalg.norm(X, axis=1)
        assert np.all(np.abs(n - 1) < 1e-3), f"{path} is not unit-norm (make_datadir.py)"
    return X, ids, medias


def harness_split(ids: list[int], seed: int, sim_fraction: float = 0.5) -> tuple[list[int], list[int]]:
    """``voting_iterations._split_media_ids`` with the natural-prevalence RNG stream."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(sorted(ids)).tolist()
    n_sim = max(1, int(len(shuffled) * sim_fraction))
    return shuffled[:n_sim], shuffled[n_sim:]


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    from vtscore.eval.label_curve import _auroc

    return float(_auroc(np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.float64)))


def avg_prec(s: np.ndarray, y: np.ndarray) -> float:
    from vtscore.eval.label_curve import _average_precision

    return float(_average_precision(np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.float64)))


def oracle_cost(s: np.ndarray, y: np.ndarray) -> float:
    from vtscore.eval.calibration_metrics import inclusion_weights, oracle_cut

    wfp, wfn = inclusion_weights(0)
    return float(oracle_cut(s, y, wfp, wfn)[1])


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


class Cell:
    """One (dataset, embedder, category, seed): the split, labels, and reference."""

    def __init__(self, X: np.ndarray, ids: list[int], labels: np.ndarray, seed: int):
        pos_of = {cid: i for i, cid in enumerate(ids)}
        sim, test = harness_split(ids, seed)
        self.sim = np.array([pos_of[c] for c in sim])
        self.test = np.array([pos_of[c] for c in test])
        self.sim_ids = sim
        self.X = X
        self.y = labels
        ys = labels[self.sim]
        from sklearn.linear_model import LogisticRegression

        ref = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        ref.fit(X[self.sim].astype(np.float64), ys.astype(int))
        self.ref = ref.decision_function(X[self.sim].astype(np.float64))
        self.pos = self.sim[ys == 1]
        self.neg = self.sim[ys == 0]
        rp = self.ref[ys == 1]
        rn = self.ref[ys == 0]
        self.pos_by_ref = self.pos[np.argsort(rp)]  # ascending: hardest Goods first
        self.neg_by_ref = self.neg[np.argsort(-rn)]  # descending: hardest Bads first
        self.n_hard = max(1, int(round(HARD_BAND * len(self.neg))))

    # ---- vote-set composers: each returns (row indices, labels) ----
    def base(self, rng, n_pos: int, ratio: int = RATIO, hard_frac: float = 0.5):
        n_neg = ratio * n_pos
        if n_pos > len(self.pos) or n_neg > len(self.neg):
            return None
        p = rng.choice(self.pos, n_pos, replace=False)
        n_h = int(round(hard_frac * n_neg))
        hard = rng.choice(self.neg_by_ref[: max(self.n_hard, n_h)], n_h, replace=False)
        rest = np.setdiff1d(self.neg, hard)
        rnd = rng.choice(rest, n_neg - n_h, replace=False)
        idx = np.concatenate([p, hard, rnd])
        return idx, self.y[idx].copy()

    def hard_core(self, rng, n_pos: int):
        n_neg = RATIO * n_pos
        half = self.pos_by_ref[: max(n_pos, len(self.pos_by_ref) // 2)]
        if n_pos > len(half) or n_neg > self.n_hard * 2:
            return None
        p = rng.choice(half, n_pos, replace=False)
        band = self.neg_by_ref[: max(self.n_hard, n_neg)]
        n = rng.choice(band, n_neg, replace=False)
        return np.concatenate([p, n])

    def dilute(self, rng, n_pos: int, k: int):
        core = self.hard_core(rng, n_pos)
        if core is None:
            return None
        if k == 0:
            return core, self.y[core].copy()
        easy_pos_pool = self.pos_by_ref[-max(1, len(self.pos_by_ref) // 4) :]
        easy_neg_pool = self.neg_by_ref[len(self.neg_by_ref) // 2 :]
        n_ep, n_en = k * n_pos, k * RATIO * n_pos
        # Easy Goods are scarce on rare categories; sample with replacement
        # rather than drop the level (a repeated easy vote is still an easy vote,
        # and duplicates are exactly the "no new information" the mechanism says
        # the SVM ignores).
        ep = rng.choice(easy_pos_pool, n_ep, replace=n_ep > len(easy_pos_pool))
        en = rng.choice(easy_neg_pool, n_en, replace=n_en > len(easy_neg_pool))
        idx = np.concatenate([core, ep, en])
        return idx, self.y[idx].copy()

    def noise(self, rng, n_pos: int, r: float):
        b = self.base(rng, n_pos)
        if b is None:
            return None
        idx, y = b
        n_flip = int(round(r * len(idx)))
        if n_flip:
            flip = rng.choice(len(idx), n_flip, replace=False)
            y[flip] = 1 - y[flip]
        if y.min() == y.max():
            return None
        return idx, y

    def misvote(self, rng, n_pos: int):
        b = self.base(rng, n_pos)
        if b is None:
            return None
        idx, y = b
        ref_rank = {r: i for i, r in enumerate(self.neg_by_ref)}
        negs = [j for j in range(len(idx)) if y[j] == 0]
        hardest = min(negs, key=lambda j: ref_rank[idx[j]])
        y[hardest] = 1
        return idx, y


def evaluate(
    fits: dict, X: np.ndarray, idx: np.ndarray, y: np.ndarray, test: np.ndarray, y_test: np.ndarray
) -> list[dict]:
    Xtr = X[idx]
    ref_w = {k: fits[k].w for k in ("svm", "lrconv_C1", "centroid") if k in fits}
    rows = []
    for name, fit in fits.items():
        s_te = fit.score(X[test])
        s_tr = fit.score(Xtr)
        row = {
            "arm": name,
            "test_auroc": auroc(s_te, y_test),
            "test_ap": avg_prec(s_te, y_test),
            "test_oracle_cost": oracle_cost(s_te, y_test),
            "train_auroc": auroc(s_tr, y),
            "fit_seconds": fit.seconds,
            "wnorm": float(np.linalg.norm(fit.w)) if fit.w is not None else float("nan"),
            "cos_svm": _cos(fit.w, ref_w.get("svm")),
            "cos_lrconv1": _cos(fit.w, ref_w.get("lrconv_C1")),
            "cos_centroid": _cos(fit.w, ref_w.get("centroid")),
            "epochs": getattr(fit, "epochs", float("nan")),
        }
        if fit.w is not None:
            ypm = np.where(y == 1, 1.0, -1.0)
            # Fraction of the votes the hinge still "sees": margin < 1.  For the
            # SVM arms this is the support-vector fraction; for every other arm
            # it says where the same votes would sit under the SVM's own loss.
            row["active_frac"] = float(np.mean(ypm * s_tr < 1.0))
            row["train_margin_min"] = float(np.min(ypm * s_tr))
        rows.append(row)
    return rows


def fit_all(arms: dict, X: np.ndarray, y: np.ndarray) -> dict:
    return {name: f(X, y) for name, f in arms.items()}


_PICKS_CACHE: dict = {}


def replay_sets(replay_root: Path, dataset: str, embedder: str, category: str, seed: int) -> dict:
    """``{(arm, t): (picked ids, labels)}`` from Stage B's picks side frames.

    Reads ``<replay_root>/<arm>/picks_all.csv.gz``, the per-arm concatenation
    ``harvest_picks.py`` writes once the arm drains.
    """
    import pandas as pd

    out = {}
    for arm in REPLAY_ARMS:
        path = replay_root / arm / "picks_all.csv.gz"
        if not path.exists():
            continue
        allp = _PICKS_CACHE.get(path)
        if allp is None:
            allp = pd.read_csv(path)
            _PICKS_CACHE[path] = allp
        df = allp[
            (allp["dataset"] == dataset)
            & (allp["embedder"] == embedder)
            & (allp["category"] == category)
            & (allp["seed"] == seed)
        ].sort_values("t")
        if df.empty:
            continue
        for t in REPLAY_T:
            # The harness numbers clicks from 1, so "the first t clicks" is
            # t <= T.  (A ``< T`` cut left T-1 rows and skipped every replay.)
            sub = df[df["t"] <= t]
            if len(sub) != t or sub["picked_label"].nunique() < 2:
                continue
            out[(arm, t)] = (sub["picked_id"].astype(int).tolist(), sub["picked_label"].astype(int).to_numpy())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--replay-root", type=Path, default=None)
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()

    import vtscore

    assert str(Path(vtscore.__file__).resolve()).startswith(str(REPO)), (
        f"vtscore resolved to {vtscore.__file__}, not this worktree {REPO} (source gridenv.sh)"
    )
    from vtscore.eval.labels import media_is_positive

    X, ids, medias = load_cell(args.dataset, args.embedder)
    labels = np.array([1 if media_is_positive(medias[i], args.category) else 0 for i in ids])
    fast = H.arm_table(include_slow=False)
    everything = H.arm_table(include_slow=True)
    core = {k: fast[k] for k in CORE_ARMS}
    conds = [c for c in args.conditions.split(",") if c]

    args.out.mkdir(parents=True, exist_ok=True)
    slug = f"{args.dataset}__{args.embedder}__{args.category}".replace(" ", "_").replace("/", "_")
    suffix = "__replay" if args.replay_only else ""
    out_path = args.out / f"{slug}{suffix}.csv.gz"
    meta: dict = {
        "dataset": args.dataset,
        "embedder": args.embedder,
        "category": args.category,
        "n_medias": len(ids),
        "n_pos": int(labels.sum()),
        "fidelity": None,
    }
    rows: list[dict] = []
    t_start = time.time()

    def emit(seed, cond, level, n_pos, n_neg, *, size, recs):
        for r in recs:
            r.update(
                dataset=args.dataset,
                embedder=args.embedder,
                category=args.category,
                seed=seed,
                condition=cond,
                level=level,
                size=size,
                n_pos_votes=n_pos,
                n_neg_votes=n_neg,
            )
            rows.append(r)

    for seed in [int(s) for s in args.seeds.split(",")]:
        cell = Cell(X, ids, labels, seed)
        yt = labels[cell.test]
        rng = np.random.RandomState(1000 + seed)
        if meta["fidelity"] is None:
            b = cell.base(rng, 8) or cell.base(rng, 4)
            if b is not None:
                meta["fidelity"] = H.fidelity(X[b[0]], b[1])

        if not args.replay_only:
            if "base" in conds:
                for n in SIZES:
                    b = cell.base(rng, n)
                    if b is None:
                        continue
                    idx, y = b
                    emit(
                        seed,
                        "base",
                        0,
                        n,
                        len(idx) - n,
                        size=n,
                        recs=evaluate(fit_all(everything, X[idx], y), X, idx, y, cell.test, yt),
                    )
            if "dilute" in conds:
                for n in (4, 8):
                    # One draw of the hard core per (seed, n), reused at every
                    # k, so the levels differ ONLY in the easy votes added.
                    state = rng.randint(1 << 30)
                    for k in DILUTE_K:
                        d = cell.dilute(np.random.RandomState(state), n, k)
                        if d is None:
                            continue
                        idx, y = d
                        emit(
                            seed,
                            "dilute",
                            k,
                            int(y.sum()),
                            int(len(y) - y.sum()),
                            size=n,
                            recs=evaluate(fit_all(core, X[idx], y), X, idx, y, cell.test, yt),
                        )
            if "noise" in conds:
                for n in (8, 16):
                    state = rng.randint(1 << 30)
                    for r in NOISE_R:
                        d = cell.noise(np.random.RandomState(state), n, r)
                        if d is None:
                            continue
                        idx, y = d
                        emit(
                            seed,
                            "noise",
                            r,
                            n,
                            RATIO * n,
                            size=n,
                            recs=evaluate(fit_all(core, X[idx], y), X, idx, y, cell.test, yt),
                        )
            if "misvote" in conds:
                for n in (4, 8, 16):
                    state = rng.randint(1 << 30)
                    for lvl, fn in ((0, lambda r, n=n: cell.base(r, n)), (1, lambda r, n=n: cell.misvote(r, n))):
                        d = fn(np.random.RandomState(state))
                        if d is None:
                            continue
                        idx, y = d
                        emit(
                            seed,
                            "misvote",
                            lvl,
                            n,
                            RATIO * n,
                            size=n,
                            recs=evaluate(fit_all(core, X[idx], y), X, idx, y, cell.test, yt),
                        )
            if "ratio" in conds:
                for ratio in RATIOS:
                    d = cell.base(rng, 8, ratio=ratio)
                    if d is None:
                        continue
                    idx, y = d
                    emit(
                        seed,
                        "ratio",
                        ratio,
                        8,
                        ratio * 8,
                        size=8,
                        recs=evaluate(fit_all(core, X[idx], y), X, idx, y, cell.test, yt),
                    )
            if "shuffled" in conds:
                perm = np.random.RandomState(7000 + seed).permutation(len(labels))
                shuf = labels[perm]
                scell = Cell(X, ids, shuf, seed)
                for n in (8, 16):
                    d = scell.base(rng, n)
                    if d is None:
                        continue
                    idx, y = d
                    emit(
                        seed,
                        "shuffled",
                        0,
                        n,
                        RATIO * n,
                        size=n,
                        recs=evaluate(fit_all(core, X[idx], y), X, idx, y, scell.test, shuf[scell.test]),
                    )
                    # The REAL twin at the same size and the same draw state, so
                    # the control's two rows are a pair.
                    d = cell.base(rng, n)
                    if d is not None:
                        idx, y = d
                        emit(
                            seed,
                            "shuffled",
                            1,
                            n,
                            RATIO * n,
                            size=n,
                            recs=evaluate(fit_all(core, X[idx], y), X, idx, y, cell.test, yt),
                        )

        if args.replay_root is not None:
            pos_of = {cid: i for i, cid in enumerate(ids)}
            test_set = set(cell.test.tolist())
            for (arm, t), (pids, plabels) in replay_sets(
                args.replay_root, args.dataset, args.embedder, args.category, seed
            ).items():
                idx = np.array([pos_of[p] for p in pids])
                if test_set.intersection(idx.tolist()):
                    raise AssertionError(f"replayed votes overlap the test half: {arm} t={t} seed={seed}")
                if not np.array_equal(plabels, labels[idx]):
                    raise AssertionError(f"replayed labels disagree with the cell's: {arm} t={t} seed={seed}")
                emit(
                    seed,
                    f"replay_{arm}",
                    t,
                    int(plabels.sum()),
                    int(len(plabels) - plabels.sum()),
                    size=t,
                    recs=evaluate(fit_all(everything, X[idx], plabels), X, idx, plabels, cell.test, yt),
                )

    if args.replay_only and not rows:
        # Every Stage B cell has 150 picks with both classes by click 150, so a
        # replay that yields nothing is a bug, not a result.
        raise SystemExit(f"replay produced 0 rows for {slug}: check the picks file and the click cut")
    meta["seconds"] = round(time.time() - t_start, 1)
    meta["n_rows"] = len(rows)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with gzip.open(out_path, "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    (args.out / f"{slug}{suffix}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
