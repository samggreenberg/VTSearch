#!/usr/bin/env python3
"""Are off-COCO positives HARDER, at matched band and class? (#3997)

``anchor_to_coco`` claims that dropping VG's non-COCO half loses "VG's non-COCO
diversity for nothing". The ``coco_quarry`` migration plan names settling that
claim as the item that runs first, because it is the only live argument for
keeping Visual Genome. ``provenance_probe.py`` answers half of it -- off-COCO
images are *distinguishable* from COCO ones at AUC 0.53-0.55 once class and band
are matched away. Readable is not the same as harder, and harder is what the
claim asserts.

So ask the shipped head. Per ``class@band`` cell:

* fit :func:`vtscore.training.svm.fit_linear_svm_head` -- the production head, not
  a stand-in -- on a training split of that cell's positives against a training
  split of the shared negative pool;
* score the HELD-OUT positives and a HELD-OUT half of the pool;
* per positive, record where it lands in that ranking.

Class and band are constant inside a cell, so the COCO / off-COCO contrast is
matched by construction. Cells are averaged with equal weight rather than pooled,
so a cell with lots of off-COCO supply cannot dominate.

**The statistic matters more than it looks.** The obvious one -- the fraction of
the pool a positive outranks -- **saturates**: most positives sit at 0.85-1.00, so
a difference living at the top of the ranking is compressed against the ceiling,
and the answer reads as a flat "no effect". The per-image rank COUNTS are
therefore stored, and ``--aggregate`` re-reads them under statistics that do not
saturate: ``log1p(above)``, which weights the top of the ranking, and the share
landing inside the pool's top 1%, which is what a user would actually see.

**The confound this exists to defeat.** Off-COCO positives never got COCO's
adjudication: VG's recall over *C* is 0.61 and 8.3% of its boxes sit on a smaller
instance than the frame's main one (#3924, #3925). "Off-COCO scores lower" is
equally consistent with more of them being mislabelled. #3926's pass verified
positives by hand, so the ``verified`` arm restricts off-COCO to (image, class)
pairs a human confirmed.

**And the confound that arm brings with it.** #3926 verified by asking a human
*is this class present*, so its survivors are conditioned on a human being able to
SEE the object -- which selects for clear instances. The COCO arm has no
equivalent filter and COCO annotates small and occluded instances a reviewer might
have called absent. So the verified arm bounds how much of the gap is labelling;
it is not "label noise removed, true difficulty revealed", and the unrestricted
arm is the one that answers the plan's question.

    python provenance_hardness.py                 # fit, write the raw JSON
    python provenance_hardness.py --aggregate     # re-read it, no fitting
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")

import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

SEED = 0
FOLDS = 5
#: Pool images sampled per cell, split half for training and half for ranking.
N_POOL = 4000
#: A cell contributes once it holds this many per arm.
MIN_ARM = 10
EMBEDDERS = ("siglip", "siglip2_l", "clip")
OUT = Path("provenance_hardness.json")


def vec(media: dict, embedder: str) -> np.ndarray | None:
    emb = media.get("embeddings") or {}
    for key in (embedder, "whole_image"):
        if emb.get(key) is not None:
            v = np.asarray(emb[key], dtype=np.float32).ravel()
            return v if v.size else None
    return None


def verified_pairs(path: Path) -> set[tuple[int, str]]:
    """(image_id, class) pairs a human confirmed PRESENT in #3926's pass."""
    rows = json.loads(path.read_text())
    return {(int(r["image_id"]), r["class"]) for r in rows if r.get("present")}


def score_with_shipped_head(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Fit the production linear-SVM head and return its decision scores on *Xte*."""
    import torch  # noqa: PLC0415

    from vtscore.training.svm import fit_linear_svm_head  # noqa: PLC0415

    model = fit_linear_svm_head(Xtr, ytr, input_dim=Xtr.shape[1], seed=SEED)
    with torch.no_grad():
        dev = next(model.parameters()).device
        out = model(torch.from_numpy(np.ascontiguousarray(Xte)).to(dev))
    return out.detach().cpu().numpy().ravel()


def measure(corrections: Path) -> dict:
    verified = verified_pairs(corrections)
    rng = np.random.default_rng(SEED)
    report: dict = {}

    for embedder in EMBEDDERS:
        pkl = pc.EMBEDDINGS / f"vg_scale__{embedder}.pkl"
        if not pkl.exists():
            continue
        m = load_medias(pkl)

        pool: list[np.ndarray] = []
        cells: dict[str, list] = collections.defaultdict(list)
        for iid, d in m.items():
            v = vec(d, embedder)
            if v is None:
                continue
            cat = d.get("category")
            if not cat:
                pool.append(v)
            else:
                cells[cat].append((v, bool(d.get("coco_scored")), (int(iid), cat.split("@")[0]) in verified))
        pool_arr = np.asarray(pool, dtype=np.float32)
        print(f"\n{embedder}: {len(cells)} cells, pool {len(pool_arr):,}", flush=True)

        rows = []
        for cat in sorted(cells):
            items = cells[cat]
            y_prov = np.array([1 if it[1] else 0 for it in items], dtype=np.int8)
            if min(int((y_prov == 1).sum()), int((y_prov == 0).sum())) < MIN_ARM:
                continue
            X = np.asarray([it[0] for it in items], dtype=np.float32)
            is_ver = np.array([it[2] for it in items], dtype=bool)

            pidx = rng.choice(len(pool_arr), min(N_POOL, len(pool_arr)), replace=False)
            half = len(pidx) // 2
            pool_tr, pool_rank = pool_arr[pidx[:half]], pool_arr[pidx[half:]]

            pct = np.full(len(X), np.nan, dtype=np.float32)
            above = np.full(len(X), np.nan, dtype=np.float32)
            cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
            for tr, te in cv.split(X, y_prov):
                Xtr = np.vstack([X[tr], pool_tr])
                ytr = np.concatenate([np.ones(len(tr), np.int8), np.zeros(len(pool_tr), np.int8)])
                s = score_with_shipped_head(Xtr, ytr, np.vstack([X[te], pool_rank]))
                order = np.sort(s[len(te) :])
                idx = np.searchsorted(order, s[: len(te)], side="left")
                pct[te] = idx / len(order)
                above[te] = len(order) - idx

            ok = ~np.isnan(pct)
            masks = {
                "coco": ok & (y_prov == 1),
                "off": ok & (y_prov == 0),
                "off_verified": ok & (y_prov == 0) & is_ver,
            }
            rows.append(
                {
                    "cell": cat,
                    "pool_ranked": int(len(pool_rank)),
                    "n": {k: int(v.sum()) for k, v in masks.items()},
                    "mean_pct": {k: float(pct[v].mean()) for k, v in masks.items() if v.sum()},
                    "pct": {k: pct[v].round(5).tolist() for k, v in masks.items()},
                    "above": {k: above[v].astype(int).tolist() for k, v in masks.items()},
                }
            )
            mp = rows[-1]["mean_pct"]
            print(
                f"  {cat:22s} coco {mp['coco']:.3f}  off {mp['off']:.3f}"
                + (f"  off-verified {mp['off_verified']:.3f}" if masks["off_verified"].sum() >= MIN_ARM else ""),
                flush=True,
            )
        report[embedder] = rows

    return report


def _arm_stats(above: list[int], pool_n: int) -> dict:
    a = np.asarray(above, dtype=np.float64)
    return {
        "log1p": float(np.log1p(a).mean()),
        "top1pct": float((a <= max(1, pool_n // 100)).mean()),
        "median_above": float(np.median(a)),
    }


def aggregate(report: dict) -> None:
    """Paired per-cell differences under statistics that do not saturate.

    Sign is normalised so that **positive always means off-COCO ranks BETTER**:
    ``log1p`` and ``median_above`` are costs, so their differences are negated.
    """
    for embedder, rows in report.items():
        print(f"\n=== {embedder}  ({len(rows)} cells) ===")
        for arm in ("off", "off_verified"):
            deltas: dict[str, list[float]] = collections.defaultdict(list)
            for r in rows:
                if len(r["above"].get(arm, [])) < MIN_ARM:
                    continue
                c = _arm_stats(r["above"]["coco"], r["pool_ranked"])
                o = _arm_stats(r["above"][arm], r["pool_ranked"])
                deltas["pct"].append(float(np.mean(r["pct"][arm]) - np.mean(r["pct"]["coco"])))
                deltas["log1p"].append(-(o["log1p"] - c["log1p"]))
                deltas["median_above"].append(-(o["median_above"] - c["median_above"]))
                deltas["top1pct"].append(o["top1pct"] - c["top1pct"])
            if not deltas["pct"]:
                continue
            label = "off - coco" if arm == "off" else "off(verified) - coco"
            print(f"  {label}   [{len(deltas['pct'])} cells]   (+ = off-COCO EASIER)")
            for k in ("pct", "log1p", "top1pct", "median_above"):
                d = np.asarray(deltas[k])
                se = d.std(ddof=1) / np.sqrt(len(d))
                up, down = int((d > 0).sum()), int((d < 0).sum())
                print(f"    {k:13s} {d.mean():+8.4f}  +/- {se:.4f} (se)   {up} up / {down} down")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregate", action="store_true", help="re-read the raw JSON; fit nothing")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--corrections", type=Path, default=Path(pc.PILE) / "corrections.json")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(json.loads(args.out.read_text()))
        return
    report = measure(args.corrections)
    args.out.write_text(json.dumps(report) + "\n")
    print(f"\nwrote {args.out}")
    aggregate(report)


if __name__ == "__main__":
    main()
