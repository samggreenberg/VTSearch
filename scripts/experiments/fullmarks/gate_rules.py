"""Accept rules for the structural path's returned set, scored offline (#4367).

Reads the per-page frames ``sota_documents.py --frames`` saves (clicks 10 / 25 / 50)
and scores the rules pre-registered on #4367 on each class's held-out test half:

``R0``  shipped: verified and inliers >= 8
``R1``  Bad ceiling: verified and inliers > max(8, the Bads' best inliers)
``R2``  vote-fit: verified and inliers >= t, t maximising F1 over the votes (Goods
        leave-one-out, Bads as they are; ties to the lowest t; floor 8)
``R3``  R2 plus every page whose Stage-1 score is >= the 10th percentile of the Goods'
        leave-one-out Stage-1 scores, verified or not
``R4``  oracle: the best-F1 cut of the ranking (verified by inliers, then Stage 1)

Writes ``rules.csv`` (class x click x rule: precision, recall, F1, returned) and
``summary.md`` (means, then each rule paired against R0, overall and by class
size).

    python gate_rules.py --frames <run>/frames --out <dir>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

MIN_INLIERS = 8
RULES = ("R0", "R1", "R2", "R3", "R4")
LARGE = 50  # positives: the pre-registered split by class size
BOOTSTRAP = 10000


def prf(accept: np.ndarray, positive: np.ndarray) -> tuple[float, float, float, int]:
    k, tp, pos = int(accept.sum()), int((accept & positive).sum()), int(positive.sum())
    precision = tp / k if k else float("nan")
    recall = tp / pos if pos else float("nan")
    f1 = 2 * tp / (k + pos) if (k + pos) else float("nan")
    return precision, recall, f1, k


def vote_fit_threshold(good: np.ndarray, bad: np.ndarray) -> float:
    """The inlier threshold that best separates the votes (F1 over the votes); ties to the lowest."""
    good = good[~np.isnan(good)]
    bad = bad[~np.isnan(bad)]
    if good.size == 0:
        return float(MIN_INLIERS)
    candidates = np.unique(np.concatenate([good, bad, [MIN_INLIERS]]))
    best_t, best_f1 = float(MIN_INLIERS), -1.0
    for t in sorted(candidates):
        tp = int((good >= t).sum())
        fp = int((bad >= t).sum())
        f1 = 2 * tp / (2 * tp + fp + (good.size - tp)) if tp else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return max(float(MIN_INLIERS), best_t)


def accept(rule: str, z: Any) -> np.ndarray:
    inl = np.nan_to_num(z["inliers"], nan=-1.0)
    verified = z["shortlisted"]
    if rule == "R0":
        return verified & (inl >= MIN_INLIERS)
    if rule == "R1":
        bad = z["bad_inliers"][~np.isnan(z["bad_inliers"])]
        ceiling = max(float(MIN_INLIERS - 1), float(bad.max()) if bad.size else MIN_INLIERS - 1)
        return verified & (inl > ceiling)
    t = vote_fit_threshold(z["good_loo_inliers"], z["bad_inliers"])
    base = verified & (inl >= t)
    if rule == "R2":
        return base
    if rule == "R3":
        g = z["good_loo_stage1"][~np.isnan(z["good_loo_stage1"])]
        if g.size == 0:
            return base
        return base | (z["stage1"] >= np.percentile(g, 10))
    raise KeyError(rule)


def oracle(z: Any, test: np.ndarray) -> np.ndarray:
    """R4: the best-F1 cut of the app's order (verified pages by inliers first, then Stage 1)."""
    inl = np.nan_to_num(z["inliers"], nan=-1.0)
    key = np.where(z["shortlisted"], 1e6 + inl, z["stage1"])
    idx = np.flatnonzero(test)
    order = idx[np.argsort(-key[idx], kind="stable")]
    hits = z["positive"][order]
    pos = int(hits.sum())
    out = np.zeros(len(test), dtype=bool)
    if pos == 0:
        return out
    tp = np.cumsum(hits)
    f1 = 2 * tp / (np.arange(1, len(hits) + 1) + pos)
    out[order[: int(f1.argmax()) + 1]] = True
    return out


def paired(d: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(BOOTSTRAP, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for f in sorted(args.frames.glob("*.npz")):
        cid, v = f.stem.rsplit("__v", 1)
        z = np.load(f)
        test = z["test"].astype(bool)
        positive = z["positive"].astype(bool)
        for rule in RULES:
            acc = oracle(z, test) if rule == "R4" else accept(rule, z)
            p, r, f1, k = prf(acc[test], positive[test])
            rows.append(
                {
                    "class_id": cid.replace("__", "/", 1),
                    "v": int(v),
                    "rule": rule,
                    "n_positive": int(positive.sum()),
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "returned": k,
                }
            )
    with (args.out / "rules.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    val = {(r["class_id"], r["v"], r["rule"]): r for r in rows}
    classes = sorted({r["class_id"] for r in rows})
    vs = sorted({r["v"] for r in rows})
    out = [f"{len(classes)} classes.", "", "### Means over classes (precision / recall / F1)", ""]
    out += ["| rule | " + " | ".join(f"{v} clicks" for v in vs) + " |", "|---|" + "---|" * len(vs)]
    for rule in RULES:
        cells = []
        for v in vs:
            sel = [val[(c, v, rule)] for c in classes if (c, v, rule) in val]

            def m(key: str, sel: list[dict[str, Any]] = sel) -> float:
                xs = [x[key] for x in sel if not np.isnan(x[key])]
                return float(np.mean(xs)) if xs else float("nan")

            cells.append(f"{m('precision'):.2f} / {m('recall'):.2f} / {m('f1'):.2f}")
        out.append(f"| {rule} | " + " | ".join(cells) + " |")
    out += ["", "### Paired against R0: F1 (and precision on classes with >= 50 positives)", ""]
    out += [
        "| rule | clicks | classes | F1 diff | 95% | precision diff, >= 50 | 95% |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for rule in RULES[1:]:
        for v in vs:
            d = [
                val[(c, v, rule)]["f1"] - val[(c, v, "R0")]["f1"]
                for c in classes
                if (c, v, rule) in val
                and (c, v, "R0") in val
                and not np.isnan(val[(c, v, rule)]["f1"])
                and not np.isnan(val[(c, v, "R0")]["f1"])
            ]
            big = [
                val[(c, v, rule)]["precision"] - val[(c, v, "R0")]["precision"]
                for c in classes
                if (c, v, rule) in val
                and val[(c, v, rule)]["n_positive"] >= LARGE
                and not np.isnan(val[(c, v, rule)]["precision"])
                and not np.isnan(val[(c, v, "R0")]["precision"])
            ]
            m, lo, hi = paired(np.array(d))
            bm, blo, bhi = paired(np.array(big)) if big else (float("nan"),) * 3
            out.append(
                f"| {rule} | {v} | {len(d)} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {bm:+.3f} | [{blo:+.3f}, {bhi:+.3f}] |"
            )
    text = "\n".join(out) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
