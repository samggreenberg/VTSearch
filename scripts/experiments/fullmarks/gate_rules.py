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

#4434 adds two rules on top of R1 (the shipped line since #4367), using each
verified page's best-fit geometry (frames saved with ``ratio`` / ``reproj``):

``M1``  R1 and inlier ratio >= r* and median reprojection error <= e*, the cuts fit on
        tier ``s``'s template matrix (``--fit-cuts``), never on the scored tier
``M2``  R1 and P(true) >= 0.5 under a logistic regression on (log inliers, ratio,
        reprojection error) fit on the detector's votes (Goods leave-one-out, Bads);
        R1 until there are >= 3 of each

Writes ``rules.csv`` (class x click x rule: precision, recall, F1, returned) and
``summary.md`` (means, then each rule paired against R0, overall and by class
size).

    python gate_rules.py --frames <run>/frames --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

MIN_INLIERS = 8
RULES = ("R0", "R1", "R2", "R3", "R4")
#: #4434's rules, scored when the frames carry the fit geometry.
GEOMETRY_RULES = ("M1", "M2")
MIN_VOTES_M2 = 3
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


def fit_cuts(matrix: Path) -> dict[str, float]:
    """M1's cuts: the (ratio, reprojection) pair maximising F1 over box-template pairs past the gate.

    Reads #4162's template matrix for one tier. Box templates only (template 0, the query
    crop, is a cross-scale fit whose residuals mean something else, #3349).
    """
    ratios, reprojs, labels = [], [], []
    for f in sorted(matrix.glob("*.npz")):
        if f.name.startswith("vectors-"):
            continue
        z = np.load(f)
        st = z["stats"].astype(np.float32)[1:]  # box templates
        if st.shape[0] == 0:
            continue
        pos = np.broadcast_to(z["positives"].astype(bool), st.shape[:2])
        inl = np.expm1(st[..., 0])
        gate = (st[..., 8] > 0.5) & (inl >= MIN_INLIERS)
        ratios.append(st[..., 1][gate])
        reprojs.append(st[..., 4][gate])
        labels.append(pos[gate])
    r, e, y = np.concatenate(ratios), np.concatenate(reprojs), np.concatenate(labels)
    best = (-1.0, 0.0, float("inf"))
    for rc in np.quantile(r, np.linspace(0, 0.95, 20)):
        for ec in np.quantile(e, np.linspace(0.05, 1.0, 20)):
            acc = (r >= rc) & (e <= ec)
            tp = int((acc & y).sum())
            f1 = 2 * tp / (int(acc.sum()) + int(y.sum())) if acc.any() else 0.0
            if f1 > best[0]:
                best = (f1, float(rc), float(ec))
    return {"ratio_min": best[1], "reproj_max": best[2], "pair_f1": best[0], "pairs": int(y.size)}


def accept_geometry(rule: str, z: Any, cuts: Optional[dict[str, float]]) -> np.ndarray:
    """#4434's M1 / M2 on top of R1."""
    base = accept("R1", z)
    ratio = np.nan_to_num(z["ratio"], nan=-1.0)
    reproj = np.nan_to_num(z["reproj"], nan=np.inf)
    if rule == "M1":
        assert cuts is not None, "M1 needs --cuts"
        return base & (ratio >= cuts["ratio_min"]) & (reproj <= cuts["reproj_max"])
    # M2: logistic on the votes
    g = np.column_stack([np.log1p(z["good_loo_inliers"]), z["good_loo_ratio"], z["good_loo_reproj"]])
    b = np.column_stack([np.log1p(z["bad_inliers"]), z["bad_ratio"], z["bad_reproj"]])
    g, b = g[np.isfinite(g).all(axis=1)], b[np.isfinite(b).all(axis=1)]
    if len(g) < MIN_VOTES_M2 or len(b) < MIN_VOTES_M2:
        return base
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(
        np.vstack([g, b]), np.r_[np.ones(len(g)), np.zeros(len(b))]
    )
    x = np.column_stack(
        [np.log1p(np.nan_to_num(z["inliers"], nan=0.0)), ratio, np.where(np.isfinite(reproj), reproj, 1.0)]
    )
    p = model.predict_proba(x)[:, 1]
    return base & (p >= 0.5)


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
    ap.add_argument("--fit-cuts", type=Path, help="a tier's template matrix: fit M1's cuts there and write cuts.json")
    ap.add_argument("--cuts", type=Path, help="M1's cuts.json (fit on another tier)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fit_cuts:
        cuts = fit_cuts(args.fit_cuts)
        (args.out / "cuts.json").write_text(json.dumps(cuts, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(cuts, indent=2))
        return 0
    cuts = json.loads(args.cuts.read_text(encoding="utf-8")) if args.cuts else None
    rules: tuple[str, ...] = RULES

    rows: list[dict[str, Any]] = []
    for f in sorted(args.frames.glob("*.npz")):
        cid, v = f.stem.rsplit("__v", 1)
        z = np.load(f)
        test = z["test"].astype(bool)
        positive = z["positive"].astype(bool)
        geometry = "ratio" in z.files
        if geometry and rules == RULES:
            rules = RULES + (("M1",) if cuts else ()) + ("M2",)
        for rule in rules:
            if rule == "R4":
                acc = oracle(z, test)
            elif rule in GEOMETRY_RULES:
                acc = accept_geometry(rule, z, cuts)
            else:
                acc = accept(rule, z)
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
    for rule in rules:
        cells = []
        for v in vs:
            sel = [val[(c, v, rule)] for c in classes if (c, v, rule) in val]

            def m(key: str, sel: list[dict[str, Any]] = sel) -> float:
                xs = [x[key] for x in sel if not np.isnan(x[key])]
                return float(np.mean(xs)) if xs else float("nan")

            cells.append(f"{m('precision'):.2f} / {m('recall'):.2f} / {m('f1'):.2f}")
        out.append(f"| {rule} | " + " | ".join(cells) + " |")
    controls = [("R0", list(rules[1:]))]
    if any(r in GEOMETRY_RULES for r in rules):
        controls.append(("R1", [r for r in rules if r in GEOMETRY_RULES]))  # #4434's control: the shipped line
    for control, compared in controls:
        out += ["", f"### Paired against {control}: F1 (and precision on classes with >= 50 positives)", ""]
        out += [
            "| rule | clicks | classes | F1 diff | 95% | precision diff, >= 50 | 95% |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
        for rule in compared:
            for v in vs:
                d = [
                    val[(c, v, rule)]["f1"] - val[(c, v, control)]["f1"]
                    for c in classes
                    if (c, v, rule) in val
                    and (c, v, control) in val
                    and not np.isnan(val[(c, v, rule)]["f1"])
                    and not np.isnan(val[(c, v, control)]["f1"])
                ]
                big = [
                    val[(c, v, rule)]["precision"] - val[(c, v, control)]["precision"]
                    for c in classes
                    if (c, v, rule) in val
                    and val[(c, v, rule)]["n_positive"] >= LARGE
                    and not np.isnan(val[(c, v, rule)]["precision"])
                    and not np.isnan(val[(c, v, control)]["precision"])
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
