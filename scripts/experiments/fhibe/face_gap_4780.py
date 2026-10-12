#!/usr/bin/env python
"""What carries the face review's gaps between groups of people (#4780).

#4762 found the app does worse on FHIBE's face crops for she/her people than he/him (0.73 vs 0.85
after the check, beta 1, one photo, 1024 px), for younger people than older, and for the middle skin
tones. This reads the face reviews' sessions person by person and splits each gap three ways:

* **stages** - where in a session the gap appears: the example sort (click 0), click 25, after the
  check, the final ranking and the full-label ceiling's. The final and ceiling rankings are read off
  the rank frames (``kind`` ``last`` and ``skyline_train_full``), so each has F at three cuts: the
  app's line (``thr_returned_final``, the objective's own count), the true count (``n_test_pos``: the
  ranking's own F, with no line to set) and the best cut. ``line_loss`` = F at the line minus F at
  the true count is what the line costs on top of the ranking.
* **matched** - whether the pronoun and age gaps hold among people whose photos were taken alike.
  Each person's photos are summarised from FHIBE's annotations (``face_gap_4780_covariates.py``):
  face size, camera distance, head pose, camera position, accessories, roll, and, separately, how
  the person looks in them (hair over the face, visible skin, make-up, facial hair, how many
  hairstyles). The gap is re-read (a) in coarsened exact matching on the photo conditions and (b) as
  the group's coefficient in a least-squares fit, raw, with the photo conditions, and with both.
* **arms** - the same split for every arm given (``--set``): plain, aligned, ArcFace in place of
  FaceNet, so an arm that pairs person for person with another gives the gap's change with an SE.

    python face_gap_4780.py --strata <person_strata.csv> --covariates <photo_covariates.csv> \\
        --landmarks <landmarks_1024.pkl> --out <dir> \\
        --set NAME=<analysis dir>::<runs prefix>::<baseline dir>::<faces dataset prefix> ...

Each ``--set``: the ``sota_face_analyze.sh`` output, the run dirs' common prefix (``<runs>/<date>-<tag>``,
read as ``<prefix>-k<K>-b1/results/cells``), the click-0 baseline dir and the dataset prefix
(``fhibe_faces_roll``). Only beta 1 at 1024 px is read: the review's table and #4780's question.
Writes ``people.csv`` (per person x set x K; names people, owner-only), and the aggregates
``stages.csv``, ``matched.csv``, ``regress.csv``, ``photos_by_pronoun.csv``, ``arms.csv``,
``curves_by_group.csv`` (the objective at every click per pronoun, filled as the review fills it) and, given
``--pairs``, ``pairs.csv`` (the genuine-pair read, :func:`pair_table`); given ``--static``,
``static_gaps.csv`` (each embedder's static pronoun and age gaps, with the age-balanced impostor pool).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pile"))

KS = (1, 4)
BETA = 1.0
SIZE = "1024"
SHE, HE = "She/her/hers", "He/him/his"
#: The stage columns, in session order.
STAGES = [
    "click0_ap",
    "click0_f",
    "f_25",
    "session_mean",
    "f_line",
    "precision_line",
    "recall_line",
    "final_ap",
    "final_f_true_count",
    "final_f_best",
    "ceiling_ap",
    "ceiling_f_true_count",
    "ceiling_f_best",
    "line_loss",
    "returned",
    "n_test_pos",
]
#: How the photos were taken (matched on) and how the person looks in them (added after).
CONDITIONS = ["face_px", "camera_distance", "head_atypical", "position_atypical", "accessory", "abs_roll", "outdoor"]
#: Facial hair and make-up are left out: each is (near) zero for one pronoun, so a fit with them cannot
#: tell them from the pronoun itself.
APPEARANCE = ["hairstyles", "frac_hair", "frac_skin", "months"]


# ---------------------------------------------------------------------------
# per person
# ---------------------------------------------------------------------------


def fbeta_at(ranks: np.ndarray, n_pos: int, k: float, beta: float = BETA) -> float:
    """F-beta of the top *k* of a ranking whose positives sit at 0-based *ranks*."""
    k = int(round(k))
    if k <= 0 or n_pos <= 0:
        return 0.0
    tp = int((ranks < k).sum())
    if tp == 0:
        return 0.0
    p, r = tp / k, tp / n_pos
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def fbeta_best(ranks: np.ndarray, n_pos: int, beta: float = BETA) -> float:
    """The best F-beta any cut reaches: only a cut just below a positive can be best."""
    return max((fbeta_at(ranks, n_pos, r + 1, beta) for r in ranks), default=0.0)


def _ranks(s) -> np.ndarray:
    return (
        np.asarray([int(v) for v in str(s).split()], dtype=np.int64)
        if isinstance(s, str) and s.strip()
        else np.zeros(0, np.int64)
    )


def frames(cells_dir: Path, dataset: str) -> pd.DataFrame:
    """Each person's final and ceiling ranking: the positives' test ranks, from ``task_*__rankframes.csv``."""
    rows = []
    for f in sorted(cells_dir.glob("task_*__rankframes.csv")):
        df = pd.read_csv(f, usecols=["dataset", "category", "kind", "n_test_pos", "test_pos_ranks"])
        df = df[(df["dataset"] == dataset) & df["kind"].isin(["last", "skyline_train_full"])]
        for r in df.itertuples():
            rows.append(
                {
                    "category": r.category,
                    "kind": r.kind,
                    "n_test_pos": int(r.n_test_pos),
                    "ranks": _ranks(r.test_pos_ranks),
                }
            )
    return pd.DataFrame(rows)


def person_rows(analysis: Path, runs_prefix: str, baseline: Path, faces: str, k: int) -> pd.DataFrame:
    """One row per person: every stage of their beta-1 session at 1024 px from *k* photos."""
    from align_compare_4777 import session_set  # noqa: PLC0415

    dataset = f"{faces}_{SIZE}"
    cells = pd.read_csv(analysis / f"k{k}-b1-{SIZE}" / "cells.csv")
    cells = cells[cells["dataset"] == dataset].set_index("category")
    sm = session_set(analysis, baseline, k, SIZE, BETA, faces).reset_index().set_index("category")["session_mean"]
    fr = frames(Path(f"{runs_prefix}-k{k}-b1") / "results" / "cells", dataset)
    last = fr[fr["kind"] == "last"].set_index("category")
    sky = fr[fr["kind"] == "skyline_train_full"].set_index("category")
    out = []
    for cat, c in cells.iterrows():
        row = {
            "category": cat,
            "K": k,
            "click0_ap": c["text_ap"],
            "click0_f": c["text_thr_fbeta"],
            "f_25": c["thr_fbeta_25"],
            "session_mean": sm.get(cat, np.nan),
            "f_line": c["thr_fbeta_final"],
            "precision_line": c["thr_precision_final"],
            "recall_line": c["thr_recall_final"],
            "returned": c["thr_returned_final"],
            "final_ap": c["final_ap"],
            "ceiling_ap": c["ceiling_ap"],
        }
        if cat in last.index:
            lr, n = last.loc[cat, "ranks"], int(last.loc[cat, "n_test_pos"])
            row.update(
                n_test_pos=n,
                final_f_true_count=fbeta_at(lr, n, n),
                final_f_best=fbeta_best(lr, n),
                f_line_from_ranks=fbeta_at(lr, n, c["thr_returned_final"]),
            )
        if cat in sky.index:
            sr, n = sky.loc[cat, "ranks"], int(sky.loc[cat, "n_test_pos"])
            row.update(ceiling_f_true_count=fbeta_at(sr, n, n), ceiling_f_best=fbeta_best(sr, n))
        out.append(row)
    df = pd.DataFrame(out)
    df["line_loss"] = df["f_line"] - df["final_f_true_count"]
    return df


def group_curves(analysis: Path, baseline: Path, faces: str, k: int, strata: pd.DataFrame) -> pd.DataFrame:
    """The objective at every click, averaged per pronoun: each run's curve filled as the review fills it."""
    sys.path.insert(0, str(HERE.parent / "state_of_app"))
    from by_click import RUN, filled_runs, typed_query  # noqa: PLC0415

    d = analysis / f"k{k}-b1-{SIZE}"
    cells = pd.read_csv(d / "cells.csv")
    base = pd.read_csv(baseline / f"example_baseline_k{k}.csv")
    base = base[base["dataset"] == f"{faces}_{SIZE}"]
    text = typed_query(base, "face", BETA)
    filled, _ = filled_runs(pd.read_csv(d / "curves.csv"), cells, text)
    filled[0] = [text.get((cat, int(seed)), np.nan) for cat, seed in filled.index]
    pron = strata.set_index("category")["pronoun"].reindex(filled.index.get_level_values(RUN[0])).to_numpy()
    rows = []
    for who in (SHE, HE):
        w = filled[pron == who]
        for t in sorted(c for c in w.columns if 0 <= int(c) <= 150):
            v = w[t].dropna()
            rows.append(
                {"K": k, "pronoun": who, "t": int(t), "mean": v.mean(), "se": v.std() / np.sqrt(len(v)), "n": len(v)}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# per person covariates
# ---------------------------------------------------------------------------


def subject_roll(landmarks: Path) -> pd.Series:
    """|roll| in degrees of each photo's subject detection: the eye line's angle (MTCNN's landmarks)."""
    from pilebuild.loaders import fhibe  # noqa: PLC0415

    rows = pickle.loads(landmarks.read_bytes())  # noqa: S301  # #4777's detect stage
    out = {}
    for r in rows:
        if r["boxes"] is None:
            continue
        best, best_iou = None, -1.0
        for i, box in enumerate(r["boxes"]):
            v = fhibe.iou(r["face"], list(box))
            if v > best_iou or (v == best_iou and r["probs"][i] > r["probs"][best]):
                best, best_iou = i, v
        if best is None or best_iou < fhibe.SUBJECT_IOU:
            continue
        (lx, ly), (rx, ry) = r["points"][best][0], r["points"][best][1]
        out[r["pid"]] = abs(np.degrees(np.arctan2(ry - ly, rx - lx)))
    return pd.Series(out, name="abs_roll")


def person_covariates(covariates: Path, landmarks: Path | None, people: set[str]) -> pd.DataFrame:
    """Each person's photos summarised: the shares and means :data:`CONDITIONS` and :data:`APPEARANCE` name."""
    ph = pd.read_csv(covariates)
    ph = ph[ph["subject"].isin(people)].copy()
    ph["hairstyle"] = ph["hairstyle"].fillna("None")
    if landmarks is not None:
        ph = ph.merge(subject_roll(landmarks), left_on="pid", right_index=True, how="left")
    else:
        ph["abs_roll"] = np.nan
    ph["head_atypical"] = (ph["head_pose"] == 1).astype(float)
    ph["position_atypical"] = (ph["camera_position"] > 0).astype(float)
    ph["accessory"] = ((ph["has_headwear"] + ph["has_eyewear"] + ph["has_mask"]) > 0).astype(float)
    ph["outdoor"] = ph["scene_outdoor"].astype(float)
    ph["face_px"] = ph["face_px_1024"]
    g = ph.groupby("subject")
    agg = g[
        [
            "face_px",
            "camera_distance",
            "head_atypical",
            "position_atypical",
            "accessory",
            "abs_roll",
            "outdoor",
            "frac_hair",
            "frac_skin",
            "makeup",
            "facial_hair",
        ]
    ].mean()
    agg["hairstyles"] = g["hairstyle"].nunique()  # "None" (no hair) is a hairstyle: read_csv makes it NaN
    agg["hairstyle_list"] = g["hairstyle"].agg(lambda v: "; ".join(f"{k} x{n}" for k, n in v.value_counts().items()))
    agg["months"] = g["captured"].nunique()
    agg["phones"] = g["phone"].nunique()
    return agg.rename_axis("category").reset_index()


# ---------------------------------------------------------------------------
# reading a gap
# ---------------------------------------------------------------------------


def gap(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """mean(a) - mean(b) and its Welch SE."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return float(a.mean() - b.mean()), float(np.sqrt(a.var() / len(a) + b.var() / len(b)))


def ols(y: np.ndarray, X: np.ndarray, clusters: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with HC1 (heteroskedasticity-robust) SEs, or cluster-robust ones over *clusters*."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    bread = np.linalg.pinv(X.T @ X)
    if clusters is None:
        meat = (X * resid[:, None] ** 2).T @ X
        cov = bread @ meat @ bread * n / max(n - p, 1)
    else:
        u = pd.DataFrame(X * resid[:, None]).groupby(clusters).sum().to_numpy()
        g = len(u)
        cov = bread @ (u.T @ u) @ bread * g / max(g - 1, 1) * (n - 1) / max(n - p, 1)
    return beta, np.sqrt(np.diag(cov))


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    z = df[cols].astype(float)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    return z.to_numpy()


def regress(
    df: pd.DataFrame, outcome: str, group_cols: dict[str, np.ndarray], models: dict[str, list[str]]
) -> list[dict]:
    """The groups' coefficients on *outcome* under each covariate set in *models*."""
    rows = []
    for name, covs in models.items():
        d = df.dropna(subset=[outcome, *covs])
        cols = [np.ones(len(d))] + [v[d.index] for v in group_cols.values()]
        X = np.column_stack(cols + ([_design(d, covs)] if covs else []))
        b, se = ols(d[outcome].to_numpy(float), X)
        for j, g in enumerate(group_cols, start=1):
            rows.append({"outcome": outcome, "model": name, "term": g, "coef": b[j], "se": se[j], "n": len(d)})
    return rows


def cem(df: pd.DataFrame, outcome: str, treat: pd.Series, bins: dict[str, list[float]]) -> dict:
    """Coarsened exact matching: the treated-minus-control gap within each stratum, weighted by the treated."""
    keys = pd.DataFrame({c: pd.cut(df[c], edges, include_lowest=True, labels=False) for c, edges in bins.items()})
    d = df.assign(_t=treat.astype(bool), _k=keys.astype("Int64").astype(str).agg("|".join, axis=1))
    d = d.dropna(subset=[outcome])
    diffs, weights, var = [], [], []
    kept_t = kept_c = 0
    for _, s in d.groupby("_k"):
        t, c = s[s["_t"]][outcome], s[~s["_t"]][outcome]
        if len(t) == 0 or len(c) == 0:
            continue
        diffs.append(t.mean() - c.mean())
        weights.append(len(t))
        var.append((t.var(ddof=1) if len(t) > 1 else 0) / len(t) + (c.var(ddof=1) if len(c) > 1 else 0) / len(c))
        kept_t += len(t)
        kept_c += len(c)
    w = np.asarray(weights, float) / max(sum(weights), 1)
    return {
        "outcome": outcome,
        "gap": float(np.dot(w, diffs)) if diffs else np.nan,
        "se": float(np.sqrt(np.dot(w**2, var))) if diffs else np.nan,
        "treated_matched": kept_t,
        "treated": int(d["_t"].sum()),
        "control_matched": kept_c,
        "strata": len(diffs),
    }


def pair_table(pairs: Path, covariates: Path, strata: pd.DataFrame) -> pd.DataFrame:
    """The pronoun gap in genuine-pair cosine, raw, among matched pairs, and adjusted, per embedder.

    A genuine pair is two photos of one person. Matched pairs share a hairstyle and a capture month,
    and both are typical head poses without headwear, eyewear or a mask: photos taken alike, of a
    person who looks alike in them. ``gap_sd`` puts each gap in units of the embedder's own spread of
    genuine cosines, so FaceNet and ArcFace compare. SEs cluster on the person.
    """
    pr = pd.read_csv(pairs)
    ph = pd.read_csv(covariates).set_index("pid")
    ph["hairstyle"] = ph["hairstyle"].fillna("None")
    ph["clean"] = (ph["head_pose"] == 0) & (ph["has_headwear"] + ph["has_eyewear"] + ph["has_mask"] == 0)
    a, b = ph.reindex(pr["id_a"]).reset_index(drop=True), ph.reindex(pr["id_b"]).reset_index(drop=True)
    pr["same_hairstyle"] = (a["hairstyle"] == b["hairstyle"]).to_numpy(float)
    pr["same_month"] = (a["captured"] == b["captured"]).to_numpy(float)
    pr["both_clean"] = (a["clean"] & b["clean"]).to_numpy(float)
    pr["d_distance"] = (a["camera_distance"] - b["camera_distance"]).abs().to_numpy(float)
    pr["face_px"] = ((a["face_px_1024"] + b["face_px_1024"]) / 2).to_numpy(float)
    pr["hair"] = ((a["frac_hair"] + b["frac_hair"]) / 2).to_numpy(float)
    pr = pr.merge(strata[["category", "pronoun"]], on="category")
    pr = pr[pr["pronoun"].isin([SHE, HE])].reset_index(drop=True)
    rows = []
    for cell, d in pr.groupby("cell", sort=False):
        d = d.reset_index(drop=True)
        sd = d["cos"].std()
        she = (d["pronoun"] == SHE).to_numpy(float)
        subsets = {
            "all pairs": np.ones(len(d), bool),
            "same hairstyle": d["same_hairstyle"].to_numpy() == 1,
            "different hairstyle": d["same_hairstyle"].to_numpy() == 0,
            "matched: same hairstyle and month, both clean": (d[["same_hairstyle", "same_month", "both_clean"]] == 1)
            .all(axis=1)
            .to_numpy(),
        }
        for name, m in subsets.items():
            x = np.column_stack([np.ones(m.sum()), she[m]])
            bb, se = ols(d.loc[m, "cos"].to_numpy(float), x, d.loc[m, "category"].to_numpy())
            rows.append(
                {
                    "cell": cell,
                    "read": name,
                    "pairs": int(m.sum()),
                    "she_pairs": int(she[m].sum()),
                    "gap": bb[1],
                    "se": se[1],
                    "gap_sd": bb[1] / sd,
                }
            )
        covs = ["same_hairstyle", "same_month", "both_clean", "d_distance", "face_px", "hair"]
        x = np.column_stack([np.ones(len(d)), she, _design(d, covs)])
        bb, se = ols(d["cos"].to_numpy(float), x, d["category"].to_numpy())
        rows.append(
            {
                "cell": cell,
                "read": "adjusted for the pair's covariates",
                "pairs": len(d),
                "she_pairs": int(she.sum()),
                "gap": bb[1],
                "se": se[1],
                "gap_sd": bb[1] / sd,
            }
        )
        for who in (SHE, HE):
            w = d[d["pronoun"] == who]
            g, e = gap(w.loc[w["same_hairstyle"] == 1, "cos"], w.loc[w["same_hairstyle"] == 0, "cos"])
            rows.append(
                {
                    "cell": cell,
                    "read": f"same minus different hairstyle, {who}",
                    "pairs": len(w),
                    "she_pairs": np.nan,
                    "gap": g,
                    "se": e,
                    "gap_sd": g / sd,
                }
            )
        rows.append(
            {
                "cell": cell,
                "read": "share of pairs with the same hairstyle, she / he",
                "pairs": len(d),
                "she_pairs": np.nan,
                "gap": d.loc[she == 1, "same_hairstyle"].mean(),
                "se": d.loc[she == 0, "same_hairstyle"].mean(),
                "gap_sd": np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, help="NAME=<analysis>::<runs prefix>::<baseline>::<faces>")
    ap.add_argument("--strata", type=Path, required=True)
    ap.add_argument("--covariates", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, default=None)
    ap.add_argument("--static", type=Path, default=None, help="static_people.csv (face_gap_4780_static.py)")
    ap.add_argument("--pairs", type=Path, default=None, help="static_pairs.csv (face_gap_4780_static.py)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    strata = pd.read_csv(args.strata, dtype={"skin_tone": str})
    people = set(strata["category"])
    cov = person_covariates(args.covariates, args.landmarks, people)
    frames_, curves = [], []
    for spec in args.set:
        name, rest = spec.split("=", 1)
        analysis, prefix, baseline, faces = rest.split("::")
        for k in KS:
            df = person_rows(Path(analysis), prefix, Path(baseline), faces, k)
            df.insert(0, "set", name)
            frames_.append(df)
            cv = group_curves(Path(analysis), Path(baseline), faces, k, strata)
            cv.insert(0, "set", name)
            curves.append(cv)
            print(f"{name} K{k}: {len(df)} people", flush=True)
    ppl = (
        pd.concat(frames_, ignore_index=True)
        .merge(strata, on="category", how="left")
        .merge(cov, on="category", how="left")
    )
    if args.static is not None:
        st = pd.read_csv(args.static)
        vals = [c for c in ["loo_ap", "loo_ap_balanced", "genuine_mean", "impostor_top1", "margin"] if c in st]
        wide = st.pivot_table(index="category", columns="cell", values=vals)
        wide.columns = [f"static_{c}_{v}" for v, c in wide.columns]
        ppl = ppl.merge(wide.reset_index(), on="category", how="left")
    args.out.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        ppl.to_csv(args.out / "people.csv", index=False)
    finally:
        os.umask(old)
    pd.concat(curves, ignore_index=True).round(4).to_csv(args.out / "curves_by_group.csv", index=False)
    chk = (ppl["f_line"] - ppl["f_line_from_ranks"]).abs()
    print(f"F at the line re-read from the ranks: max |diff| {chk.max():.4f}, share > 0.01 {np.mean(chk > 0.01):.3f}")

    # --- stages: each stratum against its reference -------------------------------------------
    contrasts = {
        "pronoun": ("pronoun", SHE, HE),
        "age 18-29 vs 45+": ("age_group", "18-29", "45+"),
        "skin 2-3 vs 0": ("skin_group", "2-3", "0"),
    }
    ppl["age_group"] = np.where(ppl["age"] < 30, "18-29", np.where(ppl["age"] >= 45, "45+", "30-44"))
    ppl["skin_group"] = ppl["skin_tone"].map({"0": "0", "2": "2-3", "3": "2-3"}).fillna("other")
    stage_rows = []
    for (s, k), d in ppl.groupby(["set", "K"], sort=False):
        for cname, (col, a, b) in contrasts.items():
            for st in STAGES:
                if st not in d:
                    continue
                g, se = gap(d.loc[d[col] == a, st], d.loc[d[col] == b, st])
                stage_rows.append(
                    {
                        "set": s,
                        "K": k,
                        "contrast": cname,
                        "stage": st,
                        "a_mean": d.loc[d[col] == a, st].mean(),
                        "b_mean": d.loc[d[col] == b, st].mean(),
                        "gap": g,
                        "se": se,
                    }
                )
    stages = pd.DataFrame(stage_rows)
    stages.round(4).to_csv(args.out / "stages.csv", index=False)

    # --- matched and adjusted --------------------------------------------------------------------
    bins = {
        "face_px": list(np.nanquantile(cov["face_px"], [0, 1 / 3, 2 / 3, 1])),
        "camera_distance": [-1, 1.5, 2.5, 5],
        "head_atypical": [-0.01, 0.0, 0.34, 1.0],
        "accessory": [-0.01, 0.0, 1.0],
    }
    outcomes = ["f_line", "session_mean", "final_ap", "ceiling_ap", "final_f_true_count", "line_loss"]
    match_rows, reg_rows = [], []
    for (s, k), d in ppl.groupby(["set", "K"], sort=False):
        d = d.reset_index(drop=True)
        pron = d[d["pronoun"].isin([SHE, HE])].reset_index(drop=True)
        for o in outcomes:
            r = cem(pron, o, pron["pronoun"] == SHE, bins)
            raw, raw_se = gap(pron.loc[pron["pronoun"] == SHE, o], pron.loc[pron["pronoun"] == HE, o])
            match_rows.append({"set": s, "K": k, "contrast": "pronoun", "raw_gap": raw, "raw_se": raw_se, **r})
            young = d[d["age_group"].isin(["18-29", "45+"])].reset_index(drop=True)
            r = cem(young, o, young["age_group"] == "18-29", bins)
            raw, raw_se = gap(young.loc[young["age_group"] == "18-29", o], young.loc[young["age_group"] == "45+", o])
            match_rows.append({"set": s, "K": k, "contrast": "age 18-29 vs 45+", "raw_gap": raw, "raw_se": raw_se, **r})
        groups = {
            "she": (d["pronoun"] == SHE).to_numpy(float),
            "age_per_decade": (d["age"].fillna(d["age"].median()) / 10).to_numpy(float),
            **{f"skin_{t}": (d["skin_tone"] == t).to_numpy(float) for t in ("1", "2", "3", "4", "5")},
        }
        models = {
            "raw": [],
            "+conditions": CONDITIONS,
            "+conditions+hairstyles": CONDITIONS + ["hairstyles"],
            "+conditions+appearance": CONDITIONS + APPEARANCE,
        }
        for o in outcomes:
            for row in regress(d, o, groups, models):
                reg_rows.append({"set": s, "K": k, **row})
    pd.DataFrame(match_rows).round(4).to_csv(args.out / "matched.csv", index=False)
    pd.DataFrame(reg_rows).round(4).to_csv(args.out / "regress.csv", index=False)

    # --- what differs between the groups' photos -----------------------------------------------
    one = ppl.drop_duplicates("category")
    diff_rows = []
    for c in CONDITIONS + APPEARANCE + ["months", "phones", "n_photos"]:
        g, se = gap(one.loc[one["pronoun"] == SHE, c], one.loc[one["pronoun"] == HE, c])
        diff_rows.append(
            {
                "covariate": c,
                "she": one.loc[one["pronoun"] == SHE, c].mean(),
                "he": one.loc[one["pronoun"] == HE, c].mean(),
                "gap": g,
                "se": se,
            }
        )
    pd.DataFrame(diff_rows).round(4).to_csv(args.out / "photos_by_pronoun.csv", index=False)

    if args.static is not None:
        st = pd.read_csv(args.static)
        rows = []
        for cell, g in st.groupby("cell", sort=False):
            young = g["age_bracket"] == "18-29"
            older = g["age_bracket"].isin(["45-59", "60+"])
            for c in [v for v in ["loo_ap", "loo_ap_balanced", "genuine_mean", "impostor_top1", "margin"] if v in g]:
                for cname, a, b in (
                    ("pronoun", g["pronoun"] == SHE, g["pronoun"] == HE),
                    ("age 18-29 vs 45+", young, older),
                ):
                    gg, se = gap(g.loc[a, c], g.loc[b, c])
                    rows.append(
                        {
                            "cell": cell,
                            "contrast": cname,
                            "measure": c,
                            "a_mean": g.loc[a, c].mean(),
                            "b_mean": g.loc[b, c].mean(),
                            "gap": gg,
                            "se": se,
                        }
                    )
        pd.DataFrame(rows).round(4).to_csv(args.out / "static_gaps.csv", index=False)

    if args.pairs is not None:
        pair_table(args.pairs, args.covariates, strata).round(4).to_csv(args.out / "pairs.csv", index=False)

    # --- arms: the pronoun gap's change between sets, person for person --------------------------
    sets = list(dict.fromkeys(ppl["set"]))
    arm_rows = []
    for k in KS:
        for st in ["session_mean", "f_line", "final_ap", "ceiling_ap", "final_f_true_count", "line_loss", "click0_ap"]:
            w = ppl[ppl["K"] == k].pivot_table(index="category", columns="set", values=st)
            pr = strata.set_index("category")["pronoun"].reindex(w.index)
            for a in sets:
                for b in sets:
                    if a >= b or a not in w or b not in w:
                        continue
                    dlt = w[a] - w[b]
                    g, se = gap(dlt[pr == SHE], dlt[pr == HE])
                    arm_rows.append(
                        {
                            "K": k,
                            "stage": st,
                            "arm": a,
                            "minus": b,
                            "she_change": dlt[pr == SHE].mean(),
                            "he_change": dlt[pr == HE].mean(),
                            "gap_change": g,
                            "se": se,
                        }
                    )
    pd.DataFrame(arm_rows).round(4).to_csv(args.out / "arms.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    show = stages[stages["contrast"] == "pronoun"].pivot_table(
        index=["set", "K"], columns="stage", values="gap", sort=False
    )
    print("\nshe/her minus he/him, by stage:\n", show[[c for c in STAGES if c in show]].round(3).to_string())
    print(
        f"\nwrote people.csv (owner-only), stages.csv, matched.csv, regress.csv, photos_by_pronoun.csv, arms.csv -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
